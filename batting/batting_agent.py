import torch
from networks.conv_dqn import ConvDQN
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage, LazyMemmapStorage
from dolphin import memory # type: ignore
from baseball import OUTS_ADDRESS
from itertools import count

class BattingAgent:
    def __init__(self,
                 env,
                 state_dim: tuple[int, int, int], 
                 action_dim: int, 
                 model_name: str,
                 exploration_rate: float = 1,
                 exploration_rate_decay: float = 0.99999,
                 exploration_rate_min: float = 0.1,
                 gamma: float = 0.9,
                 lr: float = 0.00025,
                 batch_size: int = 32) -> None:
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.gamma = gamma
        self.net = ConvDQN(self.state_dim, self.action_dim).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.steps_done = 0
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device="cpu"))
        self.batch_size = batch_size
        self.burnin = 1e4
        self.sync_every = 1e4
        self.learn_every = 1
        self.total_reward = 0
        self.last_action = 0
        # self.infer_queue = queue.Queue()
        # self.result_queue = queue.Queue()
        # self.infer_thread = Thread(target=self._inference_worker, daemon=True)
        # self.infer_thread.start()
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if(np.random.rand() < self.exploration_rate and training):
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                action_values = self.net(state, "online")
                action = torch.argmax(action_values, axis=1).item()
            
        if self.steps_done >= self.burnin:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        self.steps_done += 1
            
        return action
        
    async def learn(self):
        episodes = 600
        for e in count():
            # print(f"enviornment reset")
            state, _ = await self.env.reset()
            # print(f"reset done")
            self.total_reward = 0
            while True:
                # print(f"in while true loop")
                action = self.select_action(state)
                # print(f"action selected")
                next_state, reward, done, _, _ = await self.env.step(action)
                # time.sleep(0.001)
                # print(f"step done")
                memory.write_u32(OUTS_ADDRESS, 0)
                # print(f"memory write done")
                self.cache(state, next_state, action, reward, done)
                # print(f"cache done")
                # state = next_state
                self.total_reward += reward
                
                print(f"episode: {e}, step: {self.steps_done}, action: {action}, total_reward: {self.total_reward}")
                
                if self.steps_done % self.sync_every == 0:
                    # print(f"sync q starting")
                    self.sync_Q_target()
                    # print(f"sync q target done")
                    
                if self.steps_done >= self.burnin and self.steps_done % self.learn_every == 0:
                    # print(f"recall starting")
                    batch_state, batch_next_state, batch_action, batch_reward, batch_done = self.recall()
                    # print(f"recall done")
                    
                    td_est = self.td_estimate(batch_state, batch_action)
                    # print(f"td estimate calculated")
                    td_targ = self.td_target(batch_next_state, batch_reward, batch_done)
                    # print(f"td target calculated")
                    
                    loss = self.update_Q_online(td_est, td_targ)
                    # print(f"update q online done")
                    
                    q = td_est.mean().item()
                else:
                    # print(f"in burn in period...")
                    loss = None
                    q = None
                
                print(f"loss: {loss}, q: {q}, exploration rate: {self.exploration_rate}")
                
                state = next_state
                
                if done:
                    break
            if e % 25 == 0:
                torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, step=self.steps_done), 
                           f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\{self.model_name}_ep{e}.pth"
                           )
                
    def cache(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, done: bool) -> None:
        state = torch.tensor(state, dtype=torch.float32 )  # CPU storage
        next_state = torch.tensor(next_state, dtype=torch.float32 )
        action = torch.tensor([action] )
        reward = torch.tensor([reward] )
        done = torch.tensor([done] )
        self.memory.add(TensorDict({
            "state": state, 
            "next_state": next_state, 
            "action": action, 
            "reward": reward, 
            "done": done
        }, batch_size=[]))
        
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")
        return current_Q.gather(1, action.unsqueeze(1)).squeeze(1)
    
    @torch.no_grad()
    def td_target(self, next_state, reward, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
        
    def load_model(self, path: str):
        dict = torch.load(path)
        self.net.load_state_dict(dict["model"])
        self.exploration_rate = dict["exploration_rate"]
        self.steps_done = dict["step"]