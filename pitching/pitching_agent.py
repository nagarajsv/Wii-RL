from networks.dqn import DQN
from networks.replay_memory import ReplayMemory, Transition
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
from itertools import count
import random
import math
from baseball import PITCHING_TYPES, OUTS_ADDRESS, OUR_SCORE_ADDRESS
from dolphin import memory #type: ignore

class Agent:
    
    def __init__(self,
                 env,
                 n_obs: int,
                 n_act: int,
                 model_name: str,
                 batch_size:int=128, 
                 gamma:float=0.99, 
                 eps_start:float=0.9, 
                 eps_end:float=0.05, 
                 eps_decay:float=1000, 
                 tau:float=0.005, 
                 lr:float=1e-4):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.model_name = model_name
        
        self.policy_net = DQN(n_obs, n_act).to(device)
        self.target_net = DQN(n_obs, n_act).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
        self.memory = ReplayMemory(10000)
        
        self.steps_done = 0
        
        self.episode_durations = []
        
        self.total_reward = 0
        
        self.burnin = 3000
        
    async def learn(self, num_episodes: int = 600):
        for i_episode in count():
            state, _ = await self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state, True)
                print(f"action: {PITCHING_TYPES[action.item()]} ({action})")
                observation, reward, terminated, truncated, _ = await self.env.step(action.item())
                memory.write_u32(OUTS_ADDRESS, 0)
                self.total_reward += reward
                print(f"total reward: {self.total_reward}")
                reward = torch.tensor([reward], device=device)

                if(terminated):
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated:
                    self.episode_durations.append(t + 1)
                    torch.save(self.policy_net.state_dict(), f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\pitching\\{self.model_name}_{i_episode}.pth")
                    print(f"saved episode {i_episode}")
                    break
                
    def optimize_model(self):
        if len(self.memory) < max(self.batch_size, self.burnin):
            print(f"Step: {self.steps_done}")
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def select_action(self, state, training:bool):
        if training:
            self.steps_done += 1
            if(len(self.memory) < self.burnin):
                return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
                
            print(f"epsilon value: {eps_threshold}")
            
            if sample <= eps_threshold:
                return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
            else:
                with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
                
    def load_model(self, path: str) -> None:
        state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        # self.steps_done = 50s00
        
    async def play(self) -> None:
        state, _ = await self.env.reset()
        memory.write_u32(OUR_SCORE_ADDRESS, 0)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = self.select_action(state, False)
            observation, reward, terminated, truncated, _ = await self.env.step(action.item())
            
            if(terminated):
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
            state = next_state
            
            if terminated or truncated:
                print(f"Pitching completed")
                return