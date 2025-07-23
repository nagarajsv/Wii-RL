import torch
import torch.nn.functional as F
from networks.RainbowDQN import RainbowDQN
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from dolphin import memory, savestate, emulation #type: ignore
from baseball import OUTS_ADDRESS
from itertools import count
import random

class Agent:
    def __init__(self,
                 env,
                 state_dim: tuple[int, int, int], 
                 action_dim: int, 
                 model_name: str,
                 n_atoms: int = 51,
                 v_min: float = -10,
                 v_max: float = 10,
                 n_steps: int = 3,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 gamma: float = 0.99,
                 lr: float = 0.00025,
                 batch_size: int = 32,
                 eps_start: float = 0.9,
                 eps_end: float = 0.1,
                 eps_decay: float = 0.995) -> None:
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.online_net = RainbowDQN(self.state_dim, self.action_dim, n_atoms, v_min, v_max).to(self.device)
        self.target_net = RainbowDQN(self.state_dim, self.action_dim, n_atoms, v_min, v_max).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        for p in self.target_net.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        self.steps_done = 0
        self.batch_size = batch_size
        self.burnin = 1e4
        self.sync_every = 1e4
        self.learn_every = 4 
        self.total_reward = 0
        
        self.memory = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            storage=LazyMemmapStorage(100000, device="cpu"),
            batch_size=batch_size
        )
        
        self.n_step_buffer = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            if training:
                if self.steps_done < 50000 and random.random() < self.eps:
                    action = random.randint(0, 1)
                    self.eps = max(self.eps * self.eps_decay, self.eps_end)
                else:
                    state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                    q_values = self.online_net.get_q_values(state)        
                    action = torch.argmax(q_values, dim=1).item()
            else:
                self.online_net.eval()
                q_values = self.online_net.get_q_values(state)
                self.online_net.train()    
                action = torch.argmax(q_values, dim=1).item()
                
        
        self.steps_done += 1
        return action
    
    def calculate_n_step_returns(self, rewards: list, next_state: np.ndarray, done: bool) -> tuple[float, np.ndarray, bool]:
        n_step_reward = 0
        gamma_pow = 1
        
        for i in range(len(rewards)):
            n_step_reward += gamma_pow * rewards[i]
            gamma_pow *= self.gamma
            
        return n_step_reward, next_state, done
    
    def cache(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, done: bool) -> None:
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) >= self.n_steps or done:
            while len(self.n_step_buffer) > 0:
                n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]
                
                rewards = [exp[2] for exp in self.n_step_buffer]
                final_state = self.n_step_buffer[-1][3]
                final_done = self.n_step_buffer[-1][4]
                
                n_step_reward, n_step_next_state, n_step_done = self.calculate_n_step_returns(
                    rewards, final_state, final_done
                )
                
                state_tensor = torch.tensor(n_step_state, dtype=torch.float32)
                next_state_tensor = torch.tensor(n_step_next_state, dtype=torch.float32)
                action_tensor = torch.tensor([n_step_action])
                reward_tensor = torch.tensor([n_step_reward])
                done_tensor = torch.tensor([n_step_done])
                
                self.memory.add(TensorDict({
                    "state": state_tensor,
                    "next_state": next_state_tensor,
                    "action": action_tensor,
                    "reward": reward_tensor,
                    "done": done_tensor
                }, batch_size=[]))
                
                self.n_step_buffer.pop(0)
                
                if len(self.n_step_buffer) == self.n_steps - 1:
                    break
    
    def recall(self):
        batch = self.memory.sample()
        batch = batch.to(self.device)
        
        state = batch["state"]
        next_state = batch["next_state"]
        action = batch["action"].squeeze()
        reward = batch["reward"].squeeze()
        done = batch["done"].squeeze()
        
        weights = batch.get("_weight", torch.ones(self.batch_size, device=self.device))
        indices = batch.get("index", torch.arange(self.batch_size, device=self.device))
        
        return state, next_state, action, reward, done, weights, indices
    
    def compute_distributional_loss(self, state, action, reward, next_state, done, weights):
        batch_size = state.size(0)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            current_q_dist = self.online_net(state)
            current_q_dist = current_q_dist.gather(1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
            
            with torch.no_grad():
                next_q_values = self.online_net.get_q_values(next_state)
                next_actions = next_q_values.argmax(dim=1)
                
                next_q_dist = self.target_net(next_state)
                next_q_dist = next_q_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
                
                # target_support = reward.unsqueeze(1) + (self.gamma ** self.n_steps) * self.support.unsqueeze(0) * (1 - done.unsqueeze(1))
                
                done_mask = done.float().unsqueeze(1)  # Convert bool to float
                target_support = reward.unsqueeze(1) + (self.gamma ** self.n_steps) * self.support.unsqueeze(0) * (1 - done_mask)
                
                target_support = target_support.clamp(self.v_min, self.v_max)
                
                b = (target_support - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                target_q_dist = torch.zeros_like(next_q_dist)
                
                for i in range(batch_size):
                    for j in range(self.n_atoms):
                        l_idx = l[i, j]
                        u_idx = u[i, j]
                        
                        if l_idx == u_idx:
                            target_q_dist[i, l_idx] += next_q_dist[i, j]
                        else:
                            target_q_dist[i, l_idx] += next_q_dist[i, j] * (u[i, j] - b[i, j])
                            target_q_dist[i, u_idx] += next_q_dist[i, j] * (b[i, j] - l[i, j])
                            
        current_q_dist = current_q_dist.float()
        target_q_dist = target_q_dist.float()
        
        loss = -torch.sum(target_q_dist * torch.log(current_q_dist + 1e-8), dim=1)
        
        weighted_loss = weights * loss
        
        return weighted_loss.mean(), loss.detach()
    
    def update_priorities(self, indices, td_errors):
        priorities = (td_errors + 1e-6) ** self.alpha
        self.memory.update_priority(indices, priorities)
    
    async def learn(self):
        for e in count():
            state, _ = await self.env.reset()
            self.total_reward = 0
            
            self.online_net.reset_noise()
            self.target_net.reset_noise()
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, _, _ = await self.env.step(action)
                
                memory.write_u32(OUTS_ADDRESS, 0)
                self.cache(state, next_state, action, reward, done)
                self.total_reward += reward
                
                print(f"episode: {e}, step: {self.steps_done}, action: {action}, total_reward: {self.total_reward}")
                
                if self.steps_done % self.sync_every == 0:
                    self.sync_Q_target()
                    
                if self.steps_done >= self.burnin and self.steps_done % self.learn_every == 0:
                    state_batch, next_state_batch, action_batch, reward_batch, done_batch, weights, indices = self.recall()
                    
                    loss, td_errors = self.compute_distributional_loss(
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights
                    )
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
                    
                    self.optimizer.step()
                    
                    self.update_priorities(indices, td_errors)
                    
                    self.beta = min(1.0, self.beta + self.beta_increment)
                    self.memory.beta = self.beta
                    
                    self.online_net.reset_noise()
                    self.target_net.reset_noise()
                    
                    loss_value = loss.item()
                    q_value = None
                else:
                    loss_value = None
                    q_value = None
                
                print(f"loss: {loss_value}, q: {q_value}, beta: {self.beta}")
                
                state = next_state
                
                if done:
                    break
                    
            if e % 25 == 0:
                torch.save({
                    'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'step': self.steps_done,
                    'beta': self.beta
                }, f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\{self.model_name}_rainbow_ep{e}.pth")
            
            if self.steps_done >= 200_000:
                print(f"training complete after {e} episodes")
                torch.save({
                    'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'step': self.steps_done,
                    'beta': self.beta
                }, f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\{self.model_name}_rainbow_ep_final.pth")
                emulation.reset()
                break
                
    def sync_Q_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.steps_done = checkpoint["step"]
        self.beta = checkpoint.get("beta", 0.4)
        
    async def play(self, episodes: int = 10):
        """
        Evaluate the loaded model without training.
        Loads savestate 3 and runs the agent in evaluation mode.
        """
        print(f"Starting evaluation for {episodes} episodes...")
        
        # Load savestate 3
        
        
        episode_rewards = []
        
        for e in range(episodes):
            state, _ = await self.env.test_state()
            # state, _ = await self.env.reset()
            total_reward = 0
            step_count = 0
            
            # Set networks to evaluation mode
            self.online_net.eval()
            
            print(f"\nEpisode {e + 1}/{episodes}")
            
            while True:
                # Select action without training (no exploration)
                action = self.select_action(state, training=False)
                next_state, reward, done, _, _ = await self.env.step(action)
                
                # Reset outs (if needed for your environment)
                memory.write_u32(OUTS_ADDRESS, 0)
                
                total_reward += reward
                step_count += 1
                
                print(f"Step {step_count}: Action {action}, Reward {reward:.3f}, Total Reward: {total_reward:.3f}")
                
                state = next_state
                
                # if done:
                    # break
            
            episode_rewards.append(total_reward)
            print(f"Episode {e + 1} completed - Total Reward: {total_reward:.3f}")