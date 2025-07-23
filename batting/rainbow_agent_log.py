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
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

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
                 lr: float = 0.0001,
                 batch_size: int = 32,
                 eps_start: float = 0.9,
                 eps_end: float = 0.1,
                 eps_decay: float = 0.995,
                 eps_steps: int = 50_000,
                 total_steps: int = 200_000,
                 log_dir: str = None) -> None:
        
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
        self.eps_steps = eps_steps
        self.total_steps = total_steps
        
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
        
        # TensorBoard logging setup
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/{model_name}_{timestamp}"
        
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_hits = []  # Track successful hits per episode
        self.episode_steps = []
        self.current_episode_hits = 0
        self.recent_losses = []
        self.recent_q_values = []
        self.current_episode_swings = 0
        self.episode_swings = []
        
        print(f"TensorBoard logging to: {log_dir}")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            if training:
                if self.steps_done < self.eps_steps and random.random() < self.eps:
                    action = random.randint(0, 1)
                    self.eps = max(self.eps * self.eps_decay, self.eps_end)
                else:
                    state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                    q_values = self.online_net.get_q_values(state_tensor)
                    
                    # Log Q-values for monitoring
                    if training:
                        self.recent_q_values.append(q_values.mean().item())
                        
                    action = torch.argmax(q_values, dim=1).item()
            else:
                self.online_net.eval()
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.online_net.get_q_values(state_tensor)
                self.online_net.train()    
                action = torch.argmax(q_values, dim=1).item()
                
        self.steps_done += 1
        if action == 1:
            self.current_episode_swings += 1
        return action
    
    def calculate_n_step_returns(self, rewards: list, next_state: np.ndarray, done: bool) -> tuple[float, np.ndarray, bool]:
        n_step_reward = 0
        gamma_pow = 1
        
        for i in range(len(rewards)):
            n_step_reward += gamma_pow * rewards[i]
            gamma_pow *= self.gamma
            
        return n_step_reward, next_state, done
    
    def cache(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, done: bool) -> None:
        # Track hits (assuming positive rewards indicate successful hits)
        if reward > 0:
            self.current_episode_hits += 1
            
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
                
                done_mask = done.float().unsqueeze(1)
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
    
    def log_episode_metrics(self, episode: int, episode_reward: float, episode_steps: int):
        """Log episode-level metrics to TensorBoard"""
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Steps', episode_steps, episode)
        self.writer.add_scalar('Episode/Hits', self.current_episode_hits, episode)
        
        # Calculate hit rate
        hit_rate = self.current_episode_hits / max(episode_steps, 1)
        self.writer.add_scalar('Episode/Hit_Rate', hit_rate, episode)
        
        swing_rate = self.current_episode_swings / max(episode_steps, 1)
        self.writer.add_scalar('Episode/Swing_Rate', swing_rate, episode)
        
        # Log exploration metrics
        self.writer.add_scalar('Training/Epsilon', self.eps, episode)
        self.writer.add_scalar('Training/Beta', self.beta, episode)
        
        # Log moving averages
        self.episode_rewards.append(episode_reward)
        self.episode_hits.append(self.current_episode_hits)
        self.episode_steps.append(episode_steps)
        self.episode_swings.append(self.current_episode_swings)
        
        if len(self.episode_rewards) >= 10:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_hits = np.mean(self.episode_hits[-10:])
            avg_hit_rate = np.mean([h/max(s, 1) for h, s in zip(self.episode_hits[-10:], self.episode_steps[-10:])])
            avg_swings = np.mean(self.episode_swings[-10:])
            avg_swing_rate = np.mean([h/max(s, 1) for h, s in zip(self.episode_swings[-10:], self.episode_steps[-10:])])
            
            self.writer.add_scalar('Episode/Reward_MA10', avg_reward, episode)
            self.writer.add_scalar('Episode/Hits_MA10', avg_hits, episode)
            self.writer.add_scalar('Episode/Hit_Rate_MA10', avg_hit_rate, episode)
            self.writer.add_scalar('Episode/Swings_MA10', avg_swings, episode)
            self.writer.add_scalar('Episode/Swing_rate_MA10', avg_swing_rate, episode)
        
        if len(self.episode_rewards) >= 100:
            avg_reward_100 = np.mean(self.episode_rewards[-100:])
            avg_hit_rate_100 = np.mean([h/max(s, 1) for h, s in zip(self.episode_hits[-100:], self.episode_steps[-100:])])
            avg_swing_rate_100 = np.mean([h/max(s, 1) for h, s in zip(self.episode_swings[-100:], self.episode_steps[-100:])])
            
            self.writer.add_scalar('Episode/Reward_MA100', avg_reward_100, episode)
            self.writer.add_scalar('Episode/Hit_Rate_MA100', avg_hit_rate_100, episode)
            self.writer.add_scalar('Episode/Swing_rate_MA100', avg_swing_rate_100, episode)
        
        print(f"Episode {episode}: Reward={episode_reward:.3f}, Hits={self.current_episode_hits}, Hit_Rate={hit_rate:.3f}")
        
        # Reset episode counters
        self.current_episode_hits = 0
    
    def log_training_metrics(self, loss_value: float):
        """Log training-specific metrics to TensorBoard"""
        if loss_value is not None:
            self.writer.add_scalar('Training/Loss', loss_value, self.steps_done)
            self.recent_losses.append(loss_value)
        
        if len(self.recent_q_values) > 0:
            avg_q_value = np.mean(self.recent_q_values)
            self.writer.add_scalar('Training/Q_Value_Mean', avg_q_value, self.steps_done)
            self.recent_q_values = []  # Reset for next batch
        
        # Log moving average of loss
        if len(self.recent_losses) >= 100:
            avg_loss = np.mean(self.recent_losses[-100:])
            self.writer.add_scalar('Training/Loss_MA100', avg_loss, self.steps_done)
        
        # Log memory buffer size
        self.writer.add_scalar('Training/Memory_Size', len(self.memory), self.steps_done)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/Learning_Rate', current_lr, self.steps_done)

    async def learn(self):
        for e in count():
            state, _ = await self.env.reset()
            self.total_reward = 0
            episode_steps = 0
            
            self.online_net.reset_noise()
            self.target_net.reset_noise()
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, _, _ = await self.env.step(action)
                
                memory.write_u32(OUTS_ADDRESS, 0)
                self.cache(state, next_state, action, reward, done)
                self.total_reward += reward
                episode_steps += 1
                
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
                    self.recent_losses.append(loss_value)
                    
                    # Log training metrics
                    self.log_training_metrics(loss_value)
                    
                else:
                    loss_value = None
                
                print(f"loss: {loss_value}, beta: {self.beta}")
                
                state = next_state
                
                if done:
                    break
            
            # Log episode metrics
            self.log_episode_metrics(e, self.total_reward, episode_steps)
            
            if e % 25 == 0:
                model_path = f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\{self.model_name}_rainbow_ep{e}.pth"
                torch.save({
                    'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'step': self.steps_done,
                    'beta': self.beta,
                    'episode': e,
                    'total_reward': self.total_reward
                }, model_path)
                
                # Log model save event
                self.writer.add_text('Model/Checkpoint', f'Model saved at episode {e}', e)
            
            if self.steps_done >= self.total_steps:
                print(f"training complete after {e} episodes")
                final_model_path = f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\{self.model_name}_rainbow_ep_final.pth"
                torch.save({
                    'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'step': self.steps_done,
                    'beta': self.beta,
                    'episode': e,
                    'total_reward': self.total_reward
                }, final_model_path)
                
                self.writer.add_text('Model/Final', f'Final model saved after {e} episodes', e)
                self.writer.close()  # Close TensorBoard writer
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
        
        # Load additional training metrics if available
        if "episode" in checkpoint:
            print(f"Loaded model from episode {checkpoint['episode']}")
        if "total_reward" in checkpoint:
            print(f"Last episode reward: {checkpoint['total_reward']}")
        
    async def play(self, episodes: int = 10):
        """
        Evaluate the loaded model without training.
        Loads savestate 3 and runs the agent in evaluation mode.
        """
        print(f"Starting evaluation for {episodes} episodes...")
        
        # Create evaluation writer
        eval_writer = SummaryWriter(f"{self.log_dir}_evaluation")
        
        episode_rewards = []
        episode_hits = []
        
        for e in range(episodes):
            state, _ = await self.env.test_state()
            total_reward = 0
            step_count = 0
            hits = 0
            
            # Set networks to evaluation mode
            self.online_net.eval()
            
            print(f"\nEpisode {e + 1}/{episodes}")
            
            while True:
                # Select action without training (no exploration)
                action = self.select_action(state, training=False)
                next_state, reward, done, _, _ = await self.env.step(action)
                
                # Reset outs (if needed for your environment)
                memory.write_u32(OUTS_ADDRESS, 0)
                
                if reward > 0:
                    hits += 1
                
                total_reward += reward
                step_count += 1
                
                print(f"Step {step_count}: Action {action}, Reward {reward:.3f}, Total Reward: {total_reward:.3f}")
                
                state = next_state
                
                # if done:
                #     break
            
            episode_rewards.append(total_reward)
            episode_hits.append(hits)
            hit_rate = hits / max(step_count, 1)
            
            # Log evaluation metrics
            eval_writer.add_scalar('Eval/Episode_Reward', total_reward, e)
            eval_writer.add_scalar('Eval/Episode_Hits', hits, e)
            eval_writer.add_scalar('Eval/Episode_Hit_Rate', hit_rate, e)
            eval_writer.add_scalar('Eval/Episode_Steps', step_count, e)
            
            print(f"Episode {e + 1} completed - Total Reward: {total_reward:.3f}, Hits: {hits}, Hit Rate: {hit_rate:.3f}")
        
        # Log summary statistics
        avg_reward = np.mean(episode_rewards)
        avg_hits = np.mean(episode_hits)
        avg_hit_rate = np.mean([h/max(1, len(episode_rewards)) for h in episode_hits])
        
        eval_writer.add_scalar('Eval/Average_Reward', avg_reward, episodes)
        eval_writer.add_scalar('Eval/Average_Hits', avg_hits, episodes)
        eval_writer.add_scalar('Eval/Average_Hit_Rate', avg_hit_rate, episodes)
        
        print(f"\nEvaluation Summary:")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Hits: {avg_hits:.3f}")
        print(f"Average Hit Rate: {avg_hit_rate:.3f}")
        
        eval_writer.close()
        return episode_rewards, episode_hits
    
    def close_logger(self):
        """Close the TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()