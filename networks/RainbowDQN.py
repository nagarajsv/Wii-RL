import torch.nn as nn
import torch.nn.functional as F
import torch
from NoisyLinear import NoisyLinear

class RainbowDQN(nn.Module):
    def __init__(self, input_dim: tuple[int, int, int], output_dim: int, n_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        super(RainbowDQN, self).__init__()
        
        channels, height, width = input_dim
        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_out_size(input_dim)
        
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, output_dim * n_atoms)
        )
        
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
            
    def _get_conv_out_size(self, input_dim):
        channels, height, width = input_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            conv_out = self.conv_layers(dummy_input)
            return conv_out.size(1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        conv_out = self.conv_layers(x)
        
        value = self.value_stream(conv_out).view(batch_size, 1, self.n_atoms)
        advantage = self.advantage_stream(conv_out).view(batch_size, self.output_dim, self.n_atoms)
        
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def get_q_values(self, x):
        q_dist = self.forward(x)
        q_values = (q_dist * self.support).sum(dim=2)
        return q_values