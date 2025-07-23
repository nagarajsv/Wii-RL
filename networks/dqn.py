from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, 
                 n_obs:int, 
                 n_act:int):
        
        super(DQN, self).__init__()
        self.layer_1 = nn.Linear(n_obs, 128)
        self.layer_2 = nn.Linear(128, n_act)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        return self.layer_2(x)