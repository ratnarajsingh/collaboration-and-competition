import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import soft_update

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class Actor(nn.Module):
    """
    Defines the Actor class
    """
    def __init__(self,state_size, action_size, l1 = 32,l2 = 32):
        """
        Initializes the Actor network
        Params
        ======
            state_size  : Dimension of each state
            action_size : Dimension of each action
            l1,l2       : Layer sizes
        """
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_size,l1))
        self.fc2 = layer_init(nn.Linear(l1,l2))
        self.fc3 = layer_init(nn.Linear(l2,action_size),1e-3)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """
    Defines the Critic network
    """
    def __init__(self, full_state_size,action_size,l1=64,l2=64):
        """
        Initializes the Actor network
        Params
        ======
            state_size  : Dimension of each state
            action_size : Dimension of each action
            l1,l2       : Layer sizes
        """
        super().__init__()
        # concatenate full states and actions into a single input to produce Q value
        self.fc1 = layer_init(nn.Linear(full_state_size+action_size,l1))
        self.fc2 = layer_init(nn.Linear(l1,l2))
        self.fc3 = layer_init(nn.Linear(l2,1))
        
    def forward(self,full_state,action):
        concatenated = torch.cat((full_state, action), dim=-1)
        x = F.relu(self.fc1(concatenated))
        x = F.relu(self.fc2(x))
        return self.fc3(x)