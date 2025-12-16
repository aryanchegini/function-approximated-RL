import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RainbowDQN(nn.Module):
    def __init__(self, env_args):
        super(RainbowDQN, self).__init__()



        self.epsilon = 0.1
        self.output_dims = 6
        
    def train(self):
        pass

    def get_action(self, state):
        with torch.no_grad():
            q = self.Sequential(state)

            if np.random.uniform() > self.epsilon:
                action = q.argmax().item()
            else:
                action = np.random.randint(0, self.output_dims-1)
            
            return action



