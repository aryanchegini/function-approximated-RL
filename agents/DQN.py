import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from typing import Tuple
import random
import math

from configs.DQNSpaceInvadersConfig import (
    TRAINING_CONFIG,
    AGENT_CONFIG,
)



tau = AGENT_CONFIG['tau']
gamma = AGENT_CONFIG['gamma']
eps_start = AGENT_CONFIG['eps_start']
eps_end = AGENT_CONFIG['eps_end']
eps_decay = AGENT_CONFIG['eps_decay']
tau = AGENT_CONFIG['tau']
lr = AGENT_CONFIG['learning_rate']
capacity = AGENT_CONFIG['buffer_capacity']
n_actions = AGENT_CONFIG['num_actions']
input_channels = AGENT_CONFIG['input_channels']

batch_size = TRAINING_CONFIG['batch_size']

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, device, env):
        super(DQN, self).__init__()
        
        self.batch_size = batch_size
        self.env = env
        self.feature_extractor = self.create_feature_extractor()
        self.memory = ReplayMemory(capacity)
        self.device = device
        self.steps_done = 0
        self.target_net = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        ).to(device)

        self.policy_net = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        ).to(device)
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.learn_step_counter = 0

    def create_feature_extractor(self):
        self.feature_size = 64 * 7 * 7
        return nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Input channels need to be specified
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )
    
    def get_action(self, state):
        steps_done= self.steps_done
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        self.learn_step_counter += 1

        if len(self.memory) < batch_size:
                return
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

 
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()  

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return loss
    
    def forward(self, x):
        return self.policy_net(x)
              

    def save(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'online_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"Model loaded from {filepath}")

    def train_mode(self):
        """Set networks to training mode."""
        self.policy_net.train()
    
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.policy_net.eval()