import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from typing import Tuple

from layers.NoisyNets import NoisyLinear
from layers.DuelingNets import DuelingNetwork
from configs.SpaceInvadersConfig import ( AGENT_CONFIG, DEVICE, TRAINING_CONFIG )

class RainbowDQN(nn.Module):
    def __init__(self, config = AGENT_CONFIG, device = DEVICE):
        super(RainbowDQN, self).__init__()
        self.device = device

        self.output_dims = config['num_actions']
        self.v_min = config['v_min']
        self.v_max = config['v_max']
        self.num_atoms = config['num_atoms']
        self.batch_size = config['batch_size']
        
        # Missing attributes
        self.gamma = config['gamma']
        self.n_step = config['n_step']
        self.target_update_freq = config['target_update_freq']
        self.config = config

        self.feature_extractor = self.create_feature_extractor()
        self.dueling_net = DuelingNetwork(self.feature_size, self.num_atoms, self.batch_size, self.output_dims, self.num_atoms)

        self.online = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            self.dueling_net
        ).to(self.device)

        #torch.compile(self.online)

        # Create a target network
        self.target = nn.Sequential(
            copy.deepcopy(self.feature_extractor),
            copy.deepcopy(nn.Flatten()),
            copy.deepcopy(self.dueling_net)
        ).to(self.device)

        #torch.compile(self.target)

        self.get_online_q_values = self.online[2].get_q_values
        self.get_target_q_values = self.target[2].get_q_values

        self.optimizer = optim.Adam(self.online.parameters(), lr=config['learning_rate'])

        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.learn_step_counter = 0


    def get_action(self, state):
        with torch.no_grad():
            q = self.online(state) # distibution of propability for each action-reward
            q_values = (q * self.atoms.view(1, 1, -1)).sum(dim=-1) # utility of each action
            action = q_values.argmax(dim=1).item()
  
            return action
        
    def create_feature_extractor(self):
        self.feature_size = 64 * 7 * 7
        return nn.Sequential(
            nn.Conv2d(self.config['input_channels'], 32, kernel_size=8, stride=4),  # Input channels need to be specified
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )
              

    def learn(
        self,
        batch: dict,
        indices: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Update network weights using a batch of experiences.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
            indices: Indices for prioritized replay
            weights: Importance sampling weights
        
        Returns:
            loss: Training loss
            priorities: New priorities for replay buffer
        """

        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        weights = batch['weights'].to(self.device)
        
        batch_size = states.size(0)

        # Current distribution (online network)
        current_dist = self.online(states)
        # Select distribution for chosen actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        current_dist = current_dist.gather(1, actions_expanded).squeeze(1)
        
        with torch.no_grad():
            # Double DQN: use online network to select actions
            next_q_dist = self.online(next_states)
            next_q_values = self.get_online_q_values(next_q_dist)

            # features = self.online[0:2](next_states)  # CNN + Flatten
            # next_q_values = self.online[2].get_q_values(features)

            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate actions
            next_dist = self.target(next_states)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
            next_dist = next_dist.gather(1, next_actions_expanded).squeeze(1)
            
            # Compute projected distribution for n-step returns
            gamma_n = self.gamma ** self.n_step
            rewards_expanded = rewards.unsqueeze(1).expand_as(next_dist)
            dones_expanded = dones.unsqueeze(1).expand_as(next_dist)
            
            # Compute Bellman update: r + gamma^n * z
            Tz = rewards_expanded + (1 - dones_expanded) * gamma_n * self.atoms.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            
            # Project onto support
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge cases
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            
            # Distribute probability
            target_dist = torch.zeros_like(next_dist)
            offset = torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size
            ).long().unsqueeze(1).expand(batch_size, self.num_atoms).to(self.device)
            
            target_dist.view(-1).index_add_(
                0,
                (l + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0,
                (u + offset).view(-1),
                (next_dist * (b - l.float())).view(-1)
            )
        
        # Cross-entropy loss
        log_current_dist = torch.log(current_dist + 1e-8)
        loss = -(target_dist * log_current_dist).sum(dim=1)
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        
        self.online[2].reset_noise()
        self.target[2].reset_noise()


        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        
        # Calculate new priorities (TD errors)
        priorities = loss.detach().cpu().numpy()
        
        return weighted_loss.item(), priorities
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'online_net': self.online.state_dict(),
            'target_net': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online.load_state_dict(checkpoint['online_net'])
        self.target.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"Model loaded from {filepath}")
    
    def train_mode(self):
        """Set networks to training mode."""
        self.online.train()
    
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.online.eval()

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        self.online[2].reset_noise()
        self.target[2].reset_noise()   
