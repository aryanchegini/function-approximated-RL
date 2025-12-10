"""
Rainbow DQN Agent
Combines all Rainbow improvements:
- Double Q-Learning
- Prioritized Experience Replay
- Dueling Networks
- Multi-step Learning (n-step returns)
- Distributional RL (C51)
- Noisy Networks
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import os

from ..networks.rainbow_network import RainbowDQNNetwork


class RainbowDQNAgent:
    """Rainbow DQN Agent with all improvements."""
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        num_actions: int,
        device: str = 'cuda',
        # Network parameters
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # Learning parameters
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        # Multi-step learning
        n_step: int = 3,
        # Target network update
        target_update_freq: int = 1000,
    ):
        """
        Args:
            state_shape: Shape of state (channels, height, width)
            num_actions: Number of possible actions
            device: Device to use ('cuda' or 'cpu')
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distribution support
            v_max: Maximum value for distribution support
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            n_step: Number of steps for n-step returns
            target_update_freq: Frequency of target network updates (in steps)
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Create networks
        self.online_net = RainbowDQNNetwork(
            input_channels=state_shape[0],
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(self.device)
        
        self.target_net = RainbowDQNNetwork(
            input_channels=state_shape[0],
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate, eps=1.5e-4)
        
        # Support for distributional RL
        self.atoms = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Training step counter
        self.learn_step_counter = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using the current policy (greedy with noisy networks).
        
        Args:
            state: Current state
        
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net.get_q_values(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action
    
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
        current_dist = self.online_net(states)
        # Select distribution for chosen actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        current_dist = current_dist.gather(1, actions_expanded).squeeze(1)
        
        with torch.no_grad():
            # Double DQN: use online network to select actions
            next_q_values = self.online_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate actions
            next_dist = self.target_net(next_states)
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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Reset noise in noisy layers
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Calculate new priorities (TD errors)
        priorities = loss.detach().cpu().numpy()
        
        return weighted_loss.item(), priorities
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['learn_step_counter']
        print(f"Model loaded from {filepath}")
    
    def train_mode(self):
        """Set networks to training mode."""
        self.online_net.train()
    
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.online_net.eval()
