"""
Rainbow DQN Network Architecture
Combines Dueling DQN, Noisy Networks, and Distributional RL (C51)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .noisy_linear import NoisyLinear


class RainbowDQNNetwork(nn.Module):
    """
    Rainbow DQN Network with:
    - Convolutional feature extractor
    - Dueling architecture (value and advantage streams)
    - Noisy linear layers for exploration
    - Distributional RL (C51) for value distribution
    """
    
    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        """
        Args:
            input_channels: Number of input channels (e.g., 4 for frame stack)
            num_actions: Number of possible actions
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distribution support
            v_max: Maximum value for distribution support
        """
        super(RainbowDQNNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer(
            'atoms',
            torch.linspace(v_min, v_max, num_atoms)
        )
        
        # Convolutional feature extractor (Nature DQN architecture)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after conv layers
        # For 84x84 input: 84 -> 20 -> 9 -> 7
        self.feature_size = 64 * 7 * 7
        
        # Dueling architecture with noisy layers
        # Value stream
        self.value_fc = NoisyLinear(self.feature_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        # Advantage stream
        self.advantage_fc = NoisyLinear(self.feature_size, 512)
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Distribution over values for each action: (batch_size, num_actions, num_atoms)
        """
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)  # (batch_size, num_atoms)
        value = value.view(batch_size, 1, self.num_atoms)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)  # (batch_size, num_actions * num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine value and advantage (dueling architecture)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values from distribution.
        
        Args:
            x: Input tensor
        
        Returns:
            Q-values for each action: (batch_size, num_actions)
        """
        q_dist = self.forward(x)
        q_values = (q_dist * self.atoms).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
