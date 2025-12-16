import torch 
import torch.nn as nn
from .NoisyNets import NoisyLinear


class DuelingNetwork(nn.Module):
    def __init__(self, feature_size, num_atoms, batch_size=32, num_actions=6, atoms=51):
        super(DuelingNetwork, self).__init__()
        print("Feature size in DuelingNetwork:", feature_size)
        self.feature_size = feature_size
        self.num_atoms = num_atoms
        self.num_actions = num_actions  
        self.register_buffer("atoms", torch.linspace(-10, 10, atoms))

        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.num_atoms),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.num_actions * self.num_atoms) ,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network."""
        batch_size = x.size(0) 
        value = self.value_stream(x).view(batch_size, 1, self.num_atoms)  # Shape: (batch_size, 1, num_atoms)
        advantage = self.advantage_stream(x).view(batch_size, self.num_actions, self.num_atoms)  # Shape: (batch_size, num_actions, num_atoms)
        # Combine value and advantage to get Q-values
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_dist = nn.functional.softmax(q_dist, dim=-1)  # Apply softmax to get probabilities

        return q_dist
    
    def get_q_values(self, q_dist):
        """Get Q-values by taking the expectation over the distribution."""
        
        q_values = (q_dist * self.atoms).sum(dim=-1)  # Shape: (batch_size, num_actions)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    


