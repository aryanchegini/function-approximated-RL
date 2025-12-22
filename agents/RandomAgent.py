import torch

class randomAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def get_action(self, state=None):

        return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def save(self, filepath: str):
        """Save model checkpoint- placeholder."""
        pass
    
    def load(self, filepath: str):
        """Load model checkpoint - placeholder."""
        pass