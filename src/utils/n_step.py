"""
N-Step Return Calculator
Computes n-step returns for multi-step learning.
"""
from collections import deque
import numpy as np
from typing import Optional


class NStepBuffer:
    """
    Buffer for computing n-step returns.
    
    Stores the last n transitions and computes the n-step return:
    R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n})
    """
    
    def __init__(self, n_step: int, gamma: float):
        """
        Args:
            n_step: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[tuple]:
        """
        Add transition and compute n-step return if buffer is full.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        
        Returns:
            If buffer is full: (state, action, n_step_reward, n_step_next_state, n_step_done)
            Otherwise: None
        """
        self.buffer.append((state, action, reward, next_state, done))
        
        if len(self.buffer) < self.n_step:
            return None
        
        # Compute n-step return
        n_step_reward = 0.0
        gamma_power = 1.0
        
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += gamma_power * r
            gamma_power *= self.gamma
            
            if d:
                # Episode ended, return early
                state_0, action_0, _, _, _ = self.buffer[0]
                n_step_next_state = self.buffer[i][3]
                return (state_0, action_0, n_step_reward, n_step_next_state, True)
        
        # Full n-step return
        state_0, action_0, _, _, _ = self.buffer[0]
        _, _, _, n_step_next_state, n_step_done = self.buffer[-1]
        
        return (state_0, action_0, n_step_reward, n_step_next_state, n_step_done)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
