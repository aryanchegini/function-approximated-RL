"""
Prioritized Experience Replay Buffer
Paper: https://arxiv.org/abs/1511.05952

Uses sum tree for efficient sampling based on TD error priorities.
"""
import numpy as np
import torch
from typing import Tuple, Dict
from collections import namedtuple


# Experience tuple
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Binary tree where parent = sum of children.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of experiences
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority (root of tree)."""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """Add experience with priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """
        Get experience by priority value.
        
        Returns:
            tree_idx: Index in tree
            priority: Priority value
            data: Experience tuple
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using Sum Tree.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which to anneal beta to 1
            epsilon: Small constant to prevent zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self.max_priority = 1.0
    
    def _get_beta(self) -> float:
        """Get current beta value (anneals from beta_start to 1)."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience with maximum priority."""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences.
        
        Returns:
            batch: Dictionary of tensors (states, actions, rewards, next_states, dones)
            indices: Tree indices for updating priorities
            weights: Importance sampling weights
        """
        batch = []
        indices = np.empty(batch_size, dtype=np.int32)
        weights = np.empty(batch_size, dtype=np.float32)
        
        # Divide priority range into segments
        priority_segment = self.tree.total() / batch_size
        beta = self._get_beta()
        
        # Calculate max weight for normalization
        min_prob = np.min(self.tree.tree[-self.tree.capacity:][self.tree.tree[-self.tree.capacity:] > 0]) / self.tree.total()
        max_weight = (min_prob * self.tree.n_entries) ** (-beta)
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(value)
            
            # Calculate importance sampling weight
            sampling_prob = priority / self.tree.total()
            weight = (sampling_prob * self.tree.n_entries) ** (-beta)
            weights[i] = weight / max_weight
            
            indices[i] = idx
            batch.append(data)
        
        self.frame += 1
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor(np.array([e.action for e in batch]))
        rewards = torch.FloatTensor(np.array([e.reward for e in batch]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor(np.array([e.done for e in batch]))
        weights = torch.FloatTensor(weights)
        
        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights
        }
        
        return batch_dict, indices, weights.numpy()
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.tree.n_entries
