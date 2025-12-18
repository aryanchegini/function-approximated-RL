import numpy as np
from typing import NamedTuple, List, Tuple, Optional


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritisedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.eps = float(eps)

        self.rng = np.random.default_rng(seed)

        self.data: List[Optional[Experience]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.frame = 1
        self.max_priority = 1.0

    def _beta(self) -> float:
        if self.beta_frames <= 0:
            return 1.0
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done, priority: Optional[float] = None):
        if priority is None:
            priority = self.max_priority

        self.data[self.pos] = Experience(state, action, float(reward), next_state, bool(done))
        self.priorities[self.pos] = float(priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        self.max_priority = max(self.max_priority, float(priority))

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        assert self.size > 0, "Cannot sample from an empty buffer."

        prios = self.priorities[:self.size]
        scaled = (prios + self.eps) ** self.alpha
        probs = scaled / scaled.sum()

        replace = self.size < batch_size
        indices = self.rng.choice(self.size, size=batch_size, replace=replace, p=probs)

        beta = self._beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= (weights.max() + 1e-8)

        batch = [self.data[i] for i in indices] 
        self.frame += 1
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        for idx, p in zip(indices, priorities):
            p = float(abs(p) + self.eps)
            self.priorities[int(idx)] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size
