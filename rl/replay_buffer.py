from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, seed: int = 7):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[Transition] = []
        self.priorities: list[float] = []
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done, priority: float | None = None):
        p = priority if priority is not None else (max(self.priorities, default=1.0))
        item = Transition(np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done))
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0); self.priorities.pop(0)
        self.buffer.append(item)
        self.priorities.append(max(float(p), 1e-6))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, beta: float = 0.4):
        probs = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idx]
        weights = (len(self.buffer) * probs[idx]) ** (-beta)
        weights /= weights.max()
        return samples, idx, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[int(i)] = max(float(p), 1e-6)
