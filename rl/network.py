from __future__ import annotations

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Dueling MLP used by Double-DQN."""

    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        value = self.value(z)
        adv = self.advantage(z)
        return value + adv - adv.mean(dim=1, keepdim=True)
