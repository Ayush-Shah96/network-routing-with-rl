from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from torch import optim
torch.set_num_threads(1)

from rl.network import DuelingQNetwork
from rl.replay_buffer import PrioritizedReplayBuffer


class DoubleDQNAgent:
    """Research-grade DQN: Double-DQN target, dueling head, prioritized replay, masking."""

    def __init__(self, state_size: int, action_size: int, seed: int = 7):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.state_size = state_size; self.action_size = action_size
        self.device = torch.device("cpu")
        self.policy = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=1e-3, weight_decay=1e-5)
        self.buffer = PrioritizedReplayBuffer(seed=seed)
        self.gamma = 0.985
        self.batch_size = 64
        self.target_every = 250
        self.updates = 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(x).squeeze(0).cpu().numpy()

    def act(self, state, epsilon=0.03, valid_actions=None) -> int:
        candidates = list(valid_actions) if valid_actions is not None else list(range(self.action_size))
        if not candidates:
            return 0
        if random.random() < epsilon:
            return random.choice(candidates)
        q = self.q_values(state)
        return int(max(candidates, key=lambda a: q[a]))

    def remember(self, *args):
        self.buffer.add(*args)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch, indices, weights = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        current_q = self.policy(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1 - dones)
        td = target - current_q
        loss = (weights_t * torch.nn.functional.smooth_l1_loss(current_q, target, reduction="none")).mean()
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0); self.optimizer.step()
        self.buffer.update_priorities(indices, td.detach().abs().cpu().numpy().flatten() + 1e-3)
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "policy_state": self.policy.state_dict(),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, state_size: int, action_size: int) -> "DoubleDQNAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(state_size, action_size)
        agent.policy.load_state_dict(payload["policy_state"])
        agent.target.load_state_dict(payload["policy_state"])
        return agent


# Backward-compatible name used by the first release.
DQNAgent = DoubleDQNAgent
