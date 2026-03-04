"""
deep_tamer_agent.py
Implements Algorithm 1 from the Deep TAMER paper.

The agent:
  1. Observes state s_i
  2. Selects action a_i = argmax_a H_hat(s_i, a)   (greedy w.r.t. H_hat)
  3. Records experience x_i = (s_i, a_i, t_i, t_{i+1})
  4. On new human feedback: SGD update with that feedback's mini-batch
  5. Every b timesteps: SGD update from the replay buffer

Loss (Eq. 1):  l = w * (H_hat(s, a) - h)^2
  - w: importance weight
  - h: scalar human feedback
  - Error is backpropagated only through the output node for action a.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import BufferEntry, FeedbackReplayBuffer
from .reward_model import RewardModel


class DeepTamerAgent:
    """
    Minimal Deep TAMER agent for low-dimensional state spaces (no CNN).

    Args:
        state_dim: dimensionality of the state vector
        n_actions: number of discrete actions
        hidden_dim: hidden layer size for the reward model MLP
        lr: learning rate for Adam optimiser
        buffer_update_interval: number of timesteps between buffer SGD updates
        minibatch_size: number of samples per SGD update from buffer
        device: torch device string
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        buffer_update_interval: int = 10,
        minibatch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.buffer_update_interval = buffer_update_interval
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.model = RewardModel(state_dim, n_actions, hidden_dim).to(self.device)
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr)

        self.replay_buffer = FeedbackReplayBuffer()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Greedy policy: argmax_a H_hat(s, a)."""
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.model.predict_action(s)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update_on_feedback(
        self,
        feedback: float,
        weighted_experiences: List[Tuple[np.ndarray, int, float]],
    ) -> Optional[float]:
        """
        SGD update triggered immediately by new human feedback (update a).
        Uses the experiences associated with this feedback signal as the mini-batch.

        Returns the loss value, or None if no experiences.
        """
        # Add to replay buffer
        self.replay_buffer.add_feedback_batch(feedback, weighted_experiences)

        if not weighted_experiences:
            return None

        # Build mini-batch from these experiences only
        batch = [(s, a, w, feedback) for s, a, w in weighted_experiences if w > 0]
        if not batch:
            return None

        return self._sgd_update(batch)

    def maybe_update_from_buffer(self) -> Optional[float]:
        """
        SGD update from replay buffer at fixed rate (update b).
        Called every buffer_update_interval timesteps.
        """
        self._step_count += 1
        if self._step_count % self.buffer_update_interval != 0:
            return None
        if len(self.replay_buffer) == 0:
            return None

        entries: List[BufferEntry] = self.replay_buffer.sample_minibatch(
            self.minibatch_size
        )
        batch = [(e.state, e.action, e.weight, e.feedback) for e in entries]
        return self._sgd_update(batch)

    def _sgd_update(
        self,
        batch: List[Tuple[np.ndarray, int, float, float]],
    ) -> float:
        """
        Perform one mini-batch SGD step.

        Loss = mean over batch of w * (H_hat(s, a) - h)^2

        Only the output node for the chosen action receives gradient
        (matching the paper: "errors are only fed back through the single
        relevant output node").
        """
        self.model.train()
        self.optimiser.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)

        for state, action, weight, human_feedback in batch:
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            all_rewards = self.model(s)           # (n_actions,)
            predicted = all_rewards[action]       # scalar for this action only

            h = torch.tensor(human_feedback, dtype=torch.float32, device=self.device)
            w = torch.tensor(weight, dtype=torch.float32, device=self.device)
            loss = w * (predicted - h) ** 2
            total_loss = total_loss + loss

        mean_loss = total_loss / len(batch)
        mean_loss.backward()
        self.optimiser.step()

        return float(mean_loss.item())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Return H_hat(s, .) for all actions (for debugging/analysis)."""
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            return self.model(s).cpu().numpy()
