"""
replay_buffer.py
Implements the feedback replay buffer D from the Deep TAMER paper.

D = { (xi, yj) | w(xi, yj) != 0 }

Each entry stores:
  state   : numpy array
  action  : int
  weight  : float  (importance weight > 0)
  feedback: float  (scalar human feedback h)

The buffer grows throughout training (paper does not cap it; ~1000 signals
over 15 minutes is manageable).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BufferEntry:
    state: np.ndarray
    action: int
    weight: float
    feedback: float


class FeedbackReplayBuffer:
    """Stores all (state, action, weight, feedback) tuples with w > 0."""

    def __init__(self) -> None:
        self._buffer: List[BufferEntry] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        weight: float,
        feedback: float,
    ) -> None:
        """Add a single entry (only called when weight > 0)."""
        self._buffer.append(BufferEntry(state, action, weight, feedback))

    def add_feedback_batch(
        self,
        feedback: float,
        experiences: List[tuple],
    ) -> None:
        """
        Process one feedback signal against a list of recent experiences.

        Args:
            feedback: scalar h from the human
            experiences: list of (state, action, weight) tuples where weight
                         has already been computed via compute_importance_weight
        """
        for state, action, weight in experiences:
            if weight > 0:
                self.add(state, action, weight, feedback)

    def sample_minibatch(self, batch_size: int) -> Optional[List[BufferEntry]]:
        """Sample with replacement. Returns None if buffer is empty."""
        if not self._buffer:
            return None
        return random.choices(self._buffer, k=batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
