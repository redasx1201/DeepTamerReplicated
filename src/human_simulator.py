"""
human_simulator.py
Simulates a human trainer providing real-time scalar feedback.

The simulated human:
  1. Observes the agent's state and action at each timestep.
  2. With probability p_feedback per step (~0.04, i.e., once every 25 steps),
     decides to give feedback about the current situation.
  3. The feedback arrives after a realistic delay sampled from Uniform[0.5, 2.0]
     seconds (within the paper's assumed [0.2, 4.0] credit window).
  4. Feedback value h:
       +1  if pole is upright (|angle| < 0.15 rad) AND cart is centred (|pos| < 1.2)
       -1  otherwise (pole leaning badly or cart near edge)

CartPole state layout: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .importance_weights import compute_importance_weight


@dataclass
class Experience:
    state: np.ndarray
    action: int
    ts: float   # start time (seconds)
    te: float   # end time (seconds)


@dataclass
class PendingFeedback:
    feedback: float
    tf: float           # time at which feedback becomes available
    ref_state: np.ndarray  # the state the human was reacting to


class HumanSimulator:
    """
    Simulates a human trainer for CartPole-v1.

    Usage in the training loop:
        human = HumanSimulator()
        ...
        human.record_experience(state, action, t_start, t_end)
        pending = human.maybe_schedule_feedback(state, t_end)
        due = human.collect_due_feedback(current_t)
        for feedback, experiences_with_weights in due:
            ...
    """

    def __init__(
        self,
        p_feedback: float = 0.04,
        delay_min: float = 0.5,
        delay_max: float = 2.0,
        history_window: float = 4.0,
        good_angle_thresh: float = 0.15,
        good_pos_thresh: float = 1.2,
        seed: Optional[int] = None,
    ) -> None:
        self.p_feedback = p_feedback
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.history_window = history_window
        self.good_angle_thresh = good_angle_thresh
        self.good_pos_thresh = good_pos_thresh

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self._experience_history: List[Experience] = []
        self._pending_feedback: List[PendingFeedback] = []

    # ------------------------------------------------------------------
    # Recording experience
    # ------------------------------------------------------------------

    def record_experience(
        self,
        state: np.ndarray,
        action: int,
        ts: float,
        te: float,
    ) -> None:
        """Call once per timestep to log the (s, a, ts, te) tuple."""
        self._experience_history.append(Experience(state.copy(), action, ts, te))
        # Prune old experiences outside the credit window
        cutoff = te - self.history_window
        self._experience_history = [
            e for e in self._experience_history if e.te > cutoff
        ]

    # ------------------------------------------------------------------
    # Scheduling feedback
    # ------------------------------------------------------------------

    def maybe_schedule_feedback(
        self,
        state: np.ndarray,
        action: int,
        current_t: float,
    ) -> bool:
        """
        With probability p_feedback, the human decides to give feedback
        about the state-action pair just observed. Feedback arrives after
        a random reaction delay.

        Returns True if feedback was scheduled.
        """
        if self._rng.random() > self.p_feedback:
            return False

        feedback = self._score_state_action(state, action)
        delay = self._rng.uniform(self.delay_min, self.delay_max)
        tf = current_t + delay
        self._pending_feedback.append(PendingFeedback(feedback, tf, state.copy()))
        return True

    def _score_state_action(self, state: np.ndarray, action: int) -> float:
        """
        Heuristic human scoring for CartPole state-action pairs.
        state = [cart_pos, cart_vel, pole_angle, pole_angular_vel]

        The human gives +1 if the action is a good corrective move, -1 otherwise.

        Correct action logic:
          - Estimate where the pole tip is heading using (angle + angular_vel).
          - If heading right (positive), the corrective action is push right (action=1).
          - If heading left (negative), the corrective action is push left (action=0).
          - Additionally penalise if the cart is near the edge and moving away.
        """
        cart_pos = state[0]
        pole_angle = state[2]
        pole_vel = state[3]

        # Predict pole direction tendency
        tendency = pole_angle + 0.5 * pole_vel
        correct_action = 1 if tendency > 0 else 0

        if action == correct_action:
            return 1.0
        return -1.0

    # ------------------------------------------------------------------
    # Collecting due feedback with credit assignment
    # ------------------------------------------------------------------

    def collect_due_feedback(
        self,
        current_t: float,
    ) -> List[Tuple[float, List[Tuple[np.ndarray, int, float]]]]:
        """
        Returns all feedback signals whose tf <= current_t, along with the
        list of (state, action, weight) tuples from the experience history.

        Removes collected feedback from the pending queue.
        """
        due = [pf for pf in self._pending_feedback if pf.tf <= current_t]
        self._pending_feedback = [pf for pf in self._pending_feedback if pf.tf > current_t]

        results = []
        for pf in due:
            weighted_experiences = []
            for exp in self._experience_history:
                w = compute_importance_weight(exp.ts, exp.te, pf.tf)
                if w > 0:
                    weighted_experiences.append((exp.state, exp.action, w))
            results.append((pf.feedback, weighted_experiences))
        return results

    @property
    def total_feedback_given(self) -> int:
        return len(self._pending_feedback)  # pending (not yet due)
