"""
test_deep_tamer.py
Unit tests for all Deep TAMER components.
Run with: python test_deep_tamer.py
"""

import sys
import os
import math
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.importance_weights import compute_importance_weight
from src.replay_buffer import FeedbackReplayBuffer
from src.reward_model import RewardModel
from src.human_simulator import HumanSimulator
from src.deep_tamer_agent import DeepTamerAgent


# ======================================================================
# 1. Importance weights
# ======================================================================

class TestImportanceWeights(unittest.TestCase):

    def test_feedback_before_action_is_zero(self):
        # tf < ts: feedback arrived before the state-action pair — impossible
        w = compute_importance_weight(ts=5.0, te=5.05, tf=4.9)
        self.assertEqual(w, 0.0)

    def test_feedback_too_late_is_zero(self):
        # tf - ts = 10 >> delay_max=4 → no overlap
        w = compute_importance_weight(ts=0.0, te=0.05, tf=10.0)
        self.assertEqual(w, 0.0)

    def test_feedback_too_early_is_zero(self):
        # tf - te = 0.05 < delay_min=0.2 → no overlap
        w = compute_importance_weight(ts=1.0, te=1.05, tf=1.1)
        self.assertEqual(w, 0.0)

    def test_perfect_overlap(self):
        # Delay interval [tf-te, tf-ts] = [0.2, 4.0] exactly covers uniform support
        # → overlap = 3.8, w = 1.0
        w = compute_importance_weight(ts=0.0, te=3.8, tf=4.0)
        self.assertAlmostEqual(w, 1.0, places=6)

    def test_partial_overlap(self):
        # ts=0.0, te=0.05, tf=1.0 → delay interval [0.95, 1.0]
        # overlap with [0.2, 4.0] = [0.95, 1.0] → length=0.05
        # w = 0.05 / 3.8
        w = compute_importance_weight(ts=0.0, te=0.05, tf=1.0)
        expected = 0.05 / 3.8
        self.assertAlmostEqual(w, expected, places=6)

    def test_weight_in_unit_interval(self):
        for _ in range(50):
            ts = np.random.uniform(0, 5)
            te = ts + np.random.uniform(0.01, 0.2)
            tf = np.random.uniform(ts, ts + 6)
            w = compute_importance_weight(ts, te, tf)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0 + 1e-9)


# ======================================================================
# 2. Replay buffer
# ======================================================================

class TestReplayBuffer(unittest.TestCase):

    def test_empty_sample_returns_none(self):
        buf = FeedbackReplayBuffer()
        self.assertIsNone(buf.sample_minibatch(8))

    def test_add_and_len(self):
        buf = FeedbackReplayBuffer()
        buf.add(np.zeros(4), 0, 0.5, 1.0)
        buf.add(np.ones(4), 1, 0.3, -1.0)
        self.assertEqual(len(buf), 2)

    def test_sample_returns_correct_count(self):
        buf = FeedbackReplayBuffer()
        for i in range(20):
            buf.add(np.zeros(4), 0, 0.5, 1.0)
        batch = buf.sample_minibatch(8)
        self.assertEqual(len(batch), 8)

    def test_add_feedback_batch_filters_zero_weight(self):
        buf = FeedbackReplayBuffer()
        exps = [
            (np.zeros(4), 0, 0.0),   # zero weight — should be skipped
            (np.ones(4),  1, 0.5),   # non-zero — should be added
        ]
        buf.add_feedback_batch(1.0, exps)
        self.assertEqual(len(buf), 1)
        entry = buf.sample_minibatch(1)[0]
        self.assertEqual(entry.action, 1)
        self.assertAlmostEqual(entry.feedback, 1.0)

    def test_sample_with_replacement(self):
        buf = FeedbackReplayBuffer()
        buf.add(np.zeros(4), 0, 0.5, 1.0)
        # Requesting more than buffer size should still work (sampling with replacement)
        batch = buf.sample_minibatch(50)
        self.assertEqual(len(batch), 50)


# ======================================================================
# 3. Reward model
# ======================================================================

class TestRewardModel(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.n_actions = 2
        self.model = RewardModel(self.state_dim, self.n_actions)

    def test_output_shape_single(self):
        s = torch.zeros(self.state_dim)
        out = self.model(s)
        self.assertEqual(out.shape, (self.n_actions,))

    def test_output_shape_batch(self):
        s = torch.zeros(8, self.state_dim)
        out = self.model(s)
        self.assertEqual(out.shape, (8, self.n_actions))

    def test_predict_action_valid_index(self):
        s = torch.randn(self.state_dim)
        action = self.model.predict_action(s)
        self.assertIn(action, list(range(self.n_actions)))

    def test_predict_action_is_int(self):
        s = torch.randn(self.state_dim)
        action = self.model.predict_action(s)
        self.assertIsInstance(action, int)


# ======================================================================
# 4. Human simulator
# ======================================================================

class TestHumanSimulator(unittest.TestCase):

    def test_good_state_action_gives_positive_feedback(self):
        human = HumanSimulator(seed=0)
        # Pole tilting right (angle > 0), correct action is push right (1)
        state = np.array([0.0, 0.0, 0.1, 0.0])
        fb = human._score_state_action(state, action=1)
        self.assertEqual(fb, 1.0)

    def test_bad_state_action_gives_negative_feedback(self):
        human = HumanSimulator(seed=0)
        # Pole tilting right (angle > 0), wrong action is push left (0)
        state = np.array([0.0, 0.0, 0.1, 0.0])
        fb = human._score_state_action(state, action=0)
        self.assertEqual(fb, -1.0)

    def test_feedback_collected_after_delay(self):
        human = HumanSimulator(p_feedback=1.0, delay_min=1.0, delay_max=1.0, seed=0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        human.record_experience(state, 0, ts=0.0, te=0.05)
        human.maybe_schedule_feedback(state, action=0, current_t=0.05)

        # Before delay expires: no feedback
        due_early = human.collect_due_feedback(current_t=0.5)
        self.assertEqual(len(due_early), 0)

        # After delay (1.0 s) expires
        due_late = human.collect_due_feedback(current_t=1.1)
        self.assertEqual(len(due_late), 1)
        feedback_val, _ = due_late[0]
        self.assertEqual(feedback_val, 1.0)

    def test_experience_history_pruned(self):
        human = HumanSimulator(history_window=1.0, seed=0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        # Add experiences spanning 0 to 2 seconds (should prune old ones)
        for i in range(100):
            t = i * 0.05
            human.record_experience(state, 0, ts=t, te=t + 0.05)
        # History should only contain last ~1/0.05 = 20 entries
        self.assertLessEqual(len(human._experience_history), 25)


# ======================================================================
# 5. Deep TAMER agent end-to-end
# ======================================================================

class TestDeepTamerAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DeepTamerAgent(state_dim=4, n_actions=2, hidden_dim=32, lr=1e-2)

    def test_action_selection_valid(self):
        state = np.zeros(4)
        action = self.agent.select_action(state)
        self.assertIn(action, [0, 1])

    def test_update_on_feedback_runs(self):
        state = np.zeros(4)
        exps = [(state, 0, 0.5)]
        loss = self.agent.update_on_feedback(1.0, exps)
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

    def test_buffer_update_interval(self):
        state = np.zeros(4)
        # Populate buffer first
        exps = [(state, 0, 0.5)]
        self.agent.update_on_feedback(1.0, exps)
        # No update before interval
        for _ in range(self.agent.buffer_update_interval - 1):
            result = self.agent.maybe_update_from_buffer()
        # At interval step, should return a loss
        result = self.agent.maybe_update_from_buffer()
        self.assertIsNotNone(result)

    def test_agent_learns_consistent_feedback(self):
        """
        After many updates with +1 feedback for action 0 and -1 for action 1,
        H_hat should predict higher reward for action 0.
        """
        np.random.seed(42)
        agent = DeepTamerAgent(state_dim=4, n_actions=2, hidden_dim=32, lr=5e-3)
        state = np.array([0.1, 0.0, -0.05, 0.0])

        # 200 feedback updates strongly preferring action 0
        for _ in range(200):
            exps_good = [(state + np.random.randn(4) * 0.01, 0, 0.8)]
            exps_bad  = [(state + np.random.randn(4) * 0.01, 1, 0.8)]
            agent.update_on_feedback(1.0, exps_good)
            agent.update_on_feedback(-1.0, exps_bad)

        action = agent.select_action(state)
        self.assertEqual(action, 0,
            "Agent should prefer action 0 after consistent positive feedback for it.")

    def test_replay_buffer_grows(self):
        state = np.zeros(4)
        exps = [(state, 0, 0.5), (state, 1, 0.3)]
        self.agent.update_on_feedback(1.0, exps)
        self.assertEqual(len(self.agent.replay_buffer), 2)


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
