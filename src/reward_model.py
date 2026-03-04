"""
reward_model.py
H_hat: the agent's learned approximation of the human's reward function.

Architecture: MLP with 2 hidden layers, outputting one predicted reward
per action (analogous to the fully-connected head in the paper's Fig. 3).
"""

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Predicts H(s, a) for all actions simultaneously.

    Output shape: (batch_size, n_actions)
    Policy: pi(s) = argmax_a H_hat(s, a)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) or (state_dim,)
        Returns:
            per_action_rewards: (batch, n_actions) or (n_actions,)
        """
        return self.net(state)

    def predict_action(self, state: torch.Tensor) -> int:
        """Greedy action selection: argmax_a H_hat(s, a)."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            rewards = self.forward(state)  # (1, n_actions)
            return int(rewards.argmax(dim=1).item())
