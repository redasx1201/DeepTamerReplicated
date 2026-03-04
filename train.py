"""
train.py
Deep TAMER training loop on CartPole-v1.

Simulates 15 minutes of human-guided training at 20 fps (18,000 timesteps).
After training, evaluates the learned policy for 20 episodes and compares
against a random-action baseline.

Produces training_curve.png showing score vs simulated training time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.deep_tamer_agent import DeepTamerAgent
from src.human_simulator import HumanSimulator

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
TIMESTEP_DURATION = 0.05      # seconds per step (20 fps)
TOTAL_MINUTES = 15
TOTAL_STEPS = int(TOTAL_MINUTES * 60 / TIMESTEP_DURATION)  # 18,000

BUFFER_UPDATE_INTERVAL = 10   # update from buffer every 10 steps
MINIBATCH_SIZE = 8
LR = 1e-3
HIDDEN_DIM = 64
P_FEEDBACK = 0.04             # ~1 feedback every 25 steps
EVAL_EPISODES = 20
LOG_EVERY_SECONDS = 30        # log score every 30 simulated seconds
SEED = 42


def run_episode(env: gym.Env, agent: DeepTamerAgent, max_steps: int = 500) -> float:
    """Run one episode with the greedy policy, return total reward."""
    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def evaluate_policy(agent: DeepTamerAgent, n_episodes: int = EVAL_EPISODES) -> float:
    """Average episode score over n_episodes greedy rollouts."""
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED + 1000)
    scores = [run_episode(env, agent) for _ in range(n_episodes)]
    env.close()
    return float(np.mean(scores))


def random_baseline(n_episodes: int = EVAL_EPISODES) -> float:
    """Average score of a random policy."""
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED + 2000)
    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def train() -> None:
    np.random.seed(SEED)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n              # 2

    agent = DeepTamerAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        buffer_update_interval=BUFFER_UPDATE_INTERVAL,
        minibatch_size=MINIBATCH_SIZE,
    )
    human = HumanSimulator(p_feedback=P_FEEDBACK, seed=SEED)

    # ---- Training loop ----
    time_checkpoints = []   # simulated time (minutes)
    score_checkpoints = []  # average episode score at checkpoint

    log_interval_steps = int(LOG_EVERY_SECONDS / TIMESTEP_DURATION)
    next_log_step = log_interval_steps

    state, _ = env.reset(seed=SEED)
    episode_score = 0.0
    episode_scores_window = []   # scores in current log window
    current_episode_score = 0.0

    total_feedback_count = 0
    total_buffer_updates = 0
    total_feedback_updates = 0

    print(f"Training Deep TAMER on CartPole-v1 for {TOTAL_MINUTES} simulated minutes "
          f"({TOTAL_STEPS:,} timesteps) ...")
    print(f"{'Step':>8}  {'Sim time':>10}  {'Ep score (window)':>18}  "
          f"{'Feedback':>10}  {'Buffer size':>12}")
    print("-" * 70)

    for step in range(TOTAL_STEPS):
        t = step * TIMESTEP_DURATION
        t_next = (step + 1) * TIMESTEP_DURATION

        # --- Select and execute action ---
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        current_episode_score += reward
        done = terminated or truncated

        # --- Record experience for credit assignment ---
        human.record_experience(state, action, t, t_next)

        # --- Human maybe schedules feedback about the (state, action) pair ---
        human.maybe_schedule_feedback(state, action, t_next)

        # --- Collect any feedback that has arrived ---
        due_feedback = human.collect_due_feedback(t_next)
        for fb_value, weighted_exps in due_feedback:
            total_feedback_count += 1
            loss = agent.update_on_feedback(fb_value, weighted_exps)
            if loss is not None:
                total_feedback_updates += 1

        # --- Fixed-rate buffer update ---
        buf_loss = agent.maybe_update_from_buffer()
        if buf_loss is not None:
            total_buffer_updates += 1

        # --- Episode bookkeeping ---
        if done:
            episode_scores_window.append(current_episode_score)
            current_episode_score = 0.0
            state, _ = env.reset()
        else:
            state = next_state

        # --- Logging checkpoint ---
        if step >= next_log_step:
            sim_minutes = t / 60.0
            if episode_scores_window:
                window_avg = np.mean(episode_scores_window[-10:])
            else:
                window_avg = float("nan")
            time_checkpoints.append(sim_minutes)
            score_checkpoints.append(window_avg)
            print(
                f"{step:>8,}  {sim_minutes:>9.1f}m  {window_avg:>18.1f}  "
                f"{total_feedback_count:>10,}  {len(agent.replay_buffer):>12,}"
            )
            next_log_step += log_interval_steps

    env.close()
    print("-" * 70)
    print(f"\nTraining complete.")
    print(f"  Total feedback signals processed: {total_feedback_count:,}")
    print(f"  Feedback-triggered SGD updates:   {total_feedback_updates:,}")
    print(f"  Buffer SGD updates:               {total_buffer_updates:,}")
    print(f"  Replay buffer size:               {len(agent.replay_buffer):,}")

    # ---- Evaluation ----
    print("\nEvaluating learned policy ...")
    final_score = evaluate_policy(agent)
    rand_score = random_baseline()
    print(f"  Deep TAMER average score ({EVAL_EPISODES} eps): {final_score:.1f}")
    print(f"  Random policy average score     : {rand_score:.1f}")
    print(f"  Improvement over random         : {final_score - rand_score:+.1f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5))
    if time_checkpoints:
        ax.plot(time_checkpoints, score_checkpoints, "b-o", markersize=4,
                label="Deep TAMER (training)")
    ax.axhline(rand_score, color="gray", linestyle="--", label=f"Random ({rand_score:.0f})")
    ax.axhline(final_score, color="green", linestyle="--",
               label=f"Deep TAMER final ({final_score:.0f})")
    ax.set_xlabel("Simulated training time (minutes)")
    ax.set_ylabel("Episode score (10-episode rolling avg)")
    ax.set_title("Deep TAMER on CartPole-v1\n(simulated human feedback, 15 min training)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "training_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    train()
