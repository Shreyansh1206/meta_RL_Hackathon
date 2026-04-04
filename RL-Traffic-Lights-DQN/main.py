# ═══════════════════════════════════════════════════════════════════════════════
# main.py — DQN training loop with eval-based checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
from env import TrafficEnv
from agent import DQNAgent
from config import (
    NUM_EPISODES, EVAL_INTERVAL, EVAL_EPISODES,
    LOG_INTERVAL, CHECKPOINT_PATH,
)


def evaluate(env, agent, n_episodes=EVAL_EPISODES):
    """Run evaluation episodes with ε=0 (pure exploitation)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    agent.epsilon = old_eps
    return np.mean(rewards), np.std(rewards)


def train_agent(episodes=NUM_EPISODES, checkpoint=CHECKPOINT_PATH):
    print("=" * 65)
    print("  Dueling Double DQN — Two-Intersection Traffic Control")
    print("=" * 65)

    env      = TrafficEnv()
    eval_env = TrafficEnv()
    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Obs dim: {obs_dim} | Actions: {n_actions}")
    print(f"  Episodes: {episodes} | Eval every: {EVAL_INTERVAL} eps")
    print("-" * 65)

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)

    best_eval_reward = -float("inf")
    reward_history   = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        losses = []

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        reward_history.append(total_reward)

        # ── console logging ──────────────────────────────────────────────
        if ep % LOG_INTERVAL == 0:
            avg_loss   = np.mean(losses) if losses else 0
            avg_rew    = np.mean(reward_history[-100:])
            lr         = agent.optimizer.param_groups[0]["lr"]
            buf_fill   = len(agent.buffer)
            print(
                f"Ep {ep:04d} | ε={agent.epsilon:.3f} | LR={lr:.1e} | "
                f"Rew={total_reward:+7.1f} | Avg100={avg_rew:+7.1f} | "
                f"Loss={avg_loss:.4f} | Buf={buf_fill}"
            )

        # ── periodic evaluation (ε=0) ────────────────────────────────────
        if ep > 0 and ep % EVAL_INTERVAL == 0:
            eval_mean, eval_std = evaluate(eval_env, agent)
            print(
                f"  ► EVAL @ ep {ep:04d}: "
                f"{eval_mean:+.1f} ± {eval_std:.1f}"
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                agent.save(checkpoint)
                print(f"  ★ New best eval reward: {eval_mean:+.1f}")

    # ── final evaluation ─────────────────────────────────────────────────
    eval_mean, eval_std = evaluate(eval_env, agent)
    print("-" * 65)
    print(f"Final Eval: {eval_mean:+.1f} ± {eval_std:.1f}")
    if eval_mean > best_eval_reward:
        agent.save(checkpoint)

    return agent


def evaluate_agent(agent_path=CHECKPOINT_PATH):
    """Load best model and visualise with Pygame."""
    print("Starting Evaluation (Visual Mode)...")
    env = TrafficEnv(render_mode="human")
    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
    agent.load(agent_path)
    agent.epsilon = 0.0

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Evaluation reward: {total_reward:+.1f}")
    env.close()


if __name__ == "__main__":
    train_agent()
    # evaluate_agent()
