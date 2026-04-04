# ═══════════════════════════════════════════════════════════════════════════════
# main.py — PPO training loop with rollout-based updates
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
from env import TrafficEnv
from agent import PPOAgent
from config import (
    TOTAL_TIMESTEPS, ROLLOUT_LENGTH, EVAL_INTERVAL, EVAL_EPISODES,
    LOG_INTERVAL, CHECKPOINT_PATH,
)


def evaluate(env, agent, n_episodes=EVAL_EPISODES):
    """Run evaluation episodes with greedy actions (no sampling)."""
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act_greedy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)


def train_agent(total_steps=TOTAL_TIMESTEPS, checkpoint=CHECKPOINT_PATH):
    print("=" * 65)
    print("  PPO — Two-Intersection Traffic Control")
    print("=" * 65)

    env      = TrafficEnv()
    eval_env = TrafficEnv()
    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Obs dim: {obs_dim} | Actions: {n_actions}")
    print(f"  Total steps: {total_steps:,} | Rollout: {ROLLOUT_LENGTH}")
    print("-" * 65)

    agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)

    best_eval_reward = -float("inf")
    step_count       = 0
    episode_count    = 0
    episode_reward   = 0.0
    reward_history   = []
    update_count     = 0
    last_eval_ep     = 0

    state, _ = env.reset()

    while step_count < total_steps:
        # ── collect rollout ───────────────────────────────────────────────
        for _ in range(ROLLOUT_LENGTH):
            action, log_prob, value = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, log_prob, reward, value, float(done))
            state = next_state
            episode_reward += reward
            step_count += 1

            if done:
                episode_count += 1
                reward_history.append(episode_reward)

                if episode_count % LOG_INTERVAL == 0:
                    avg_rew = np.mean(reward_history[-100:])
                    lr = agent.optimizer.param_groups[0]["lr"]
                    print(
                        f"Ep {episode_count:04d} | "
                        f"Steps: {step_count:>9,} | LR={lr:.1e} | "
                        f"Rew={episode_reward:+7.1f} | "
                        f"Avg100={avg_rew:+7.1f}"
                    )

                episode_reward = 0.0
                state, _ = env.reset()

        # ── get bootstrap value for GAE ───────────────────────────────────
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, next_value = agent.network(t)
            next_value = next_value.item()

        # ── PPO update ────────────────────────────────────────────────────
        loss = agent.update(next_value)
        update_count += 1
        agent.step_scheduler()

        # ── periodic evaluation ───────────────────────────────────────────
        if episode_count - last_eval_ep >= EVAL_INTERVAL and episode_count > 0:
            last_eval_ep = episode_count
            eval_mean, eval_std = evaluate(eval_env, agent)
            print(
                f"  ► EVAL @ ep {episode_count:04d} "
                f"(step {step_count:,}): "
                f"{eval_mean:+.1f} ± {eval_std:.1f} | "
                f"PPO loss: {loss:.4f}"
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                agent.save(checkpoint)
                print(f"  ★ New best eval reward: {eval_mean:+.1f}")

    # ── final evaluation ─────────────────────────────────────────────────
    eval_mean, eval_std = evaluate(eval_env, agent)
    print("-" * 65)
    print(f"Final Eval: {eval_mean:+.1f} ± {eval_std:.1f}")
    print(f"Total updates: {update_count} | Total episodes: {episode_count}")
    if eval_mean > best_eval_reward:
        agent.save(checkpoint)

    return agent


def evaluate_agent(agent_path=CHECKPOINT_PATH):
    """Load best model and visualise with Pygame."""
    print("Starting Evaluation (Visual Mode)...")
    env = TrafficEnv(render_mode="human")
    obs_dim   = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions)
    agent.load(agent_path)

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act_greedy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Evaluation reward: {total_reward:+.1f}")
    env.close()


if __name__ == "__main__":
    train_agent()
    # evaluate_agent()
