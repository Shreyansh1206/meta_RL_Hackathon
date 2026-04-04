# ═══════════════════════════════════════════════════════════════════════════════
# agent.py — PPO (Proximal Policy Optimization) agent
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   • Actor-Critic with shared backbone + separate heads
#   • Input normalisation (same as DQN variant)
#   • Clipped surrogate objective
#   • Generalized Advantage Estimation (GAE)
#   • Entropy bonus for sustained exploration

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from config import (
    LR, GAMMA, GAE_LAMBDA, CLIP_EPS, ENTROPY_COEFF, VALUE_COEFF,
    MAX_GRAD_NORM, HIDDEN, PPO_EPOCHS, MINI_BATCH_SIZE,
    MAX_QUEUE, MIN_GREEN_STEPS, CORRIDOR_DELAY, DISCHARGE_RATE,
    LR_STEP_SIZE, LR_GAMMA,
)


# ── observation normalisation vector ────────────────────────────────────────
_queue_block    = [MAX_QUEUE] * 4 + [1.0, float(MIN_GREEN_STEPS * 4)]
_corridor_block = [float(DISCHARGE_RATE)] * CORRIDOR_DELAY   # correct upper bound

_obs_high = np.array(
    _queue_block + _queue_block +
    _corridor_block + _corridor_block,
    dtype=np.float32,
)
# ────────────────────────────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """Shared backbone → Actor (policy logits) + Critic (state value)."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()

        self.register_buffer(
            "obs_high",
            torch.tensor(_obs_high[:obs_dim], dtype=torch.float32),
        )

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),
        )

        # Orthogonal initialisation (standard for PPO)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Policy head: small init for near-uniform initial distribution
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Value head: standard init
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        x = torch.clamp(x / self.obs_high, 0.0, 1.0)
        shared = self.shared(x)
        logits = self.actor(shared)
        value  = self.critic(shared).squeeze(-1)
        return logits, value

    def act(self, state_t: torch.Tensor):
        """Sample an action. Returns (action, log_prob, value)."""
        with torch.no_grad():
            logits, value = self.forward(state_t)
            dist   = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def act_greedy(self, state_t: torch.Tensor):
        """Greedy action for evaluation."""
        with torch.no_grad():
            logits, _ = self.forward(state_t)
            return logits.argmax(dim=-1).item()

    def evaluate_actions(self, states, actions):
        """Re-evaluate stored actions (for PPO update)."""
        logits, values = self.forward(states)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy


class RolloutBuffer:
    """Stores a single rollout of experience for on-policy learning."""

    def __init__(self):
        self.clear()

    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: str = "cpu"):
        self.device    = torch.device(device)
        self.n_actions = n_actions

        self.network   = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
        )
        self.buffer = RolloutBuffer()

    # ── act ────────────────────────────────────────────────────────────────
    def act(self, state: np.ndarray):
        """Returns (action, log_prob, value)."""
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.network.act(t)

    def act_greedy(self, state: np.ndarray) -> int:
        """Greedy action for evaluation."""
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.network.act_greedy(t)

    # ── store ──────────────────────────────────────────────────────────────
    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.push(state, action, log_prob, reward, value, done)

    # ── GAE ────────────────────────────────────────────────────────────────
    def compute_gae(self, next_value: float):
        """Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        values  = self.buffer.values
        dones   = self.buffer.dones
        n       = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_val * non_terminal - values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ── PPO update ─────────────────────────────────────────────────────────
    def update(self, next_value: float) -> float:
        """Run PPO clipped update on the collected rollout."""
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states    = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions   = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp    = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        adv_t     = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.buffer)
        total_loss = 0.0
        n_updates  = 0

        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(n)

            for start in range(0, n, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                if end > n:
                    continue

                idx = indices[start:end]

                log_probs, values, entropy = self.network.evaluate_actions(
                    states[idx], actions[idx]
                )

                # ── clipped surrogate ──────────────────────────────────────
                ratio = torch.exp(log_probs - old_lp[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS,
                                           1.0 + CLIP_EPS) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── value loss ─────────────────────────────────────────────
                value_loss = nn.functional.mse_loss(values, returns_t[idx])

                # ── entropy bonus ──────────────────────────────────────────
                entropy_loss = -entropy.mean()

                # ── total ──────────────────────────────────────────────────
                loss = (policy_loss
                        + VALUE_COEFF * value_loss
                        + ENTROPY_COEFF * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), MAX_GRAD_NORM
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        self.buffer.clear()
        return total_loss / max(n_updates, 1)

    # ── LR scheduler step ─────────────────────────────────────────────────
    def step_scheduler(self):
        self.scheduler.step()

    # ── save / load ────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[PPO] Loaded ← {path}")
