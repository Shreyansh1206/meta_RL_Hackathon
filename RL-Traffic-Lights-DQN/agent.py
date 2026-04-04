# ═══════════════════════════════════════════════════════════════════════════════
# agent.py — Dueling Double DQN with all improvements
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fixes vs. original:
#   • Corridor normalization uses DISCHARGE_RATE (=2), not MAX_QUEUE (=20)
#   • Soft target updates (Polyak averaging) instead of hard copy every N steps
#   • Learning rate scheduler (StepLR)
#   • Lower LR (5e-4 vs 1e-3)
#   • Bigger replay buffer (100k vs 50k)

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import (
    LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, TAU,
    EPS_START, EPS_END, EPS_DECAY, HIDDEN, GRAD_CLIP,
    MAX_QUEUE, MIN_GREEN_STEPS, CORRIDOR_DELAY, DISCHARGE_RATE,
    LR_STEP_SIZE, LR_GAMMA,
)


# ── observation normalisation vector ────────────────────────────────────────
# Each feature is divided by its upper bound so the network sees [0, 1] inputs.
_queue_block    = [MAX_QUEUE] * 4 + [1.0, float(MIN_GREEN_STEPS * 4)]   # 6 values
_corridor_block = [float(DISCHARGE_RATE)] * CORRIDOR_DELAY               # FIXED!
# ↑  Was MAX_QUEUE (=20) before which squashed corridor features to [0, 0.1].
#    Corridor values are 0..DISCHARGE_RATE, so DISCHARGE_RATE is the correct
#    upper bound.  This is the single most impactful fix for green-wave learning.

_obs_high = np.array(
    _queue_block + _queue_block +          # both intersections
    _corridor_block + _corridor_block,     # both corridors
    dtype=np.float32,
)
# ────────────────────────────────────────────────────────────────────────────


class DuelingDQN(nn.Module):
    """Dueling DQN with built-in input normalisation."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()

        # Register as buffer → auto-moves with .to(device), saved in state_dict
        self.register_buffer(
            "obs_high",
            torch.tensor(_obs_high[:obs_dim], dtype=torch.float32),
        )

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x / self.obs_high, 0.0, 1.0)
        shared = self.shared(x)
        value  = self.value_stream(shared)
        adv    = self.adv_stream(shared)
        return value + adv - adv.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: str = "cpu"):
        self.n_actions = n_actions
        self.device    = torch.device(device)
        self.epsilon   = EPS_START
        self.steps     = 0

        self.online_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
        )
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    # ── act ────────────────────────────────────────────────────────────────
    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(t).argmax(dim=1).item())

    # ── store & learn ──────────────────────────────────────────────────────
    def store(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def learn(self) -> float | None:
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Current Q values
        q_vals = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: online selects action, target evaluates
        with torch.no_grad():
            best_a   = self.online_net(ns).argmax(dim=1, keepdim=True)
            q_target = self.target_net(ns).gather(1, best_a).squeeze(1)
            y        = r + GAMMA * q_target * (1 - d)

        loss = F.smooth_l1_loss(q_vals, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # ── soft target update (Polyak averaging) ──────────────────────────
        for tp, op in zip(self.target_net.parameters(),
                          self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

        self.steps += 1
        return loss.item()

    # ── epsilon decay + LR schedule (call once per episode) ────────────────
    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        self.scheduler.step()

    # ── save / load ────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "online":    self.online_net.state_dict(),
            "target":    self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon":   self.epsilon,
            "steps":     self.steps,
        }, path)
        print(f"[DQN] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"], strict=False)
        self.target_net.load_state_dict(ckpt["target"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", EPS_START)
        self.steps   = ckpt.get("steps", 0)
        print(f"[DQN] Loaded ← {path}")
