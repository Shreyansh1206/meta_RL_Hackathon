# """
# traffic_rl/agent.py
# -------------------
# Deep Q-Network (DQN) agent with:
#   • Experience replay (deque-based)
#   • Target network (hard update every N steps)
#   • Epsilon-greedy exploration with exponential decay
#   • Dueling network architecture for more stable value estimation
# """

# import random
# import numpy as np
# from collections import deque

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F


# # ── hyper-parameters ────────────────────────────────────────────────────────
# LR              = 1e-3
# GAMMA           = 0.99
# BUFFER_SIZE     = 50_000
# BATCH_SIZE      = 64
# TARGET_UPDATE   = 500        # steps between target-net hard updates
# EPS_START       = 1.0
# EPS_END         = 0.05
# EPS_DECAY       = 0.995      # per episode
# HIDDEN          = 128
# # ───────────────────────────────────────────────────────────────────────────


# class DuelingDQN(nn.Module):
#     """Dueling DQN: separate value and advantage streams."""

#     def __init__(self, obs_dim: int, n_actions: int):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
#             nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
#         )
#         self.value_stream = nn.Sequential(
#             nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
#             nn.Linear(HIDDEN // 2, 1),
#         )
#         self.adv_stream = nn.Sequential(
#             nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
#             nn.Linear(HIDDEN // 2, n_actions),
#         )

#     def forward(self, x):
#         shared = self.shared(x)
#         value  = self.value_stream(shared)
#         adv    = self.adv_stream(shared)
#         # combine: Q = V + (A - mean(A))
#         return value + adv - adv.mean(dim=-1, keepdim=True)


# class ReplayBuffer:
#     def __init__(self, capacity: int):
#         self.buf = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buf.append((state, action, reward, next_state, done))

#     def sample(self, batch_size: int):
#         batch = random.sample(self.buf, batch_size)
#         s, a, r, ns, d = zip(*batch)
#         return (
#             np.array(s,  dtype=np.float32),
#             np.array(a,  dtype=np.int64),
#             np.array(r,  dtype=np.float32),
#             np.array(ns, dtype=np.float32),
#             np.array(d,  dtype=np.float32),
#         )

#     def __len__(self):
#         return len(self.buf)


# class DQNAgent:
#     def __init__(self, obs_dim: int, n_actions: int, device: str = "cpu"):
#         self.n_actions = n_actions
#         self.device    = torch.device(device)
#         self.epsilon   = EPS_START
#         self.steps     = 0

#         self.online_net = DuelingDQN(obs_dim, n_actions).to(self.device)
#         self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
#         self.target_net.load_state_dict(self.online_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
#         self.buffer    = ReplayBuffer(BUFFER_SIZE)

#     # ── act ────────────────────────────────────────────────────────────────
#     def act(self, state: np.ndarray) -> int:
#         if random.random() < self.epsilon:
#             return random.randrange(self.n_actions)
#         with torch.no_grad():
#             t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             return int(self.online_net(t).argmax(dim=1).item())

#     # ── store & learn ──────────────────────────────────────────────────────
#     def store(self, s, a, r, ns, done):
#         self.buffer.push(s, a, r, ns, done)

#     def learn(self) -> float | None:
#         if len(self.buffer) < BATCH_SIZE:
#             return None

#         s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
#         s  = torch.FloatTensor(s).to(self.device)
#         a  = torch.LongTensor(a).to(self.device)
#         r  = torch.FloatTensor(r).to(self.device)
#         ns = torch.FloatTensor(ns).to(self.device)
#         d  = torch.FloatTensor(d).to(self.device)

#         # current Q
#         q_vals   = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

#         # Double DQN target: online selects action, target evaluates
#         with torch.no_grad():
#             best_a   = self.online_net(ns).argmax(dim=1, keepdim=True)
#             q_target = self.target_net(ns).gather(1, best_a).squeeze(1)
#             y        = r + GAMMA * q_target * (1 - d)

#         loss = F.smooth_l1_loss(q_vals, y)
#         self.optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
#         self.optimizer.step()

#         self.steps += 1
#         if self.steps % TARGET_UPDATE == 0:
#             self.target_net.load_state_dict(self.online_net.state_dict())

#         return loss.item()

#     # ── epsilon decay (call once per episode) ──────────────────────────────
#     def decay_epsilon(self):
#         self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

#     # ── save / load ────────────────────────────────────────────────────────
#     def save(self, path: str):
#         torch.save({
#             "online":  self.online_net.state_dict(),
#             "target":  self.target_net.state_dict(),
#             "epsilon": self.epsilon,
#             "steps":   self.steps,
#         }, path)
#         print(f"[Agent] Saved → {path}")

#     def load(self, path: str):
#         ckpt = torch.load(path, map_location=self.device)
#         self.online_net.load_state_dict(ckpt["online"])
#         self.target_net.load_state_dict(ckpt["target"])
#         self.epsilon = ckpt["epsilon"]
#         self.steps   = ckpt["steps"]
#         print(f"[Agent] Loaded ← {path}")
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── hyper-parameters ────────────────────────────────────────────────────────
LR              = 1e-3
GAMMA           = 0.999       # Essential for sync: care about delayed consequences
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 64
TARGET_UPDATE   = 500        
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995      
HIDDEN          = 256        # Bumped to 256 to handle the larger corridor state space
# ───────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling DQN: separate value and advantage streams."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
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

    def forward(self, x):
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
        self.buffer    = ReplayBuffer(BUFFER_SIZE)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(t).argmax(dim=1).item())

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

        q_vals = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a   = self.online_net(ns).argmax(dim=1, keepdim=True)
            q_target = self.target_net(ns).gather(1, best_a).squeeze(1)
            y        = r + GAMMA * q_target * (1 - d)

        loss = F.smooth_l1_loss(q_vals, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path: str):
        torch.save({
            "online":  self.online_net.state_dict(),
            "target":  self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "steps":   self.steps,
        }, path)
        print(f"[Agent] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"[Agent] Loaded ← {path}")