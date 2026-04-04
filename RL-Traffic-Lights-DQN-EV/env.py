# ═══════════════════════════════════════════════════════════════════════════════
# env.py — Improved TrafficEnv with reward shaping
# ═══════════════════════════════════════════════════════════════════════════════

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

from config import (
    MAX_QUEUE, SPAWN_RATE_BASE, MIN_GREEN_STEPS, DISCHARGE_RATE,
    MAX_STEPS, CORRIDOR_DELAY, REWARD_SCALE, SWITCH_PENALTY, THROUGHPUT_BONUS,
    EV_PENALTY_WEIGHT,
)


class TrafficEnv(gym.Env):
    """Two-intersection traffic signal control with travel corridors.

    Improvements over original:
      • Reward scaled to [-1, 0] (configurable)
      • Phase-switch penalty discourages rapid toggling
      • Throughput bonus rewards cars that clear the intersection
      • All constants imported from config.py for easy tuning
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    NS_ARMS = [0, 1]   # indices: 0=N, 1=S
    EW_ARMS = [2, 3]   # indices: 2=E, 3=W

    def __init__(self, spawn_rates=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        if spawn_rates is None:
            self.spawn_rates = np.full((2, 4), SPAWN_RATE_BASE)
        else:
            self.spawn_rates = np.array(spawn_rates, dtype=np.float32)

        # ── spaces ──────────────────────────────────────────────────────────
        # obs: [std_q]*4 + [ev_q]*4 + [phase, time_in_phase] × 2
        #    + corridor_0to1 (CORRIDOR_DELAY)  + corridor_1to0 (CORRIDOR_DELAY)
        obs_dim = 20 + (2 * CORRIDOR_DELAY)
        low  = np.zeros(obs_dim, dtype=np.float32)

        high_base = (
            [MAX_QUEUE] * 4 + [MAX_QUEUE] * 4 + [1, MIN_GREEN_STEPS * 4]
            + [MAX_QUEUE] * 4 + [MAX_QUEUE] * 4 + [1, MIN_GREEN_STEPS * 4]
        )
        high_corridors = [DISCHARGE_RATE] * (2 * CORRIDOR_DELAY)
        high = np.array(high_base + high_corridors, dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        # ── internal state ──────────────────────────────────────────────────
        self.queues        = None
        self.phases        = None
        self.time_in_phase = None
        self.step_count    = 0
        self.corridor_0to1 = None
        self.corridor_1to0 = None
        self._renderer     = None

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.queues        = np.zeros((2, 4), dtype=np.int32)
        self.ev_queues     = np.zeros((2, 4), dtype=np.int32)
        self.phases        = np.zeros(2, dtype=np.int32)
        self.time_in_phase = np.zeros(2, dtype=np.int32)
        self.step_count    = 0
        self.corridor_0to1 = deque([0] * CORRIDOR_DELAY, maxlen=CORRIDOR_DELAY)
        self.corridor_1to0 = deque([0] * CORRIDOR_DELAY, maxlen=CORRIDOR_DELAY)
        self.ev_corridor_0to1 = deque([0] * CORRIDOR_DELAY, maxlen=CORRIDOR_DELAY)
        self.ev_corridor_1to0 = deque([0] * CORRIDOR_DELAY, maxlen=CORRIDOR_DELAY)
        return self._obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action, current_ev_spawn_rate=0.01):
        assert self.action_space.contains(action)
        target_phases = [action >> 1, action & 1]

        # ── 1. phase transitions (track switches) ───────────────────────────
        num_switches = 0
        for i in range(2):
            if (target_phases[i] != self.phases[i]
                    and self.time_in_phase[i] >= MIN_GREEN_STEPS):
                self.phases[i]        = target_phases[i]
                self.time_in_phase[i] = 0
                num_switches += 1
            else:
                self.time_in_phase[i] += 1

        # ── 2. discharge green arms & enter corridors ────────────────────────
        entering_0to1  = 0
        entering_1to0  = 0
        ev_entering_0to1 = 0
        ev_entering_1to0 = 0
        total_discharged = 0

        for i in range(2):
            green_arms = self.NS_ARMS if self.phases[i] == 0 else self.EW_ARMS
            for arm in green_arms:
                ev_discharged = min(self.ev_queues[i, arm], DISCHARGE_RATE)
                self.ev_queues[i, arm] -= ev_discharged
                
                rem_capacity = DISCHARGE_RATE - ev_discharged
                std_discharged = min(self.queues[i, arm], rem_capacity)
                self.queues[i, arm] -= std_discharged
                
                total_discharged += (ev_discharged + std_discharged)

                if i == 0 and arm == 3:
                    entering_0to1 = std_discharged
                    ev_entering_0to1 = ev_discharged
                elif i == 1 and arm == 2:
                    entering_1to0 = std_discharged
                    ev_entering_1to0 = ev_discharged

        # ── 3. advance corridors ─────────────────────────────────────────────
        arrived_at_1 = self.corridor_0to1.popleft()
        arrived_at_0 = self.corridor_1to0.popleft()
        ev_arrived_at_1 = self.ev_corridor_0to1.popleft()
        ev_arrived_at_0 = self.ev_corridor_1to0.popleft()

        self.corridor_0to1.append(entering_0to1)
        self.corridor_1to0.append(entering_1to0)
        self.ev_corridor_0to1.append(ev_entering_0to1)
        self.ev_corridor_1to0.append(ev_entering_1to0)

        self.queues[1, 3] = min(MAX_QUEUE, self.queues[1, 3] + arrived_at_1)
        self.queues[0, 2] = min(MAX_QUEUE, self.queues[0, 2] + arrived_at_0)
        self.ev_queues[1, 3] = min(MAX_QUEUE, self.ev_queues[1, 3] + ev_arrived_at_1)
        self.ev_queues[0, 2] = min(MAX_QUEUE, self.ev_queues[0, 2] + ev_arrived_at_0)

        # ── 4. spawn external vehicles ───────────────────────────────────────
        arrivals = self.np_random.random((2, 4)) < self.spawn_rates
        arrivals[0, 2] = 0   # IL East — corridor-only
        arrivals[1, 3] = 0   # IR West — corridor-only
        self.queues = np.clip(
            self.queues + arrivals.astype(np.int32), 0, MAX_QUEUE
        )
        ev_arrivals = self.np_random.random((2, 4)) < current_ev_spawn_rate
        ev_arrivals[0, 2] = 0
        ev_arrivals[1, 3] = 0
        self.ev_queues = np.clip(
            self.ev_queues + ev_arrivals.astype(np.int32), 0, MAX_QUEUE
        )

        # ── 5. reward ────────────────────────────────────────────────────────
        total_stopped_standard = self.queues.sum()
        total_stopped_evs = self.ev_queues.sum()
        max_possible = (MAX_QUEUE * 8) + (MAX_QUEUE * 8 * EV_PENALTY_WEIGHT) 

        if REWARD_SCALE:
            raw_penalty = float(total_stopped_standard) + (EV_PENALTY_WEIGHT * float(total_stopped_evs))
            reward = -raw_penalty / max_possible
        else:
            reward = -float(total_stopped_standard) - (EV_PENALTY_WEIGHT * float(total_stopped_evs))

        reward -= SWITCH_PENALTY * num_switches                # phase-switch cost

        max_throughput = DISCHARGE_RATE * 4                     # 4 green arms max
        reward += THROUGHPUT_BONUS * (total_discharged / max_throughput)  # throughput bonus

        # ── bookkeeping ──────────────────────────────────────────────────────
        self.step_count += 1
        terminated = False
        truncated  = self.step_count >= MAX_STEPS

        info = {
            "queues":           self.queues.copy(),
            "ev_queues":        self.ev_queues.copy(),
            "phases":           self.phases.copy(),
            "total_stopped":    total_stopped_standard,
            "total_evs_stopped": total_stopped_evs,
            "total_discharged": total_discharged,
            "num_switches":     num_switches,
        }

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    def _obs(self):
        obs = []
        for i in range(2):
            obs.extend([
                *self.queues[i].astype(np.float32),
                *self.ev_queues[i].astype(np.float32),
                float(self.phases[i]),
                float(self.time_in_phase[i]),
            ])
        obs.extend(list(self.corridor_0to1))
        obs.extend(list(self.corridor_1to0))
        return np.array(obs, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def render(self):
        if self._renderer is None:
            from visualize import TrafficRenderer
            self._renderer = TrafficRenderer()
        self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
