# ═══════════════════════════════════════════════════════════════════════════════
# config.py — SINGLE SOURCE OF TRUTH for ALL tunable parameters
# ─────────────────────────────────────────────────────────────────────────────
# Edit these values BEFORE training. No other file needs changes.
# ═══════════════════════════════════════════════════════════════════════════════


# ── Environment ─────────────────────────────────────────────────────────────
MAX_QUEUE        = 20        # max cars per arm (capacity cap)
SPAWN_RATE_BASE  = 0.25      # probability a car arrives per arm per step
MIN_GREEN_STEPS  = 5         # minimum steps before a phase change is allowed
DISCHARGE_RATE   = 2         # cars cleared per green arm per step
MAX_STEPS        = 1000      # episode length (steps)
CORRIDOR_DELAY   = 5         # steps for inter-intersection travel


# ── Reward Shaping ──────────────────────────────────────────────────────────
REWARD_SCALE     = True      # normalize queue penalty to [-1, 0] range
SWITCH_PENALTY   = 0.1       # penalty per phase switch (set 0.0 to disable)
THROUGHPUT_BONUS = 0.05      # reward per normalized throughput (set 0.0 to disable)


# ── PPO Agent ───────────────────────────────────────────────────────────────
LR               = 3e-4      # Adam learning rate
GAMMA            = 0.99      # discount factor
GAE_LAMBDA       = 0.95      # GAE lambda for advantage estimation
CLIP_EPS         = 0.2       # PPO clipping epsilon
ENTROPY_COEFF    = 0.01      # entropy bonus coefficient (exploration)
VALUE_COEFF      = 0.5       # value loss coefficient
MAX_GRAD_NORM    = 0.5       # max gradient norm for clipping
HIDDEN           = 256       # hidden layer width
PPO_EPOCHS       = 10        # optimization epochs per rollout
ROLLOUT_LENGTH   = 2048      # steps collected per rollout before update
MINI_BATCH_SIZE  = 64        # mini-batch size within each PPO epoch


# ── Learning Rate Scheduler ────────────────────────────────────────────────
LR_STEP_SIZE     = 100       # rollout updates between LR reductions
LR_GAMMA         = 0.9       # LR multiplier at each reduction


# ── Training ────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 2_000_000 # total environment steps (≈ 500 episodes worth)
EVAL_INTERVAL    = 50        # episodes between evaluations
EVAL_EPISODES    = 5         # episodes per evaluation run
LOG_INTERVAL     = 10        # episodes between console logs
CHECKPOINT_PATH  = "best_traffic_agent_ppo.pth"
