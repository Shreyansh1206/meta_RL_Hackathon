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
EV_SPAWN_RATE_START = 0.15
EV_SPAWN_RATE_END = 0.01
EV_PENALTY_WEIGHT = 20.0


# ── Reward Shaping ──────────────────────────────────────────────────────────
REWARD_SCALE     = True      # normalize queue penalty to [-1, 0] range
SWITCH_PENALTY   = 0.1       # penalty per phase switch (set 0.0 to disable)
THROUGHPUT_BONUS = 0.05      # reward per normalized throughput (set 0.0 to disable)


# ── DQN Agent ───────────────────────────────────────────────────────────────
LR               = 5e-4      # Adam learning rate (was 1e-3 — too aggressive)
GAMMA            = 0.99      # discount factor (high → agent plans ahead for corridor)
BUFFER_SIZE      = 100_000   # replay buffer capacity (was 50k)
BATCH_SIZE       = 64        # transitions per gradient step
TAU              = 0.005     # soft target update coefficient (Polyak averaging)
EPS_START        = 1.0       # initial exploration rate (100% random)
EPS_END          = 0.02      # final exploration floor
EPS_DECAY        = 0.996     # per-episode multiplicative decay
HIDDEN           = 256       # hidden layer width
GRAD_CLIP        = 10.0      # max gradient norm


# ── Learning Rate Scheduler ────────────────────────────────────────────────
LR_STEP_SIZE     = 400       # episodes between LR reductions
LR_GAMMA         = 0.5       # LR multiplier at each reduction


# ── Training ────────────────────────────────────────────────────────────────
NUM_EPISODES     = 690     # total training episodes
EVAL_INTERVAL    = 50        # episodes between evaluations
EVAL_EPISODES    = 5         # episodes per evaluation run
LOG_INTERVAL     = 10        # episodes between console logs
CHECKPOINT_PATH  = "best_traffic_agent_dqn_ev.pth"
