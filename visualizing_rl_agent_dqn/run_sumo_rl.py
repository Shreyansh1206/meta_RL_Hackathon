import sys
import os

# Insert the DQN directory to the python path to use the exact same agent, config, and visualizer.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "RL-Traffic-Lights-DQN")
    ),
)

import numpy as np
import traci
from agent import DQNAgent
from visualize import TrafficRenderer
from config import MAX_QUEUE, CORRIDOR_DELAY

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & MAPPING
# ──────────────────────────────────────────────────────────────────────────────
SUMO_CONFIG = "simulation.sumocfg"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "RL-Traffic-Lights-DQN",
    "best_traffic_agent_dqn.pth",
)
GUI_MODE = True

# Intersection IDs in SUMO
INTERSECTIONS = ["int_left", "int_right"]

# Mapping ARMS: 0=N, 1=S, 2=E, 3=W
ARMS_MAPPING = {
    "int_left": {
        0: ["ped1n2il_0"],
        1: ["ped1s2il_0"],
        2: ["hw_ir2il_0", "hw_ir2il_1"],
        3: ["hw_w2il_0", "hw_w2il_1"],
    },
    "int_right": {
        0: ["ped2n2ir_0"],
        1: ["ped2s2ir_0"],
        2: ["hw_e2ir_0", "hw_e2ir_1"],
        3: ["hw_il2ir_0", "hw_il2ir_1"],
    },
}

CORRIDORS = {"0to1": ["hw_il2ir"], "1to0": ["hw_ir2il"]}

# Phase Mapping: RL 0 (NS Green) -> SUMO 2, RL 1 (EW Green) -> SUMO 0
# FIX: SUMO netconvert assigns Phase 0 to the main road (Highway EW) and Phase 2 to the minor road (Arterials NS).
SUMO_PHASE_NS_GREEN = 2
SUMO_PHASE_EW_GREEN = 0

# RL Step Duration
RL_STEP_SECONDS = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# MOCK ENVIRONMENT FOR VISUALIZER
# ──────────────────────────────────────────────────────────────────────────────
class MockEnv:
    """A wrapper that mimics the RL TrafficEnv for the TrafficRenderer."""

    def __init__(self):
        self.queues = np.zeros((2, 4), dtype=np.int32)
        self.phases = np.zeros(2, dtype=np.int32)
        self.step_count = 0
        self.corridor_0to1 = [0] * CORRIDOR_DELAY
        self.corridor_1to0 = [0] * CORRIDOR_DELAY


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def get_queue_lengths(intersection_id):
    queues = []
    for arm_idx in range(4):
        lanes = ARMS_MAPPING[intersection_id][arm_idx]
        halted = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        # SCALE/CLIP to match model's expected MAX_QUEUE
        queues.append(min(float(halted), float(MAX_QUEUE)))
    return queues


def get_corridor_states(corridor_id):
    edges = CORRIDORS[corridor_id]
    segment_counts = [0.0] * CORRIDOR_DELAY
    for edge in edges:
        lane_count = traci.edge.getLaneNumber(edge)
        edge_length = traci.lane.getLength(f"{edge}_0")
        seg_len = edge_length / float(CORRIDOR_DELAY)
        for l_idx in range(lane_count):
            lane_id = f"{edge}_{l_idx}"
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for veh in vehicles:
                pos = traci.vehicle.getLanePosition(veh)
                seg_idx = min(int(pos / seg_len), CORRIDOR_DELAY - 1)
                segment_counts[seg_idx] += 1.0
    return segment_counts


def run_simulation():
    # 1. Initialize Agent & Visualizer
    obs_dim = 12 + (2 * CORRIDOR_DELAY)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=4)
    try:
        agent.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Warning: model not found at {MODEL_PATH}")
    agent.epsilon = 0.0

    renderer = TrafficRenderer()
    mock_env = MockEnv()

    # 2. Start SUMO
    sumo_binary = "sumo-gui" if GUI_MODE else "sumo"
    traci.start([sumo_binary, "-c", SUMO_CONFIG, "--start"])

    # Set nice initial view
    try:
        traci.gui.setZoom("View #0", 350)
        traci.gui.setOffset("View #0", 500, 300)
    except Exception:
        pass

    step = 0
    total_waiting_time = 0
    total_cars_arrived = 0
    time_in_phase = [0, 0]
    current_phases_rl = [0, 0]

    print(
        "Simulation started. Use the Pygame window to see the model's internal view and SUMO for physical view."
    )

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # --- COLLECT OBSERVATION & UPDATE MOCK_ENV ---
            obs = []
            for i, intersection_id in enumerate(INTERSECTIONS):
                queues = get_queue_lengths(intersection_id)
                mock_env.queues[i] = np.array(queues, dtype=np.int32)
                obs.extend(queues)
                obs.append(float(current_phases_rl[i]))
                obs.append(float(time_in_phase[i]))

            c0to1 = get_corridor_states("0to1")
            c1to0 = get_corridor_states("1to0")
            mock_env.corridor_0to1 = c0to1
            # The renderer might expect list or deque, just using list
            mock_env.corridor_1to0 = c1to0

            obs.extend(c0to1)
            obs.extend(c1to0)

            # --- AGENT DECISION ---
            state = np.array(obs, dtype=np.float32)
            action = agent.act(state)

            # --- APPLY ACTIONS ---
            target_phases_rl = [action >> 1, action & 1]
            for i, intersection_id in enumerate(INTERSECTIONS):
                if target_phases_rl[i] != current_phases_rl[i]:
                    current_phases_rl[i] = target_phases_rl[i]
                    time_in_phase[i] = 0
                else:
                    time_in_phase[i] += 1

                sumo_phase = (
                    SUMO_PHASE_NS_GREEN
                    if current_phases_rl[i] == 0
                    else SUMO_PHASE_EW_GREEN
                )
                traci.trafficlight.setPhase(intersection_id, sumo_phase)
                mock_env.phases[i] = current_phases_rl[i]

            # --- RENDERER (Original Pygame View from visualize.py) ---
            mock_env.step_count = step
            renderer.draw(mock_env)

            # --- SIMULATION STEP ---
            for _ in range(int(RL_STEP_SECONDS)):
                traci.simulationStep()
                for veh in traci.vehicle.getIDList():
                    total_waiting_time += traci.vehicle.getWaitingTime(veh) / 3600.0
                total_cars_arrived += traci.simulation.getArrivedNumber()

            step += 1
            max_q_arm = np.max(mock_env.queues)  # The maximum queue across all 8 arms
            if step % 10 == 0:
                print(
                    f"Step {step} | Max Arm Queue: {int(max_q_arm)}/20 | Total (8 arms): {(int(mock_env.queues.sum()))} | Discharged: {total_cars_arrived}"
                )

            # Safety: If SUMO is jammed, don't let it run forever
            if max_q_arm >= 50:
                print(
                    "WARNING: SUMO physics congestion exceeds model capability. Skipping some steps."
                )

    except traci.exceptions.FatalTraCIError:
        print("Simulation closed by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        try:
            renderer.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_simulation()
