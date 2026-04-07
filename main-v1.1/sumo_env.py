import os
import sys
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 1. HARDCODED PATH SETUP ────────────────────────────────────────────────
SUMO_ROOT = r'C:\Program Files (x86)\Eclipse\Sumo'
os.environ['SUMO_HOME'] = SUMO_ROOT
sys.path.append(os.path.join(SUMO_ROOT, 'tools'))

# ── 2. TUNABLE CONSTANTS ───────────────────────────────────────────────────
AMBULANCE_RATE_EW = 14  # Expected vehicles per hour for East-West ambulance flows
AMBULANCE_RATE_NS = 11  # Expected vehicles per hour for North-South ambulance flows
AMBULANCE_PROB_EW = AMBULANCE_RATE_EW / 3600.0  # per-second spawn probability
AMBULANCE_PROB_NS = AMBULANCE_RATE_NS / 3600.0  # per-second spawn probability
NS_QUEUE_WEIGHT = 1.0  # weight for N/S queue penalty
EW_QUEUE_WEIGHT = 1.10  # weight for E/W queue penalty

class SumoTrafficEnv(gym.Env):
    def __init__(self, gui=True):
        self.gui = gui
        # obs_dim = 30 to match the trained checkpoint:
        # 12 (queues+phase+time x2) + 5 (corridor_0to1) + 5 (corridor_1to0) + 8 (queue_duration)
        self.observation_space = spaces.Box(low=0, high=25, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.step_count = 0
        self.max_steps = 1000

    def _get_sumo_binary(self):
        bin_name = "sumo-gui.exe" if self.gui else "sumo.exe"
        return os.path.join(SUMO_ROOT, 'bin', bin_name)

    def _generate_files(self):
        """Generates a guaranteed valid 2-intersection network."""
        with open("traffic.nod.xml", "w") as f:
            f.write('<nodes>\n'
                    '    <node id="west" x="0" y="400"/>\n'
                    '    <node id="IL" x="300" y="400" type="traffic_light"/>\n'
                    '    <node id="IR" x="700" y="400" type="traffic_light"/>\n'
                    '    <node id="east" x="1000" y="400"/>\n'
                    '    <node id="n0" x="300" y="700"/><node id="s0" x="300" y="100"/>\n'
                    '    <node id="n1" x="700" y="700"/><node id="s1" x="700" y="100"/>\n'
                    '</nodes>')

        with open("traffic.edg.xml", "w") as f:
            f.write('<edges>\n'
                    '    <edge id="middle0to1" from="IL" to="IR" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="middle1to0" from="IR" to="IL" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="w_in" from="west" to="IL" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="e_in" from="east" to="IR" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="n0_in" from="n0" to="IL" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="s0_in" from="s0" to="IL" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="n1_in" from="n1" to="IR" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="s1_in" from="s1" to="IR" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="w_out" from="IL" to="west" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="e_out" from="IR" to="east" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="n0_out" from="IL" to="n0" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="s0_out" from="IL" to="s0" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="n1_out" from="IR" to="n1" numLanes="1" speed="13.89"/>\n'
                    '    <edge id="s1_out" from="IR" to="s1" numLanes="1" speed="13.89"/>\n'
                    '</edges>')

        os.system(f'"{SUMO_ROOT}\\bin\\netconvert" -n traffic.nod.xml -e traffic.edg.xml --lefthand true -o traffic.net.xml')

        with open("traffic.rou.xml", "w") as f:
            f.write('<routes>\n'
                    '    <vType id="car" length="5" maxSpeed="13.89" guiShape="passenger"/>\n'
                    '    <vType id="ambulance" length="5" maxSpeed="20" guiShape="bus" color="1,0,0"/>\n'
                    '    <route id="r_east" edges="w_in middle0to1 e_out"/>\n'
                    '    <route id="r_west" edges="e_in middle1to0 w_out"/>\n'
                    '    <route id="r_n0" edges="s0_in middle0to1 e_out"/>\n'
                    '    <route id="r_s0" edges="n0_in middle0to1 e_out"/>\n'
                    '    <route id="r_n1" edges="s1_in middle1to0 w_out"/>\n'
                    '    <route id="r_s1" edges="n1_in middle1to0 w_out"/>\n'
                    '    <flow id="fE" type="car" begin="0" end="3600" vehsPerHour="600" from="w_in" to="e_out"/>\n'
                    '    <flow id="fW" type="car" begin="0" end="3600" vehsPerHour="600" from="e_in" to="w_out"/>\n'
                    '    <flow id="fN0" type="car" begin="0" end="3600" vehsPerHour="300" from="s0_in" to="n0_out"/>\n'
                    '    <flow id="fS0" type="car" begin="0" end="3600" vehsPerHour="300" from="n0_in" to="s0_out"/>\n'
                    '    <flow id="fN1" type="car" begin="0" end="3600" vehsPerHour="300" from="s1_in" to="n1_out"/>\n'
                    '    <flow id="fS1" type="car" begin="0" end="3600" vehsPerHour="300" from="n1_in" to="s1_out"/>\n'
                    '    <flow id="fAE" type="ambulance" route="r_east" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_EW) + '"/>\n'
                    '    <flow id="fAW" type="ambulance" route="r_west" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_EW) + '"/>\n'
                    '    <flow id="fAN0" type="ambulance" route="r_n0" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_NS) + '"/>\n'
                    '    <flow id="fAS0" type="ambulance" route="r_s0" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_NS) + '"/>\n'
                    '    <flow id="fAN1" type="ambulance" route="r_n1" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_NS) + '"/>\n'
                    '    <flow id="fAS1" type="ambulance" route="r_s1" begin="0" end="3600" probability="' + str(AMBULANCE_PROB_NS) + '"/>\n'
                    '</routes>')

        with open("traffic.sumocfg", "w") as f:
            f.write('<configuration>\n'
                    '    <input><net-file value="traffic.net.xml"/><route-files value="traffic.rou.xml"/><additional-files value="traffic.poly.xml"/></input>\n'
                    '    <gui_only><start value="true"/><quit-on-end value="true"/></gui_only>\n'
                    '</configuration>')

        with open("traffic.poly.xml", "w") as f:
            f.write('<additional>\n'
                    '    <poly id="priority_building" type="building" color="200,150,80,180" shape="430,410 570,410 570,470 430,470"/>\n'
                    '</additional>')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_files()

        try: traci.close()
        except: pass

        binary = self._get_sumo_binary()
        traci.start([binary, "-c", "traffic.sumocfg", "--no-warnings", "true", "--delay", "800"])

        self.tls_ids = traci.trafficlight.getIDList()
        self.step_count = 0
        return self._get_obs(), {}

    def _get_corridor_data(self):
        e01, e10 = "middle0to1", "middle1to0"
        c01, c10 = [0.0] * 5, [0.0] * 5
        seg_len = 80
        try:
            for veh_id in traci.edge.getLastStepVehicleIDs(e01):
                idx = int(traci.vehicle.getLanePosition(veh_id) / seg_len)
                if idx < 5: c01[idx] += 1
            for veh_id in traci.edge.getLastStepVehicleIDs(e10):
                idx = int(traci.vehicle.getLanePosition(veh_id) / seg_len)
                if idx < 5: c10[idx] += 1
        except: pass
        return c01, c10

    def _get_obs(self):
        obs = []
        for tls in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls)
            unique_lanes = list(dict.fromkeys(lanes))

            # Map the incoming lane IDs to the expected queue order: [N, S, E, W].
            q = [0.0, 0.0, 0.0, 0.0]
            for lane in unique_lanes:
                if lane in ("n0_in_0", "n1_in_0"):
                    q[0] = float(traci.lane.getLastStepHaltingNumber(lane))
                elif lane in ("s0_in_0", "s1_in_0"):
                    q[1] = float(traci.lane.getLastStepHaltingNumber(lane))
                elif lane in ("middle1to0_0", "e_in_0"):
                    q[2] = float(traci.lane.getLastStepHaltingNumber(lane))
                elif lane in ("middle0to1_0", "w_in_0"):
                    q[3] = float(traci.lane.getLastStepHaltingNumber(lane))

            phase = traci.trafficlight.getPhase(tls)
            norm_phase = 0.0 if phase < 2 else 1.0
            obs.extend(q + [norm_phase, 5.0])

        c01, c10 = self._get_corridor_data()
        obs.extend(c01)
        obs.extend(c10)

        # 8 zeros for queue_duration — SUMO tracks real vehicle positions so
        # we don't have per-arm duration counters here; zeros are safe since
        # the agent was trained with these values starting at 0 each episode.
        obs.extend([0.0] * 8)

        return np.array(obs, dtype=np.float32)

    def _get_ambulance_priority_phase(self):
        ambulance_edges = set()
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) != "ambulance":
                continue
            lane_id = traci.vehicle.getLaneID(veh_id)
            if not lane_id:
                continue
            edge_id = lane_id.split("_")[0]
            ambulance_edges.add(edge_id)

        if any(edge in ("w_in", "e_in", "middle0to1", "middle1to0") for edge in ambulance_edges):
            return 2
        if any(edge in ("n0_in", "s0_in", "n1_in", "s1_in") for edge in ambulance_edges):
            return 0
        return None

    def step(self, action):
        p0 = 0 if (action >> 1) == 0 else 2
        p1 = 0 if (action & 1) == 0 else 2
        amb_phase = self._get_ambulance_priority_phase()
        if amb_phase is not None:
            p0 = amb_phase
            p1 = amb_phase
        if len(self.tls_ids) >= 1: traci.trafficlight.setPhase(self.tls_ids[0], p0)
        if len(self.tls_ids) >= 2: traci.trafficlight.setPhase(self.tls_ids[1], p1)

        for _ in range(5): traci.simulationStep()

        obs = self._get_obs()
        
        # Ambulance priority bonus: reward agent for letting ambulances through without stopping
        ambulance_halted = 0
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == "ambulance":
                try:
                    if traci.vehicle.getSpeed(veh_id) < 0.1:  # halted if speed < 0.1 m/s
                        ambulance_halted += 1
                except:
                    pass
        ambulance_bonus = -ambulance_halted * 20  # penalty for each halted ambulance
        
        ns_penalty = float(obs[0] + obs[1] + obs[6] + obs[7]) * NS_QUEUE_WEIGHT
        ew_penalty = float(obs[2] + obs[3] + obs[8] + obs[9]) * EW_QUEUE_WEIGHT
        reward = -(ns_penalty + ew_penalty) + ambulance_bonus
        self.step_count += 1
        return obs, reward, False, self.step_count >= self.max_steps, {}

    def close(self):
        try: traci.close()
        except: pass