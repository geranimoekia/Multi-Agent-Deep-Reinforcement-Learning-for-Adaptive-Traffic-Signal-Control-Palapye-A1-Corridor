"""
SumoEnv — corrected, RL-ready environment

Changes compared to previous version:
- Uses explicit dictionaries for phase transitions (GREEN -> YELLOW -> RED)
  so we never rely on `current + 1` or `current + 2`.
- Enforces full safe transition A: current_group -> yellow -> all-red -> target_green.
- Disables idle override during training (so PPO actions are always applied).
  Idle behavior can be re-enabled manually for evaluation (use_gui mode).
- Keeps Gymnasium API: reset(...) -> (obs, info), step(...) -> (obs, reward, terminated, truncated, info).
- Adds metrics in info and stable reward normalization.
- Enforces minimum green time to avoid rapid toggling.
"""

import time
import traci
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class SumoEnv(Env):
    def __init__(
        self,
        sumo_cfg="map.sumocfg",
        use_gui=False,
        warmup_steps=10,
        step_length=1,
        max_episode_steps=1800,
    ):
        super(SumoEnv, self).__init__()

        self.use_gui = use_gui
        self.sumo_cfg = sumo_cfg

        # ==== Incoming lanes for TL 6073919354 ====
        self.incoming_lanes = [
            "-E5_0",                    # WEST → TL
            "-465932558#1.34_0",        # NORTH Left
            "-465932558#1.34_1",        # NORTH Middle
            "-465932558#1.34_2",        # NORTH Right
            "470773638#1_0",            # EAST → TL
            "465932558#0_0",            # SOUTH Lane 1
            "465932558#0_1",            # SOUTH Lane 2
        ]

        # ==== Outgoing lanes ====
        self.outgoing_lanes = [
            "E5_0",
            "-465932558#0_0",
            "-465932558#0_1",
            "-470773638#1_0",
            "465932558#1_0",
        ]

        # Traffic light ID
        self.tl_id = "6073919354"

        # ==== Action Space: high-level green phases only ====
        # Map RL action indices (0..4) -> actual SUMO green phase indices
        self.phase_map = [0, 3, 6, 9, 12]
        self.action_space = Discrete(len(self.phase_map))

        # ==== Observation: queue length per lane ====
        self.observation_space = Box(low=0, high=200, shape=(7,), dtype=np.float32)

        # timing / episode control
        self.step_length = step_length
        self.warmup_steps = warmup_steps
        self.max_episode_steps = max_episode_steps

        # safety / timing (SLOW mode defaults)
        self.yellow_duration = 4   # seconds of yellow
        self.red_duration = 3      # seconds of all-red (intergreen)
        self.min_green_time = 8    # minimum time to hold a green before switching

        # bookkeeping
        self._step_count = 0
        self._last_change_step = 0

        # normalization constants (tune for your scenario)
        self.MAX_QUEUE = 30.0
        self.MAX_TRAVEL_TIME = 300.0
        self.MAX_THROUGHPUT = 20.0
        self.MAX_PRESSURE = 40.0

        # Phase dictionaries that match your SUMO tlLogic XML (explicit)
        # GREEN_PHASES = [0, 3, 6, 9, 12]
        self.YELLOW_MAP = {0: 1, 3: 4, 6: 7, 9: 10, 12: 13}
        self.RED_MAP = {0: 2, 3: 5, 6: 8, 9: 11, 12: 14}

        # Reverse map: any phase index -> its green group base
        self.GROUP_MAP = {
            0: 0, 1: 0, 2: 0,
            3: 3, 4: 3, 5: 3,
            6: 6, 7: 6, 8: 6,
            9: 9, 10: 9, 11: 9,
            12: 12, 13: 12, 14: 12
        }

    # ============================
    # START SUMO
    # ============================
    def start_sumo(self):
        if self.use_gui:
            cmd = ["sumo-gui", "-c", self.sumo_cfg]
        else:
            cmd = ["sumo", "-c", self.sumo_cfg]

        # ensure no leftover connection
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        traci.start(cmd)

    # ============================
    # OBSERVATION STATE
    # ============================
    def _get_observation(self):
        obs = []
        for lane in self.incoming_lanes:
            try:
                q = traci.lane.getLastStepVehicleNumber(lane)
            except Exception:
                q = 0
            obs.append(q)
        return np.array(obs, dtype=np.float32)

    # ============================
    # METRICS & REWARD
    # ============================
    def _compute_metrics(self):
        """Compute metrics: avg_delay, queues, throughput, stop_ratio, avg_travel_time."""
        metrics = {}
        try:
            vehicle_ids = traci.vehicle.getIDList()
            num_veh = len(vehicle_ids)

            # avg delay proxy
            if num_veh > 0:
                delays = []
                for vid in vehicle_ids:
                    try:
                        speed = traci.vehicle.getSpeed(vid)
                        allowed = traci.vehicle.getAllowedSpeed(vid)
                        delay = 1.0 - (speed / allowed) if allowed > 0 else 0.0
                        delays.append(float(np.clip(delay, 0.0, 1.0)))
                    except Exception:
                        pass
                avg_delay = float(np.mean(delays)) if delays else 0.0
            else:
                avg_delay = 0.0

            # queues (halting numbers) per incoming lane
            queues = []
            for lane in self.incoming_lanes:
                try:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                except Exception:
                    q = 0
                queues.append(int(q))
            avg_queue = float(np.mean(queues)) if queues else 0.0

            # throughput proxy (outgoing lane vehicle counts)
            throughput_per_lane = []
            for lane in self.outgoing_lanes:
                try:
                    t = traci.lane.getLastStepVehicleNumber(lane)
                except Exception:
                    t = 0
                throughput_per_lane.append(int(t))
            throughput_total = int(sum(throughput_per_lane))

            # stop ratio (incoming lanes)
            stops = 0
            total = 0
            for lane in self.incoming_lanes:
                try:
                    vids = traci.lane.getLastStepVehicleIDs(lane)
                except Exception:
                    vids = []
                for vid in vids:
                    total += 1
                    try:
                        if traci.vehicle.getSpeed(vid) < 0.1:
                            stops += 1
                    except Exception:
                        pass
            stop_ratio = (stops / total) if total > 0 else 0.0

            # avg travel / waiting time
            travel_times = []
            for vid in vehicle_ids:
                try:
                    wt = traci.vehicle.getAccumulatedWaitingTime(vid)
                    travel_times.append(float(wt))
                except Exception:
                    pass
            avg_travel_time = float(np.mean(travel_times)) if travel_times else 0.0

            metrics = {
                "avg_delay": avg_delay,
                "queues": queues,
                "avg_queue": avg_queue,
                "throughput_per_lane": throughput_per_lane,
                "throughput_total": throughput_total,
                "stop_ratio": stop_ratio,
                "avg_travel_time": avg_travel_time,
                "num_vehicles": num_veh,
            }
        except Exception:
            metrics = {
                "avg_delay": 0.0,
                "queues": [0] * len(self.incoming_lanes),
                "avg_queue": 0.0,
                "throughput_per_lane": [0] * len(self.outgoing_lanes),
                "throughput_total": 0,
                "stop_ratio": 0.0,
                "avg_travel_time": 0.0,
                "num_vehicles": 0,
            }
        return metrics

    def _compute_reward(self):
        """Simplified, normalized reward combining queue & pressure & travel time."""
        try:
            vehicle_ids = traci.vehicle.getIDList()

            # avg queue normalized
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in self.incoming_lanes]
            avg_queue = float(np.mean(queues)) if queues else 0.0
            avg_queue_n = np.clip(avg_queue / self.MAX_QUEUE, 0.0, 1.0)

            # pressure normalized
            incoming = float(sum(traci.lane.getLastStepVehicleNumber(l) for l in self.incoming_lanes))
            outgoing = float(sum(traci.lane.getLastStepVehicleNumber(l) for l in self.outgoing_lanes))
            pressure = incoming - outgoing
            pressure_n = np.clip(abs(pressure) / self.MAX_PRESSURE, 0.0, 1.0)

            # avg travel time normalized
            travel_times = [traci.vehicle.getAccumulatedWaitingTime(v) for v in vehicle_ids]
            avg_travel_time = float(np.mean(travel_times)) if travel_times else 0.0
            travel_n = np.clip(avg_travel_time / self.MAX_TRAVEL_TIME, 0.0, 1.0)

            # Compose reward (prioritize queue & pressure)
            reward = -1.0 * (0.6 * avg_queue_n + 0.4 * pressure_n + 0.2 * travel_n)

            # Clip
            return float(np.clip(reward, -10.0, 0.0))
        except Exception:
            return -1.0

    # ============================
    # SET PHASE — safe dictionary-based transitions
    # ============================
    def _set_phase(self, action):
        """
        Full safe transition (A):
        current_group (green) -> its yellow -> its all-red -> target_group (green)
        This uses explicit YELLOW_MAP and RED_MAP so indices are always valid.
        NOTE: idle override is disabled during training; if you run with use_gui=True
        and want idle behaviour, enable it separately.
        """
        # Map RL action -> target SUMO green phase
        if action is None:
            return
        try:
            target_green = int(self.phase_map[int(action)])
        except Exception:
            return

        # read current SUMO phase
        try:
            current_phase = int(traci.trafficlight.getPhase(self.tl_id))
        except Exception:
            current_phase = target_green

        # enforce minimum green time
        if (self._step_count - self._last_change_step) < self.min_green_time:
            # do not change yet
            return

        # if already in target green, nothing to do
        if current_phase == target_green:
            return

        # Identify the current group's green base (fallback to target_green)
        current_group = self.GROUP_MAP.get(current_phase, target_green)

        # 1) set yellow for current_group
        yellow_phase = self.YELLOW_MAP.get(current_group, None)
        if yellow_phase is not None:
            try:
                traci.trafficlight.setPhase(self.tl_id, int(yellow_phase))
            except Exception:
                pass
            for _ in range(self.yellow_duration):
                traci.simulationStep()

        # 2) set all-red for current_group
        red_phase = self.RED_MAP.get(current_group, None)
        if red_phase is not None:
            try:
                traci.trafficlight.setPhase(self.tl_id, int(red_phase))
            except Exception:
                pass
            for _ in range(self.red_duration):
                traci.simulationStep()

        # 3) apply target green
        try:
            traci.trafficlight.setPhase(self.tl_id, int(target_green))
        except Exception:
            pass

        # record the step when change happened
        self._last_change_step = self._step_count

    # ============================
    # RESET
    # ============================
    def reset(self, *, seed=None, options=None):
        # close any existing connection
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        # start SUMO
        self.start_sumo()

        # warm-up simulation (spawn vehicles)
        for _ in range(self.warmup_steps):
            try:
                traci.simulationStep()
            except Exception:
                break

        # reset counters
        self._step_count = 0
        self._last_change_step = 0

        super().reset(seed=seed)
        obs = self._get_observation()
        return obs, {}

    # ============================
    # STEP (Gymnasium API)
    # ============================
    def step(self, action):
        try:
            # Apply action (safe transitions)
            self._set_phase(action)

            # advance simulation for one action interval
            for _ in range(self.step_length):
                traci.simulationStep()

            obs = self._get_observation()
            reward = self._compute_reward()

            self._step_count += 1

            # termination: allow full episode length before checking end
            sim_finished = False
            try:
                # only consider simulation finished if many steps have elapsed
                if self._step_count > 50:
                    sim_finished = traci.simulation.getMinExpectedNumber() == 0
            except Exception:
                sim_finished = False

            terminated = sim_finished or (self._step_count >= self.max_episode_steps)
            truncated = False

            info = {"metrics": self._compute_metrics()}
            return obs, reward, terminated, truncated, info

        except Exception as e:
            try:
                traci.close()
            except Exception:
                pass
            obs = self._get_observation()
            return obs, -1.0, True, False, {"error": str(e)}

    # ============================
    # CLOSE
    # ============================
    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    # quick sanity check (requires SUMO installed)
    env = SumoEnv(use_gui=False)
    obs, info = env.reset()
    print("initial obs:", obs, "info:", info)
    for i in range(5):
        a = env.action_space.sample()
        obs, r, t, trc, info = env.step(a)
        print("step", i, "action", a, "reward", r, "term", t, "info.metrics.num_vehicles", info.get("metrics", {}).get("num_vehicles"))
    env.close()
