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
            "-E5_0",                    # EAST → TL  (E5 building access, exits via -E5)
            "-465932558#1.34_0",        # NORTH lane 0 (left-turn only, dead — no feeder)
            "-465932558#1.34_1",        # NORTH lane 1 (straight, Phase 9)
            "-465932558#1.34_2",        # NORTH lane 2 (right-turn, Phase 6)
            "470773638#1_0",            # WEST → TL  (side road, Phase 12)
            "465932558#0_0",            # SOUTH lane 0 (straight, Phase 9)
            "465932558#0_1",            # SOUTH lane 1 (right-turn to E5, Phase 6)
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

        # ==== Action Space: active green phases only ====
        # Phase 3 (NORTH left turn to E5) is excluded: no vehicles can reach lane 0
        # of -465932558#1.34 from the upstream connections, making it a dead phase.
        # Map RL action indices (0..3) -> actual SUMO green phase indices
        self.phase_map = [0, 6, 9, 12]
        self.action_space = Discrete(len(self.phase_map))

        # ==== Observation space ====
        # Per lane: halting count (queue) + max accumulated waiting time
        # Global:   current phase (one-hot, 4 phases) + time since last switch
        n_lanes = len(self.incoming_lanes)
        n_phases = len(self.phase_map)
        obs_dim = n_lanes * 2 + n_phases + 1   # 7*2 + 4 + 1 = 19
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # timing / episode control
        self.step_length = step_length
        self.warmup_steps = warmup_steps
        self.max_episode_steps = max_episode_steps

        # safety / timing (SLOW mode defaults)
        self.yellow_duration = 4   # seconds of yellow
        self.red_duration = 3      # seconds of all-red (intergreen)
        self.min_green_time = 20   # minimum time to hold a green before switching

        # bookkeeping
        self._step_count = 0
        self._last_change_step = 0

        # normalization constants (tune for your scenario)
        self.MAX_QUEUE = 30.0
        self.MAX_TRAVEL_TIME = 300.0
        self.MAX_THROUGHPUT = 20.0
        self.MAX_PRESSURE = 40.0

        # Phase transition maps for programID="fixed" in tls.add.xml
        # GREEN_PHASES active = [0, 6, 9, 12]  (Phase 3 removed — dead phase)
        self.YELLOW_MAP = {0: 1, 6: 7, 9: 10, 12: 13}
        self.RED_MAP    = {0: 2, 6: 8, 9: 11, 12: 14}

        # Reverse map: any phase index -> its green group base
        self.GROUP_MAP = {
            0: 0,  1: 0,  2: 0,
            6: 6,  7: 6,  8: 6,
            9: 9,  10: 9, 11: 9,
            12: 12, 13: 12, 14: 12,
            # Phase 3 group still mapped so setPhase calls don't crash if SUMO
            # is mid-transition when we read the current phase
            3: 0,  4: 0,  5: 0,
        }

        # Signal sequencing — clear turning queues before straight phases.
        # Right-turn lanes on the SOUTH and NORTH approaches share upstream space
        # with through lanes. When their queue grows long enough it physically
        # blocks straight vehicles even while their green is active.
        # Fix: if the turning queue exceeds the threshold, run Phase 6 (turns)
        # for turn_clear_duration steps before switching to Phase 9 (straight).
        self.STRAIGHT_PHASES     = {9}
        self.TURN_CLEAR_PHASE    = 6
        self.TURN_BLOCKING_LANES = [
            "-465932558#1.34_2",  # NORTH right-turn  → can block NORTH straight
            "465932558#0_1",      # SOUTH right-turn  → can block SOUTH straight
        ]
        self.turn_clear_threshold = 3   # halting vehicles needed to trigger clearing
        self.turn_clear_duration  = 15  # steps to hold turn-clear phase

    # ============================
    # START SUMO
    # ============================
    def start_sumo(self):
        import os, sys, shutil
        sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
        ext = ".exe" if sys.platform == "win32" else ""
        binary = f"sumo-gui{ext}" if self.use_gui else f"sumo{ext}"
        sumo_bin = os.path.join(sumo_home, "bin", binary)
        if not os.path.isfile(sumo_bin):
            sumo_bin = shutil.which(binary) or sumo_bin

        if not os.path.isfile(self.sumo_cfg):
            raise FileNotFoundError(
                f"SUMO config not found: {self.sumo_cfg}\n"
                f"Working directory: {os.getcwd()}\n"
                f"Files here: {os.listdir(os.path.dirname(self.sumo_cfg) or '.')}"
            )

        cmd = [sumo_bin, "-c", self.sumo_cfg,
               "--no-step-log", "--waiting-time-memory", "1000"]

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
        """
        Enriched, normalised observation:
        - per-lane halting count (queue), normalised by MAX_QUEUE
        - per-lane mean accumulated waiting time, normalised by MAX_TRAVEL_TIME
        - current green group as one-hot over phase_map
        - time since last phase switch, normalised
        """
        obs = []

        # 1) per-lane queue (halting number) -- matches the reward signal
        for lane in self.incoming_lanes:
            try:
                q = traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                q = 0
            obs.append(float(np.clip(q / self.MAX_QUEUE, 0.0, 1.0)))

        # 2) per-lane MAX accumulated waiting time -- one starving vehicle dominates
        for lane in self.incoming_lanes:
            try:
                vids = traci.lane.getLastStepVehicleIDs(lane)
                if vids:
                    waits = [traci.vehicle.getAccumulatedWaitingTime(v) for v in vids]
                    w = float(max(waits))
                else:
                    w = 0.0
            except Exception:
                w = 0.0
            obs.append(float(np.clip(w / self.MAX_TRAVEL_TIME, 0.0, 1.0)))

        # 3) current green group as one-hot -- the agent must know what it is doing
        try:
            current_phase = int(traci.trafficlight.getPhase(self.tl_id))
            current_group = self.GROUP_MAP.get(current_phase, self.phase_map[0])
        except Exception:
            current_group = self.phase_map[0]
        for g in self.phase_map:
            obs.append(1.0 if g == current_group else 0.0)

        # 4) time since last switch -- lets the agent learn to hold a phase
        elapsed = self._step_count - self._last_change_step
        obs.append(float(np.clip(elapsed / 60.0, 0.0, 1.0)))

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
        """Reward combining queue, pressure, and max-vehicle starvation penalty."""
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

            # max single-vehicle starvation -- catches the one neglected lane
            if vehicle_ids:
                max_wait = float(max(traci.vehicle.getAccumulatedWaitingTime(v) for v in vehicle_ids))
            else:
                max_wait = 0.0
            max_wait_n = np.clip(max_wait / self.MAX_TRAVEL_TIME, 0.0, 1.0)

            # Compose: queue + pressure weighted higher; starvation term catches neglected lanes
            reward = -1.0 * (0.5 * avg_queue_n + 0.3 * pressure_n + 0.2 * max_wait_n)

            return float(np.clip(reward, -10.0, 0.0))
        except Exception:
            return -1.0

    # ============================
    # SET PHASE — safe dictionary-based transitions
    # ============================
    def _get_turning_queue(self):
        """Total halting vehicles in the right-turn lanes that cause spillback."""
        total = 0
        for lane in self.TURN_BLOCKING_LANES:
            try:
                total += traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
        return total

    def _do_transition(self, from_group, to_green):
        """Yellow → all-red → green transition between two phase groups."""
        yellow_phase = self.YELLOW_MAP.get(from_group, None)
        if yellow_phase is not None:
            try:
                traci.trafficlight.setPhase(self.tl_id, int(yellow_phase))
            except Exception:
                pass
            for _ in range(self.yellow_duration):
                traci.simulationStep()

        red_phase = self.RED_MAP.get(from_group, None)
        if red_phase is not None:
            try:
                traci.trafficlight.setPhase(self.tl_id, int(red_phase))
            except Exception:
                pass
            for _ in range(self.red_duration):
                traci.simulationStep()

        try:
            traci.trafficlight.setPhase(self.tl_id, int(to_green))
        except Exception:
            pass

    def _set_phase(self, action):
        """
        Safe phase transition with turn-before-straight sequencing.

        When the agent requests a straight-traffic phase (Phase 9) and the
        right-turn lanes hold a queue above turn_clear_threshold, the controller
        first runs the turn-clearing phase (Phase 6) for turn_clear_duration steps
        before switching to the straight phase. This prevents right-turn spillback
        from physically blocking through vehicles during their own green.

        Sequence for affected transitions:
          current → [yellow → red] → Phase 6 (turns clear, held N steps)
                  → [yellow → red] → Phase 9 (straight green)

        All other transitions follow the standard safe sequence:
          current → [yellow → red] → target green
        """
        if action is None:
            return
        try:
            target_green = int(self.phase_map[int(action)])
        except Exception:
            return

        try:
            current_phase = int(traci.trafficlight.getPhase(self.tl_id))
        except Exception:
            current_phase = target_green

        if (self._step_count - self._last_change_step) < self.min_green_time:
            return

        if current_phase == target_green:
            return

        current_group = self.GROUP_MAP.get(current_phase, target_green)

        # Turn-before-straight sequencing:
        # Only insert turn-clear step when all three conditions hold:
        #   1. target is a straight phase
        #   2. we are not already running the turn-clear phase (avoid no-op cycle)
        #   3. turning queue exceeds threshold (avoid wasting green time when clear)
        if (target_green in self.STRAIGHT_PHASES
                and current_group != self.TURN_CLEAR_PHASE
                and self._get_turning_queue() >= self.turn_clear_threshold):
            self._do_transition(current_group, self.TURN_CLEAR_PHASE)
            for _ in range(self.turn_clear_duration):
                traci.simulationStep()
            self._do_transition(self.TURN_CLEAR_PHASE, target_green)
        else:
            self._do_transition(current_group, target_green)

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

        # Ensure our custom TLS program is active (net file has programID="0")
        try:
            traci.trafficlight.setProgram(self.tl_id, "fixed")
        except Exception:
            pass

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
