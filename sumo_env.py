import os
import numpy as np
import gymnasium as gym
import traci

from traffic_scenario import TrafficScenario
import traffic_injector
import green_wave
from tl_programs import apply_tl_programs

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SUMO_CFG = "triple.sumocfg"
TL_IDS = ["6073919354", "6073919354_B", "6073919354_C"]

DELTA_T = 3           # RL step duration (seconds)
YELLOW_DUR = 4        # yellow duration before next green
MIN_GREEN = 8         # min steps of green before a switch is allowed
MIN_SWITCH_QUEUE = 8  # target lanes must have at least this many halting vehicles to justify a switch
MAX_SIM_STEPS = 2000

N_LANES_PER_TL = 8
OBS_MAX_QUEUE = 15.0

MAX_QUEUE = 15.0
MAX_PRESSURE = 40.0
MAX_TRAVEL_TIME = 300.0

# Major green phases the agent can select (index into list = action value)
MAJOR_GREEN_PHASES = {
    "6073919354":   [0, 4],
    "6073919354_B": [0, 4],
    "6073919354_C": [0, 2, 4],
}

# Yellow phase that immediately follows each major green phase (keyed by SUMO phase index)
YELLOW_AFTER = {
    "6073919354":   {0: 1, 4: 5},
    "6073919354_B": {0: 1, 4: 5},
    "6073919354_C": {0: 1, 2: 3, 4: 5},
}


class SumoEnv(gym.Env):
    _instance_count = 0  # unique label per instance so train/eval don't collide

    def __init__(self, use_gui=False, scenario_name=None):
        super().__init__()

        SumoEnv._instance_count += 1
        self._label = f"sumo_{SumoEnv._instance_count}"

        self.use_gui = use_gui
        self.scenario_name = scenario_name

        obs_size = len(TL_IDS) * (N_LANES_PER_TL + 2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 3])

        # runtime state (populated at reset)
        self._controlled_lanes = {}
        self._outgoing_lanes = {}
        self._phase_lanes = {}    # action_idx → list of incoming lanes served by that phase
        self._phase_state = {}    # "GREEN" or "YELLOW"
        self._state_timer = {}    # steps remaining in yellow
        self._current_action = {}
        self._pending_action = {}
        self._green_time = {}     # steps held at current green

        self._sim_step = 0
        self._sumo_running = False

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        scenario = (
            TrafficScenario(self.scenario_name)
            if self.scenario_name
            else TrafficScenario.random()
        )

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", SUMO_CFG, "--no-warnings", "--no-step-log"],
                    label=self._label)
        traci.switch(self._label)

        self._sumo_running = True
        self._sim_step = 0

        apply_tl_programs()
        traffic_injector.init(scenario)

        gw = green_wave.create(TL_IDS)
        gw.bootstrap(free_flow_kmh=50.0)
        gw.enabled = True

        for tl in TL_IDS:
            self._controlled_lanes[tl] = list(
                dict.fromkeys(traci.trafficlight.getControlledLanes(tl))
            )
            links_tl = traci.trafficlight.getControlledLinks(tl)
            out = []
            for link_group in links_tl:
                for (_, to_lane, _) in link_group:
                    if to_lane and to_lane not in out:
                        out.append(to_lane)
            self._outgoing_lanes[tl] = out

            # Map each action index → incoming lanes served by that green phase
            try:
                programs = traci.trafficlight.getAllProgramLogics(tl)
                phase_strs = {i: p.state for i, p in enumerate(programs[0].phases)}
            except Exception:
                phase_strs = {}
            self._phase_lanes[tl] = {}
            for act_idx, ph_idx in enumerate(MAJOR_GREEN_PHASES[tl]):
                state_str = phase_strs.get(ph_idx, "")
                served = []
                for li, ch in enumerate(state_str):
                    if ch in ('G', 'g') and li < len(links_tl):
                        for (from_lane, _, _) in links_tl[li]:
                            if from_lane and from_lane not in served:
                                served.append(from_lane)
                self._phase_lanes[tl][act_idx] = served

            self._phase_state[tl] = "GREEN"
            self._state_timer[tl] = 0
            self._current_action[tl] = 0
            self._pending_action[tl] = 0
            self._green_time[tl] = 0
            traci.trafficlight.setPhase(tl, MAJOR_GREEN_PHASES[tl][0])

        for _ in range(20):
            traci.simulationStep()
            self._sim_step += 1
            traffic_injector.inject(self._sim_step)

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
        traci.switch(self._label)
        for i, tl in enumerate(TL_IDS):
            req = int(action[i])
            state = self._phase_state[tl]

            if state == "GREEN":
                self._green_time[tl] += 1
                if req != self._current_action[tl] and self._green_time[tl] >= MIN_GREEN:
                    # Only switch if target lanes have enough queued vehicles
                    target_lanes = self._phase_lanes.get(tl, {}).get(req, [])
                    target_queue = sum(
                        traci.lane.getLastStepHaltingNumber(l) for l in target_lanes
                    ) if target_lanes else MIN_SWITCH_QUEUE
                    if target_queue >= MIN_SWITCH_QUEUE:
                        cur_green = MAJOR_GREEN_PHASES[tl][self._current_action[tl]]
                        yellow = YELLOW_AFTER[tl][cur_green]
                        traci.trafficlight.setPhase(tl, yellow)
                        self._phase_state[tl] = "YELLOW"
                        self._state_timer[tl] = YELLOW_DUR
                        self._pending_action[tl] = req

            elif state == "YELLOW":
                self._state_timer[tl] -= 1
                if self._state_timer[tl] <= 0:
                    # yellow expired — go straight to the next green
                    target = MAJOR_GREEN_PHASES[tl][self._pending_action[tl]]
                    traci.trafficlight.setPhase(tl, target)
                    self._phase_state[tl] = "GREEN"
                    self._current_action[tl] = self._pending_action[tl]
                    self._green_time[tl] = 0

        for _ in range(DELTA_T):
            traci.simulationStep()
            self._sim_step += 1
            traffic_injector.inject(self._sim_step)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._sim_step >= MAX_SIM_STEPS

        if terminated:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        return obs, reward, terminated, False, {}

    def close(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    # ─────────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self):
        parts = []
        for tl in TL_IDS:
            vals = []
            for lane in self._controlled_lanes[tl]:
                try:
                    h = traci.lane.getLastStepHaltingNumber(lane)
                    vals.append(min(h / OBS_MAX_QUEUE, 1.0))
                except Exception:
                    vals.append(0.0)
            while len(vals) < N_LANES_PER_TL:
                vals.append(0.0)
            parts.extend(vals[:N_LANES_PER_TL])
            # Normalise action index to [0, 1] so it fits the declared obs bounds
            n_phases = len(MAJOR_GREEN_PHASES[tl])
            parts.append(self._current_action[tl] / max(n_phases - 1, 1))
            parts.append(0.0 if self._phase_state[tl] == "GREEN" else 1.0)
        return np.array(parts, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self):
        try:
            # queue component
            queues = []
            for tl in TL_IDS:
                for lane in self._controlled_lanes[tl]:
                    queues.append(traci.lane.getLastStepHaltingNumber(lane))
            avg_q = float(np.clip(np.mean(queues) / MAX_QUEUE if queues else 0.0, 0.0, 1.0))

            # pressure component (|incoming - outgoing| per TL)
            pressure = 0.0
            for tl in TL_IDS:
                inc = sum(traci.lane.getLastStepVehicleNumber(l) for l in self._controlled_lanes[tl])
                out = sum(traci.lane.getLastStepVehicleNumber(l) for l in self._outgoing_lanes[tl])
                pressure += abs(inc - out)
            pressure = float(np.clip(pressure / (MAX_PRESSURE * len(TL_IDS)), 0.0, 1.0))

            # waiting time component
            vehs = traci.vehicle.getIDList()
            waits = [traci.vehicle.getAccumulatedWaitingTime(v) for v in vehs]
            travel = float(np.clip(np.mean(waits) / MAX_TRAVEL_TIME if waits else 0.0, 0.0, 1.0))

            reward = -(0.6 * avg_q + 0.4 * pressure + 0.2 * travel)

            # safety penalties
            reward -= traci.simulation.getCollidingVehiclesNumber() * 2.0
            reward -= traci.simulation.getStartingTeleportNumber() * 1.0

            return float(np.clip(reward, -5.0, 0.0))

        except Exception:
            return -1.0
