import os
import random
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
SUMO_CFG = "network/triple.sumocfg"
TL_IDS = ["6073919354", "6073919354_B", "6073919354_C"]

DELTA_T = 3           # RL step duration (seconds)
YELLOW_DUR = 1        # _tick_phases calls while yellow ≈ 3 sim-steps
MIN_GREEN = 5         # min steps of green before a switch is allowed
MAX_GREEN = 40        # force rotation after this many calls ≈ 120 sim-steps (gridlock guard)
MAX_SIM_STEPS = 1200  # shorter episodes → tighter credit assignment

N_LANES_PER_TL = 8
N_OUT_LANES_PER_TL = 4
OBS_MAX_QUEUE = 15.0
LOG_OBS_MAX = float(np.log1p(OBS_MAX_QUEUE))  # precomputed for log scaling
OBS_MAX_WAIT = 300.0                           # seconds cap for per-lane waiting time
LOG_OBS_MAX_WAIT = float(np.log1p(OBS_MAX_WAIT))

MAX_QUEUE = 15.0

# Curriculum: episode thresholds → scenario pool
# Each env progresses independently; starts simple and adds complexity
CURRICULUM = [
    (0,  ["low", "normal"]),
    (30, ["normal", "rush_hour_am", "rush_hour_pm", "holiday"]),
    (80, None),  # None = full random (all scenarios)
]

# Major green phases the agent can select (index into list = action value)
# Phase indices match network/tls.add.xml (loaded at SUMO startup)
# TL_A/B: 3-phase plan — A(0) N+S rights | B(2) E+W all | C(4) N+S all
MAJOR_GREEN_PHASES = {
    "6073919354":   [0, 2, 4],
    "6073919354_B": [0, 2, 4],
    "6073919354_C": [0, 2, 4],
}

# Yellow phase immediately following each major green (keyed by SUMO phase index)
YELLOW_AFTER = {
    "6073919354":   {0: 1, 2: 3, 4: 5},
    "6073919354_B": {0: 1, 2: 3, 4: 5},
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

        obs_size = len(TL_IDS) * (N_LANES_PER_TL + N_LANES_PER_TL + N_OUT_LANES_PER_TL + 2)  # +N_LANES for waiting time
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])

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
        self._episode_count = 0   # curriculum progression

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Retry loop: if SUMO fails to start, try up to 3 times before giving up
        for _attempt in range(3):
            try:
                return self._reset_inner()
            except Exception as e:
                print(f"[ENV {self._label}] reset() attempt {_attempt+1} failed: {e}")
                try:
                    traci.switch(self._label)
                    traci.close()
                except Exception:
                    pass
                self._sumo_running = False
                if _attempt == 2:
                    raise

        # unreachable, but satisfies type checkers
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def _reset_inner(self):
        if self._sumo_running:
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        if self.scenario_name:
            scenario = TrafficScenario(self.scenario_name)
        else:
            # Curriculum: pick from an increasingly complex pool as episodes accumulate
            pool = None
            for threshold, scenarios in CURRICULUM:
                if self._episode_count >= threshold:
                    pool = scenarios
            scenario = TrafficScenario.random() if pool is None else TrafficScenario(random.choice(pool))
        self._episode_count += 1

        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", SUMO_CFG, "--no-warnings", "--no-step-log"],
                    label=self._label)
        traci.switch(self._label)

        self._sumo_running = True
        self._sim_step = 0

        apply_tl_programs()

        _max_ms = 60.0 / 3.6
        for _eid in traci.edge.getIDList():
            try:
                if traci.edge.getMaxSpeed(_eid) > _max_ms:
                    traci.edge.setMaxSpeed(_eid, _max_ms)
            except Exception:
                pass

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
                active_id = traci.trafficlight.getProgram(tl)
                active_prog = next((p for p in programs if p.programID == active_id), programs[0])
                phase_strs = {i: p.state for i, p in enumerate(active_prog.phases)}
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
        try:
            traci.switch(self._label)
            for i, tl in enumerate(TL_IDS):
                req = int(action[i])
                state = self._phase_state[tl]

                if state == "GREEN":
                    self._green_time[tl] += 1
                    want_switch = req != self._current_action[tl] and self._green_time[tl] >= MIN_GREEN

                    # Gridlock guard: only force a rotation when an alternative is more congested
                    if not want_switch and self._green_time[tl] >= MAX_GREEN:
                        cur = self._current_action[tl]
                        cur_lanes = self._phase_lanes.get(tl, {}).get(cur, [])
                        cur_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in cur_lanes) if cur_lanes else 0
                        best_req, best_q = cur, -1
                        for alt in range(len(MAJOR_GREEN_PHASES[tl])):
                            if alt == cur:
                                continue
                            alt_lanes = self._phase_lanes.get(tl, {}).get(alt, [])
                            alt_q = (
                                sum(traci.lane.getLastStepHaltingNumber(l) for l in alt_lanes)
                                if alt_lanes else 0
                            )
                            if alt_q > best_q:
                                best_q, best_req = alt_q, alt
                        if best_req != cur and best_q > cur_q:
                            req = best_req
                            want_switch = True

                    if want_switch:
                        cur_green = MAJOR_GREEN_PHASES[tl][self._current_action[tl]]
                        yellow = YELLOW_AFTER[tl][cur_green]
                        traci.trafficlight.setPhase(tl, yellow)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        self._phase_state[tl] = "YELLOW"
                        self._state_timer[tl] = YELLOW_DUR
                        self._pending_action[tl] = req
                    else:
                        traci.trafficlight.setPhaseDuration(tl, 9999)

                elif state == "YELLOW":
                    self._state_timer[tl] -= 1
                    if self._state_timer[tl] <= 0:
                        target = MAJOR_GREEN_PHASES[tl][self._pending_action[tl]]
                        traci.trafficlight.setPhase(tl, target)
                        traci.trafficlight.setPhaseDuration(tl, 9999)
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

        except Exception as e:
            print(f"[ENV {self._label}] step() crashed: {e} — ending episode early")
            try:
                traci.switch(self._label)
                traci.close()
            except Exception:
                pass
            self._sumo_running = False
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return dummy_obs, 0.0, True, False, {}

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
            # Incoming: halting vehicles per controlled lane (log-scaled for better low-queue resolution)
            vals = []
            for lane in self._controlled_lanes[tl]:
                try:
                    h = traci.lane.getLastStepHaltingNumber(lane)
                    vals.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
                except Exception:
                    vals.append(0.0)
            while len(vals) < N_LANES_PER_TL:
                vals.append(0.0)
            parts.extend(vals[:N_LANES_PER_TL])

            # Waiting time: cumulative per-lane waiting time (log-scaled)
            wait_vals = []
            for lane in self._controlled_lanes[tl]:
                try:
                    w = traci.lane.getWaitingTime(lane)
                    wait_vals.append(min(float(np.log1p(w) / LOG_OBS_MAX_WAIT), 1.0))
                except Exception:
                    wait_vals.append(0.0)
            while len(wait_vals) < N_LANES_PER_TL:
                wait_vals.append(0.0)
            parts.extend(wait_vals[:N_LANES_PER_TL])

            # Outgoing: vehicle count per exit lane (log-scaled)
            out_vals = []
            for lane in self._outgoing_lanes[tl]:
                try:
                    v = traci.lane.getLastStepVehicleNumber(lane)
                    out_vals.append(min(float(np.log1p(v) / LOG_OBS_MAX), 1.0))
                except Exception:
                    out_vals.append(0.0)
            while len(out_vals) < N_OUT_LANES_PER_TL:
                out_vals.append(0.0)
            parts.extend(out_vals[:N_OUT_LANES_PER_TL])

            n_phases = len(MAJOR_GREEN_PHASES[tl])
            parts.append(self._current_action[tl] / max(n_phases - 1, 1))
            parts.append(0.0 if self._phase_state[tl] == "GREEN" else 1.0)
        return np.array(parts, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self):
        try:
            n_lanes = sum(len(self._controlled_lanes[tl]) for tl in TL_IDS)

            # Cumulative waiting time across all controlled lanes (primary metric)
            total_wait = sum(
                traci.lane.getWaitingTime(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # Total halting vehicles (absolute gridlock guard)
            total_q = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for tl in TL_IDS
                for lane in self._controlled_lanes[tl]
            )

            # tanh on average waiting time per lane keeps reward in (-1, 0]; 60 s is the sensitivity scale
            reward = -float(np.tanh(total_wait / max(n_lanes * 60.0, 1.0)))

            # Absolute queue penalty (prevents coasting at sustained high queue)
            reward -= 0.1 * total_q / max(n_lanes * MAX_QUEUE, 1)

            # Safety penalties
            reward -= traci.simulation.getCollidingVehiclesNumber() * 0.5
            reward -= traci.simulation.getStartingTeleportNumber() * 0.3

            return float(np.clip(reward, -2.0, 0.5))

        except Exception:
            return 0.0
