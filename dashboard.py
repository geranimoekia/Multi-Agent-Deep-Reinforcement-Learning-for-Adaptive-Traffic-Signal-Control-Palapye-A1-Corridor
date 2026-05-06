import streamlit as st
import traci
import threading
import time
import traceback
import glob
import os
import numpy as np
import plotly.graph_objs as go
from collections import deque, defaultdict
import traffic_injector
import green_wave
from traffic_scenario import TrafficScenario, PROFILE_NAMES, PROFILES
from tl_programs import apply_tl_programs

# ================= CONFIG =================
SUMO_CFG = "network/triple.sumocfg"
USE_GUI = True
MAX_HISTORY = 300

TL_IDS = ["6073919354", "6073919354_B", "6073919354_C"]

# ================= THEME =================
APPLE_FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
APPLE_MONO = "'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace"

COLORS = {
    'bg_primary': '#f2f2f7',
    'bg_card': '#ffffff',
    'text_primary': '#1d1d1f',
    'text_secondary': '#6e6e73',
    'text_tertiary': '#aeaeb2',
    'accent_green': '#34c759',
    'accent_orange': '#ff9500',
    'accent_red': '#ff3b30',
    'accent_blue': '#0071e3',
    'border_light': '#e5e5ea',
}

# Plot theme
PLOT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#fafafa',
    font=dict(color=COLORS['text_secondary'], family=APPLE_FONT, size=11),
)
AXIS_STYLE = dict(
    gridcolor='#f0f0f0',
    linecolor=COLORS['border_light'],
    zerolinecolor=COLORS['border_light'],
    tickcolor=COLORS['text_tertiary'],
)

# ================= STATE =================
class Store:
    def __init__(self):
        self.step = 0
        self.time = deque(maxlen=MAX_HISTORY)
        self.queue = deque(maxlen=MAX_HISTORY)
        self.throughput = deque(maxlen=MAX_HISTORY)
        self.running = True
        self.lock = threading.Lock()
        self.active_vehicles = 0   # vehicles currently in network (goes up AND down)
        # PPO model control
        self.model_enabled = False
        self.model = None
        self.model_name = ""

store = Store()


# ================= MODEL CONTROL HELPERS =================
_DELTA_T         = 3
_YELLOW_DUR      = 1    # _tick_phases calls while yellow ≈ 3 sim-steps
_MIN_GREEN       = 5    # must match MIN_GREEN in sumo_env.py
_MAX_GREEN       = 25   # force rotation sooner — helps middle intersection (B) from getting stuck
_N_LANES         = 8
_N_OUT_LANES     = 4
_OBS_MAX_Q       = 15.0
_LOG_OBS_MAX     = float(np.log1p(_OBS_MAX_Q))   # must match sumo_env.py
_OBS_MAX_WAIT    = 300.0
_LOG_OBS_MAX_WAIT = float(np.log1p(_OBS_MAX_WAIT))  # must match sumo_env.py

_MAJOR_GREEN = {
    "6073919354":   [0, 4],
    "6073919354_B": [0, 4],
    "6073919354_C": [0, 2, 4],
}
_YELLOW_AFTER = {
    "6073919354":   {0: 1, 4: 5},
    "6073919354_B": {0: 1, 4: 5},
    "6073919354_C": {0: 1, 2: 3, 4: 5},
}


def _build_obs(controlled_lanes, outgoing_lanes, phase_state, current_action):
    parts = []
    for tl in TL_IDS:
        # Incoming: halting vehicles per controlled lane (how blocked is the approach)
        vals = []
        for lane in controlled_lanes.get(tl, []):
            try:
                h = traci.lane.getLastStepHaltingNumber(lane)
                vals.append(min(float(np.log1p(h) / _LOG_OBS_MAX), 1.0))
            except Exception:
                vals.append(0.0)
        while len(vals) < _N_LANES:
            vals.append(0.0)
        parts.extend(vals[:_N_LANES])

        # Waiting time: cumulative per-lane waiting time (log-scaled, matches sumo_env.py)
        wait_vals = []
        for lane in controlled_lanes.get(tl, []):
            try:
                w = traci.lane.getWaitingTime(lane)
                wait_vals.append(min(float(np.log1p(w) / _LOG_OBS_MAX_WAIT), 1.0))
            except Exception:
                wait_vals.append(0.0)
        while len(wait_vals) < _N_LANES:
            wait_vals.append(0.0)
        parts.extend(wait_vals[:_N_LANES])

        # Outgoing: vehicle count per exit lane (is there space to receive traffic)
        out_vals = []
        for lane in outgoing_lanes.get(tl, []):
            try:
                v = traci.lane.getLastStepVehicleNumber(lane)
                out_vals.append(min(float(np.log1p(v) / _LOG_OBS_MAX), 1.0))
            except Exception:
                out_vals.append(0.0)
        while len(out_vals) < _N_OUT_LANES:
            out_vals.append(0.0)
        parts.extend(out_vals[:_N_OUT_LANES])

        n_phases = len(_MAJOR_GREEN[tl])
        parts.append(current_action.get(tl, 0) / max(n_phases - 1, 1))
        parts.append(0.0 if phase_state.get(tl) == "GREEN" else 1.0)
    return np.array(parts, dtype=np.float32)


def _tick_phases(action, phase_state, state_timer, cur_act, pend_act, green_time, phase_lanes=None):
    for i, tl in enumerate(TL_IDS):
        req = int(action[i]) if action is not None else cur_act.get(tl, 0)
        state = phase_state.get(tl, "GREEN")

        if state == "GREEN":
            green_time[tl] = green_time.get(tl, 0) + 1
            want_switch = req != cur_act.get(tl, 0) and green_time[tl] >= _MIN_GREEN

            # Gridlock guard: only force a rotation when an alternative is more congested
            if not want_switch and green_time[tl] >= _MAX_GREEN:
                cur = cur_act.get(tl, 0)
                cur_lanes = (phase_lanes or {}).get(tl, {}).get(cur, [])
                cur_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in cur_lanes) if cur_lanes else 0
                best_req, best_q = cur, -1
                for alt in range(len(_MAJOR_GREEN[tl])):
                    if alt == cur:
                        continue
                    alt_lanes = (phase_lanes or {}).get(tl, {}).get(alt, [])
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
                cur_green = _MAJOR_GREEN[tl][cur_act.get(tl, 0)]
                yellow = _YELLOW_AFTER[tl][cur_green]
                traci.trafficlight.setPhase(tl, yellow)
                traci.trafficlight.setPhaseDuration(tl, 9999)
                phase_state[tl] = "YELLOW"
                state_timer[tl] = _YELLOW_DUR
                pend_act[tl] = req
            else:
                traci.trafficlight.setPhaseDuration(tl, 9999)

        elif state == "YELLOW":
            state_timer[tl] = state_timer.get(tl, 1) - 1
            if state_timer[tl] <= 0:
                target = _MAJOR_GREEN[tl][pend_act.get(tl, 0)]
                traci.trafficlight.setPhase(tl, target)
                traci.trafficlight.setPhaseDuration(tl, 9999)
                phase_state[tl] = "GREEN"
                cur_act[tl] = pend_act.get(tl, 0)
                green_time[tl] = 0


# ================= SIMULATION LOOP =================
def run_sumo():
    try:
        cmd = ["sumo-gui" if USE_GUI else "sumo", "-c", SUMO_CFG]
        traci.start(cmd)
        print("SUMO Started Successfully")

        apply_tl_programs()

        # Cap every edge to 60 km/h (16.67 m/s)
        _MAX_MS = 60.0 / 3.6
        for _eid in traci.edge.getIDList():
            try:
                if traci.edge.getMaxSpeed(_eid) > _MAX_MS:
                    traci.edge.setMaxSpeed(_eid, _MAX_MS)
            except Exception:
                pass

        initial = TrafficScenario(st.session_state.get("scenario_name", "normal"))
        traffic_injector.init(initial)

        gw = green_wave.create(TL_IDS)
        gw.bootstrap(free_flow_kmh=st.session_state.get("gw_speed", 50.0))

        # Cache lanes once for model obs building
        controlled_lanes = {}
        outgoing_lanes = {}
        phase_lanes = {}
        for tl in TL_IDS:
            controlled_lanes[tl] = list(
                dict.fromkeys(traci.trafficlight.getControlledLanes(tl))
            )
            links_tl = traci.trafficlight.getControlledLinks(tl)
            out = []
            for link_group in links_tl:
                for (_, to_lane, _) in link_group:
                    if to_lane and to_lane not in out:
                        out.append(to_lane)
            outgoing_lanes[tl] = out
            try:
                programs = traci.trafficlight.getAllProgramLogics(tl)
                active_id = traci.trafficlight.getProgram(tl)
                active_prog = next((p for p in programs if p.programID == active_id), programs[0])
                phase_strs = {i: p.state for i, p in enumerate(active_prog.phases)}
            except Exception:
                phase_strs = {}
            phase_lanes[tl] = {}
            for act_idx, ph_idx in enumerate(_MAJOR_GREEN[tl]):
                state_str = phase_strs.get(ph_idx, "")
                served = []
                for li, ch in enumerate(state_str):
                    if ch in ('G', 'g') and li < len(links_tl):
                        for (fl, _, _) in links_tl[li]:
                            if fl and fl not in served:
                                served.append(fl)
                phase_lanes[tl][act_idx] = served

        # Phase state machine for model control
        phase_state  = {tl: "GREEN" for tl in TL_IDS}
        state_timer  = {tl: 0       for tl in TL_IDS}
        cur_act      = {tl: 0       for tl in TL_IDS}
        pend_act     = {tl: 0       for tl in TL_IDS}
        green_time   = {tl: 0       for tl in TL_IDS}
        window_steps = 0
        model_active = False  # last-known model state for transition detection

        while store.running:
            traci.simulationStep()
            step = traci.simulation.getTime()

            with store.lock:
                store.step = int(step)

            traffic_injector.inject(int(step))

            # Only let green wave run when the model is NOT in control
            if not model_active:
                gw.update(int(step))

            # Model inference every DELTA_T sim steps
            window_steps += 1
            if window_steps >= _DELTA_T:
                window_steps = 0
                with store.lock:
                    mdl     = store.model
                    enabled = store.model_enabled

                # Snap all TLs to major-green-0 the moment the model is turned on so
                # cur_act matches reality (SUMO may be mid-cycle at any phase)
                if enabled and not model_active:
                    for tl in TL_IDS:
                        traci.trafficlight.setPhase(tl, _MAJOR_GREEN[tl][0])
                        traci.trafficlight.setPhaseDuration(tl, 9999)
                        phase_state[tl] = "GREEN"
                        state_timer[tl] = 0
                        cur_act[tl]     = 0
                        pend_act[tl]    = 0
                        green_time[tl]  = 0

                model_active = enabled

                if enabled and mdl is not None:
                    obs    = _build_obs(controlled_lanes, outgoing_lanes, phase_state, cur_act)
                    action, _ = mdl.predict(obs, deterministic=True)
                    _tick_phases(action, phase_state, state_timer, cur_act, pend_act, green_time, phase_lanes)
                else:
                    # drain any in-progress yellow if model was just turned off
                    if any(phase_state[tl] == "YELLOW" for tl in TL_IDS):
                        _tick_phases(None, phase_state, state_timer, cur_act, pend_act, green_time, phase_lanes)

            q = 0
            t = 0
            try:
                for lane in traci.lane.getIDList():
                    try:
                        q += traci.lane.getLastStepHaltingNumber(lane)
                        t += traci.lane.getLastStepVehicleNumber(lane)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Metrics error: {e}")

            active = traci.vehicle.getIDCount()
            with store.lock:
                store.time.append(step)
                store.queue.append(q)
                store.throughput.append(t)
                store.active_vehicles = active

            time.sleep(0.05)

        traci.close()
        print("SUMO Closed")

    except Exception as e:
        print(f"SUMO Error: {e}")
        traceback.print_exc()
        try:
            traci.close()
        except Exception:
            pass
        store.running = False


# ================= START THREAD =================
if "started" not in st.session_state:
    st.session_state.started = True
    st.session_state.thread = threading.Thread(target=run_sumo, daemon=True)
    st.session_state.thread.start()

import atexit
atexit.register(lambda: setattr(store, "running", False))


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SUMO Traffic Control System",
    page_icon="🚦",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown(f"""
<style>
* {{
    box-sizing: border-box;
}}

.stApp {{
    background-color: {COLORS['bg_primary']} !important;
    color: {COLORS['text_primary']};
    font-family: {APPLE_FONT} !important;
}}

/* Main content area */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    background-color: {COLORS['bg_primary']};
}}

h1, h2, h3, h4, h5, h6 {{
    color: {COLORS['text_primary']} !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
    font-family: {APPLE_FONT} !important;
}}

/* Title styling */
h1 {{
    font-size: 32px !important;
    margin-bottom: 0.5rem !important;
}}

/* Subtitle */
.main p {{
    color: {COLORS['text_secondary']} !important;
    font-size: 14px;
}}

/* Metric containers - MORE SPECIFIC SELECTORS */
div[data-testid="stMetricValue"] > div {{
    color: {COLORS['text_primary']} !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    font-family: {APPLE_MONO} !important;
    line-height: 1.2 !important;
}}

div[data-testid="stMetric"] {{
    background-color: {COLORS['bg_card']} !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04) !important;
    border: 1px solid {COLORS['border_light']} !important;
}}

div[data-testid="stMetric"] label {{
    color: {COLORS['text_tertiary']} !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    font-family: {APPLE_FONT} !important;
}}

div[data-testid="stMetricDelta"] {{
    font-size: 13px !important;
    font-weight: 500 !important;
}}

div[data-testid="stMetricDelta"] svg {{
    display: none !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {COLORS['accent_blue']} !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: 0 2px 8px rgba(0,113,227,0.2) !important;
    transition: all 0.2s !important;
    font-family: {APPLE_FONT} !important;
}}

.stButton > button:hover {{
    background-color: #0051a8 !important;
    box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important;
}}

.stButton > button[kind="secondary"] {{
    background-color: {COLORS['bg_card']} !important;
    color: {COLORS['text_primary']} !important;
    border: 1px solid {COLORS['border_light']} !important;
}}

.stButton > button[kind="secondary"]:hover {{
    background-color: #f9f9f9 !important;
    border-color: {COLORS['text_tertiary']} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {COLORS['bg_card']} !important;
    border-right: 1px solid {COLORS['border_light']} !important;
}}

section[data-testid="stSidebar"] > div {{
    background-color: {COLORS['bg_card']} !important;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem !important;
}}

section[data-testid="stSidebar"] h3 {{
    font-size: 18px !important;
    margin-bottom: 1rem !important;
}}

section[data-testid="stSidebar"] hr {{
    margin: 1.5rem 0 !important;
    border-color: {COLORS['border_light']} !important;
}}

/* Sidebar — force all labels and text to be visible */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stToggle label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {{
    color: {COLORS['text_primary']} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: {APPLE_FONT} !important;
    opacity: 1 !important;
}}

/* Sidebar selectbox — dropdown box itself */
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background-color: {COLORS['bg_primary']} !important;
    border: 1px solid {COLORS['border_light']} !important;
    border-radius: 8px !important;
    color: {COLORS['text_primary']} !important;
}}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {{
    color: {COLORS['text_primary']} !important;
}}

/* Sidebar toggle track and thumb */
section[data-testid="stSidebar"] .stToggle [data-baseweb="checkbox"] {{
    opacity: 1 !important;
}}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stCaptionContainer {{
    color: {COLORS['text_secondary']} !important;
    opacity: 1 !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 2px solid {COLORS['border_light']} !important;
    padding: 0 !important;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    border: none !important;
    color: {COLORS['text_tertiary']} !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    border-bottom: 3px solid transparent !important;
    font-family: {APPLE_FONT} !important;
    height: auto !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: {COLORS['text_primary']} !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: transparent !important;
    border-bottom: 3px solid {COLORS['accent_blue']} !important;
    color: {COLORS['accent_blue']} !important;
    font-weight: 600 !important;
}}

.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 2rem !important;
}}

/* Selectbox */
.stSelectbox label {{
    color: {COLORS['text_secondary']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: {APPLE_FONT} !important;
}}

.stSelectbox > div > div {{
    border-color: {COLORS['border_light']} !important;
    border-radius: 8px !important;
}}

/* Slider */
.stSlider label {{
    color: {COLORS['text_secondary']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: {APPLE_FONT} !important;
}}

.stSlider [data-baseweb="slider"] {{
    margin-top: 1rem !important;
}}

/* Checkbox */
.stCheckbox label {{
    color: {COLORS['text_secondary']} !important;
    font-size: 14px !important;
    font-family: {APPLE_FONT} !important;
}}

/* Toggle */
.stToggle label {{
    color: {COLORS['text_secondary']} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: {APPLE_FONT} !important;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid {COLORS['border_light']} !important;
}}

.dataframe {{
    font-family: {APPLE_FONT} !important;
    font-size: 13px !important;
}}

.dataframe thead tr th {{
    background-color: #fafafa !important;
    color: {COLORS['text_secondary']} !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 12px !important;
    border-bottom: 2px solid {COLORS['border_light']} !important;
}}

.dataframe tbody tr td {{
    padding: 10px 12px !important;
    border-bottom: 1px solid #f5f5f5 !important;
}}

.dataframe tbody tr:hover {{
    background-color: #fafafa !important;
}}

/* Info/Warning boxes */
.stAlert {{
    border-radius: 10px !important;
    border: 1px solid {COLORS['border_light']} !important;
    font-family: {APPLE_FONT} !important;
}}

/* Section labels */
.section-label {{
    color: {COLORS['text_tertiary']} !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
    margin-top: 24px !important;
    display: block !important;
    font-family: {APPLE_FONT} !important;
}}

/* Status pill */
.status-pill {{
    padding: 6px 14px !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    display: inline-block !important;
    font-family: {APPLE_FONT} !important;
}}

.status-online {{
    background: #f0faf4 !important;
    color: #1c7a3e !important;
    border: 1px solid #c3e9d0 !important;
}}

.status-offline {{
    background: #fdf2f2 !important;
    color: #c0392b !important;
    border: 1px solid #f5c6c6 !important;
}}

/* Caption text */
.stCaptionContainer {{
    color: {COLORS['text_tertiary']} !important;
    font-size: 12px !important;
    font-family: {APPLE_FONT} !important;
}}

</style>
""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================
def section_label(text):
    """Create a properly styled section label"""
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)


def status_pill(text, online=True):
    css_class = "status-online" if online else "status-offline"
    dot = "●" if online else "○"
    return f'<span class="status-pill {css_class}">{dot} {text}</span>'


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### SUMO Control")
    st.markdown("---")

    # Auto-refresh controls
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        refresh_rate = st.slider("Refresh interval (seconds)", 1, 5, 2)

    st.markdown("---")
    
    # Quick stats
    section_label("Quick Stats")
    
    with store.lock:
        current_step = store.step
        active_vehicles = store.active_vehicles

    total_injected = traffic_injector.vehicle_count()
    log_snapshot = list(traffic_injector.injection_log)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Step", f"{current_step}")
    with col2:
        st.metric("Active", f"{active_vehicles}")

    st.metric("Total Injected", f"{total_injected}")

    st.markdown("---")
    
    # Active scenario indicator
    section_label("Active Scenario")
    current_scenario = st.session_state.get("scenario_name", "normal")
    st.markdown(f"**{current_scenario.replace('_', ' ').title()}**")
    st.caption("Change in Control tab")

    st.markdown("---")
    
    # Green wave controls (in sidebar so always accessible)
    section_label("Green Wave")
    gw = green_wave.get()
    if gw is not None:
        gw_on = st.toggle(
            "Enable Green Wave",
            value=st.session_state.get("gw_enabled", True),
            key="gw_sidebar_toggle"
        )
        if gw_on != st.session_state.get("gw_enabled", True):
            st.session_state["gw_enabled"] = gw_on
            gw.enabled = gw_on
            if gw_on:
                gw._apply_all()
        if gw_on:
            spd = st.slider(
                "Speed (km/h)", 20, 80,
                st.session_state.get("gw_speed", 50),
                key="gw_sidebar_speed"
            )
            if spd != st.session_state.get("gw_speed", 50):
                st.session_state["gw_speed"] = spd
                gw.set_speed(spd)
            st.markdown(status_pill("Active", online=True), unsafe_allow_html=True)
        else:
            st.markdown(status_pill("Disabled", online=False), unsafe_allow_html=True)
    else:
        st.caption("Waiting for simulation...")

    st.markdown("---")

    # PPO model control
    section_label("PPO Model")

    model_files = sorted(glob.glob("ppo_models/*.zip"))
    model_names = [os.path.basename(f) for f in model_files]

    if model_names:
        default_idx = model_names.index("best_model.zip") if "best_model.zip" in model_names else 0
        selected_name = st.selectbox("Model file", model_names, index=default_idx, key="model_select")
        selected_path = f"ppo_models/{selected_name}"

        with store.lock:
            currently_enabled = store.model_enabled
            currently_loaded  = store.model_name

        model_on = st.toggle("Enable Model Control", value=currently_enabled, key="model_toggle")

        if model_on and (not currently_enabled or currently_loaded != selected_name):
            # Load (or reload) model
            from stable_baselines3 import PPO
            with st.spinner(f"Loading {selected_name}..."):
                mdl = PPO.load(selected_path)
            with store.lock:
                store.model        = mdl
                store.model_name   = selected_name
                store.model_enabled = True
            st.success(f"Model active: {selected_name}")

        elif not model_on and currently_enabled:
            with store.lock:
                store.model_enabled = False
                store.model        = None
                store.model_name   = ""
            st.info("Model disabled — SUMO running its own TL programs.")

        with store.lock:
            display_enabled = store.model_enabled
            display_name    = store.model_name

        if display_enabled:
            st.markdown(
                status_pill(f"AI: {display_name}", online=True),
                unsafe_allow_html=True
            )
        else:
            st.markdown(status_pill("AI: Off", online=False), unsafe_allow_html=True)
    else:
        st.caption("No models found in ppo_models/")

    st.markdown("---")

    # Stop button
    if st.button("Stop Simulation", type="primary", use_container_width=True):
        store.running = False
        st.warning("Stopping simulation...")
        time.sleep(1)
        st.rerun()


# ================= MAIN CONTENT =================
st.title("SUMO Traffic Control System")
st.markdown("Real-time vehicle injection with adaptive demand management")

# Calculate current metrics
with store.lock:
    current_queue = store.queue[-1] if store.queue else 0
    current_throughput = store.throughput[-1] if store.throughput else 0
    avg_queue = sum(store.queue) / len(store.queue) if store.queue else 0
    total_throughput = sum(store.throughput) if store.throughput else 0
    time_data = list(store.time)
    queue_data = list(store.queue)
    throughput_data = list(store.throughput)


# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["Live Monitor", "Traffic Analysis", "Control & Configuration"])


# ═══════════════════════════════════════════════════════════════
# TAB 1: LIVE MONITOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Real-time Traffic Metrics")
    
    # Hero metrics strip
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_val = current_queue - avg_queue
        st.metric(
            "Current Queue",
            f"{current_queue}",
            delta=f"{delta_val:.1f} vs avg",
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Current Throughput", f"{current_throughput}")
    
    with col3:
        st.metric("Total Throughput", f"{total_throughput}")
    
    with col4:
        st.metric("Average Queue", f"{avg_queue:.1f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main traffic chart
    st.markdown(section_label("Live Traffic Flow"), unsafe_allow_html=True)
    
    fig_live = go.Figure()
    
    fig_live.add_trace(go.Scatter(
        x=time_data,
        y=queue_data,
        name="Queue Length",
        line=dict(width=2.5, color=COLORS['accent_red']),
        fill='tozeroy',
        fillcolor=f"rgba(255, 59, 48, 0.1)",
        mode='lines'
    ))
    
    fig_live.add_trace(go.Scatter(
        x=time_data,
        y=throughput_data,
        name="Throughput",
        line=dict(width=2.5, color=COLORS['accent_green']),
        yaxis="y2",
        mode='lines'
    ))
    
    fig_live.update_layout(
        **PLOT_BASE,
        height=400,
        xaxis_title="Time Step",
        yaxis_title="Queue Length (vehicles)",
        yaxis=dict(**AXIS_STYLE),
        yaxis2=dict(
            title="Throughput (vehicles/step)",
            overlaying='y',
            side='right',
            **AXIS_STYLE
        ),
        xaxis=dict(**AXIS_STYLE),
        hovermode="x unified",
        legend=dict(
            orientation='h',
            y=1.1,
            x=0.5,
            xanchor='center',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS['border_light'],
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_live, use_container_width=True, key="live_traffic_chart")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent injections
    st.markdown(section_label("Recent Vehicle Injections"), unsafe_allow_html=True)
    
    recent = log_snapshot[-15:][::-1]
    
    if recent:
        st.dataframe(
            [
                {
                    "Step": e["step"],
                    "Vehicle ID": e["veh_id"],
                    "Type": e["type"],
                    "Route": f"{e['origin']} → {e['destination']}",
                    "Edges": e["route_len"],
                }
                for e in recent
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No vehicles injected yet. Waiting for simulation to start...")


# ═══════════════════════════════════════════════════════════════
# TAB 2: TRAFFIC ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Traffic Patterns & Analysis")
    
    # Build heatmap data
    heatmap_data = defaultdict(int)
    for entry in log_snapshot:
        heatmap_data[(entry["origin"], entry["destination"])] += 1
    
    if heatmap_data:
        # Origin-Destination heatmap
        st.markdown(section_label("Origin-Destination Flow Distribution"), unsafe_allow_html=True)
        
        sorted_items = sorted(heatmap_data.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = [f"{o} → {d}" for (o, d), _ in sorted_items]
        values = [v for _, v in sorted_items]
        
        fig_heatmap = go.Figure(data=go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=values,
                colorscale=[
                    [0, COLORS['accent_green']],
                    [0.5, COLORS['accent_orange']],
                    [1, COLORS['accent_red']]
                ],
                colorbar=dict(title="Count", thickness=15)
            ),
            text=values,
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text_secondary'])
        ))
        
        fig_heatmap.update_layout(
            **PLOT_BASE,
            height=500,
            xaxis_title="Origin → Destination",
            yaxis_title="Injection Count",
            xaxis=dict(**AXIS_STYLE, tickangle=-45),
            yaxis=dict(**AXIS_STYLE),
            margin=dict(l=10, r=10, t=20, b=100)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, key="od_heatmap")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Routes", f"{len(heatmap_data)}")
        with col2:
            st.metric("Total Injections", f"{sum(heatmap_data.values())}")
        with col3:
            most_used = max(heatmap_data.items(), key=lambda x: x[1])
            st.metric("Most Used Route", f"{most_used[1]} vehicles")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Queue vs Throughput correlation
        st.markdown(section_label("Queue vs Throughput Correlation"), unsafe_allow_html=True)
        
        fig_correlation = go.Figure()
        
        fig_correlation.add_trace(go.Scatter(
            x=queue_data,
            y=throughput_data,
            mode='markers',
            marker=dict(
                size=6,
                color=time_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Step", thickness=15),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=[f"Step: {int(t)}" for t in time_data],
            hovertemplate="<b>Queue:</b> %{x}<br><b>Throughput:</b> %{y}<br>%{text}<extra></extra>"
        ))
        
        fig_correlation.update_layout(
            **PLOT_BASE,
            height=400,
            xaxis_title="Queue Length (vehicles)",
            yaxis_title="Throughput (vehicles/step)",
            xaxis=dict(**AXIS_STYLE),
            yaxis=dict(**AXIS_STYLE),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig_correlation, use_container_width=True, key="correlation_chart")
        
    else:
        st.info("Waiting for traffic data to build analysis...")


# ═══════════════════════════════════════════════════════════════
# TAB 3: CONTROL & CONFIGURATION
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Scenario & Control Configuration")
    
    # Scenario selection section
    st.markdown(section_label("Traffic Scenario Selection"), unsafe_allow_html=True)
    
    SCENARIO_DESCRIPTIONS = {
        "low": {
            "name": "Low Traffic",
            "desc": "Off-peak dispersed traffic flow",
            "color": COLORS['accent_green']
        },
        "normal": {
            "name": "Normal Traffic",
            "desc": "Weekday mixed traffic conditions",
            "color": COLORS['accent_blue']
        },
        "rush_hour_am": {
            "name": "Morning Rush",
            "desc": "Heavy inbound morning traffic",
            "color": COLORS['accent_orange']
        },
        "rush_hour_pm": {
            "name": "Evening Rush",
            "desc": "Heavy outbound evening traffic",
            "color": COLORS['accent_orange']
        },
        "holiday": {
            "name": "Holiday Traffic",
            "desc": "Transit traffic through town",
            "color": COLORS['accent_blue']
        },
        "incident": {
            "name": "Incident",
            "desc": "Two entry points closed",
            "color": COLORS['accent_red']
        },
    }
    
    # Create scenario cards
    cols = st.columns(3)
    
    for idx, scenario_name in enumerate(PROFILE_NAMES):
        with cols[idx % 3]:
            scenario_info = SCENARIO_DESCRIPTIONS.get(scenario_name, {
                "name": scenario_name.replace("_", " ").title(),
                "desc": "Custom scenario",
                "color": COLORS['text_secondary']
            })
            
            is_active = st.session_state.get("scenario_name", "normal") == scenario_name
            
            button_label = f"{'● ' if is_active else ''}{scenario_info['name']}"
            
            if st.button(
                button_label,
                key=f"scenario_{scenario_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    traffic_injector.set_scenario(TrafficScenario(scenario_name))
                    st.session_state["scenario_name"] = scenario_name
                    st.rerun()
            
            st.caption(scenario_info['desc'])
    
    # Show active scenario details
    if st.session_state.get("scenario_name"):
        st.markdown("<br>", unsafe_allow_html=True)
        active_name = st.session_state["scenario_name"]
        prof = PROFILES[active_name]
        
        st.markdown(f"**Active Scenario:** {active_name.replace('_', ' ').title()}")
        st.markdown(f"**Description:** {prof['description']}")
        st.markdown(f"**Direction Mode:** {prof['direction_mode']}")
        
        if prof["blocked_origins"]:
            st.warning(f"⚠ Blocked entries: {', '.join(prof['blocked_origins'])}")
    
    # Random scenario button
    if st.button("Random Scenario", use_container_width=True):
        rand = TrafficScenario.random()
        traffic_injector.set_scenario(rand)
        st.session_state["scenario_name"] = rand.name
        st.rerun()
    
    st.markdown("---")
    
    # Green wave — detail view (toggle lives in sidebar)
    section_label("Green Wave Coordination")

    gw = green_wave.get()
    if gw is not None and gw.enabled:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cycle Length", f"{gw.cycle_length:.1f} s")
        with col2:
            st.metric("Coordinated Signals", f"{len(TL_IDS)}")

        section_label("Signal Offsets")
        offset_data = [
            {
                "Traffic Light": tl_id,
                "Offset (seconds)": f"+{offset:.1f}",
                "Phase": f"{(offset / gw.cycle_length * 100):.1f}%" if gw.cycle_length else "—",
            }
            for tl_id, offset in gw.offsets.items()
        ]
        if offset_data:
            st.dataframe(offset_data, use_container_width=True, hide_index=True)
    else:
        st.caption("Green wave disabled — toggle it in the sidebar.")


# ================= AUTO-REFRESH =================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()