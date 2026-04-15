"""
SUMO Multi-Intersection Dashboard with PPO vs Fixed Metrics
Run with: streamlit run dashboard.py
"""

import streamlit as st
import traci
import threading
import time
import pandas as pd
import plotly.graph_objs as go
from collections import deque
import os
import numpy as np
from datetime import datetime

# ===== CONFIGURATION =====
SUMO_CFG = "triple.sumocfg"
USE_GUI = True
MAX_HISTORY = 200  # Increased for better trend analysis

# Traffic light IDs
TL_IDS = [
    "6073919354",
    "6073919354_B", 
    "6073919354_C"
]

# Lane mappings for traffic flow monitoring (from your comparison script)
INCOMING_LANES = [
    "-E5_0", "-465932558#1.34_0", "-465932558#1.34_1",
    "-465932558#1.34_2", "470773638#1_0", "465932558#0_0", "465932558#0_1",
]

OUTGOING_LANES = [
    "E5_0", "-465932558#0_0", "-465932558#0_1",
    "-470773638#1_0", "465932558#1_0",
]

# Lane mappings per intersection (for detailed view)
LANES = {
    "6073919354": INCOMING_LANES,  # Using same lanes for now
    "6073919354_B": [
        "-E5_B_0",
        "-465932558#1.34_B_0",
        "-465932558#1.34_B_1",
        "-465932558#1.34_B_2", 
        "470773638#1_B_0",
        "465932558#0_B_0",
        "465932558#0_B_1",
    ],
    "6073919354_C": [
        "-465932558#2_C_0",
        "470773638#1_C_0",
        "E0_C_0",
    ]
}

# ===== DATA STORAGE CLASS WITH METRICS =====
class SumoDataStore:
    def __init__(self):
        self.time_history = deque(maxlen=MAX_HISTORY)
        
        # Global metrics
        self.global_metrics = {
            "delay": deque(maxlen=MAX_HISTORY),
            "queue": deque(maxlen=MAX_HISTORY),
            "throughput": deque(maxlen=MAX_HISTORY),
            "stops": deque(maxlen=MAX_HISTORY),
            "pressure": deque(maxlen=MAX_HISTORY)
        }
        
        # Per-intersection metrics
        self.intersections = {}
        for tl in TL_IDS:
            self.intersections[tl] = {
                "delay": deque(maxlen=MAX_HISTORY),
                "queue": deque(maxlen=MAX_HISTORY),
                "throughput": deque(maxlen=MAX_HISTORY),
                "stops": deque(maxlen=MAX_HISTORY),
                "pressure": deque(maxlen=MAX_HISTORY),
                "phase": deque(maxlen=MAX_HISTORY),
                "occupancy": deque(maxlen=MAX_HISTORY)
            }
        
        self.running = True
        self.current_step = 0
        
    def collect_metrics_from_traci(self, tr):
        """Collect the 5 key metrics using direct TraCI calls"""
        self.current_step += 1
        self.time_history.append(self.current_step)
        
        # === 1. DELAY METRIC ===
        veh_ids = tr.vehicle.getIDList()
        delays = []
        for vid in veh_ids:
            try:
                v = tr.vehicle.getSpeed(vid)
                allowed = tr.vehicle.getAllowedSpeed(vid)
                if allowed > 0:
                    d = 1 - (v / allowed)
                    delays.append(np.clip(d, 0, 1))
            except:
                pass
        avg_delay = np.mean(delays) if delays else 0
        
        # === 2. QUEUE METRIC ===
        queues = []
        for lane in INCOMING_LANES:
            try:
                q = tr.lane.getLastStepHaltingNumber(lane)
                queues.append(q)
            except:
                queues.append(0)
        avg_queue = np.mean(queues) if queues else 0
        
        # === 3. THROUGHPUT METRIC ===
        throughput = 0
        for lane in OUTGOING_LANES:
            try:
                throughput += tr.lane.getLastStepVehicleNumber(lane)
            except:
                pass
        
        # === 4. STOPS METRIC ===
        stops = 0
        total = 0
        for lane in INCOMING_LANES:
            try:
                vids = tr.lane.getLastStepVehicleIDs(lane)
                for vid in vids:
                    total += 1
                    if tr.vehicle.getSpeed(vid) < 0.1:
                        stops += 1
            except:
                pass
        stop_ratio = (stops / total) if total > 0 else 0
        
        # === 5. PRESSURE METRIC ===
        incoming_sum = sum(tr.lane.getLastStepVehicleNumber(l) for l in INCOMING_LANES if l)
        outgoing_sum = sum(tr.lane.getLastStepVehicleNumber(l) for l in OUTGOING_LANES if l)
        pressure = incoming_sum - outgoing_sum
        
        # Update global metrics
        self.global_metrics["delay"].append(avg_delay)
        self.global_metrics["queue"].append(avg_queue)
        self.global_metrics["throughput"].append(throughput)
        self.global_metrics["stops"].append(stop_ratio)
        self.global_metrics["pressure"].append(pressure)
        
        # Update per-intersection metrics (simplified - using same values for now)
        # In production, you'd want to separate metrics per intersection
        for tl in TL_IDS:
            self.intersections[tl]["delay"].append(avg_delay)
            self.intersections[tl]["queue"].append(avg_queue)
            self.intersections[tl]["throughput"].append(throughput)
            self.intersections[tl]["stops"].append(stop_ratio)
            self.intersections[tl]["pressure"].append(pressure)
            
            # Get traffic light phase
            try:
                phase = tr.trafficlight.getPhase(tl)
                self.intersections[tl]["phase"].append(phase)
            except:
                self.intersections[tl]["phase"].append(0)
            
            # Calculate occupancy for this intersection's lanes
            tl_lanes = LANES.get(tl, [])
            occ_sum = 0
            occ_count = 0
            for lane in tl_lanes:
                try:
                    occ_sum += tr.lane.getLastStepOccupancy(lane)
                    occ_count += 1
                except:
                    pass
            avg_occ = occ_sum / occ_count if occ_count > 0 else 0
            self.intersections[tl]["occupancy"].append(avg_occ)

# ===== SUMO SIMULATION THREAD =====
def run_sumo():
    """Run SUMO simulation in background thread"""
    try:
        cmd = ["sumo-gui" if USE_GUI else "sumo", "-c", SUMO_CFG]
        traci.start(cmd)
        st.session_state['sumo_running'] = True
        
        while st.session_state.get('sumo_continue', True):
            traci.simulationStep()
            data_store.collect_metrics_from_traci(traci)
            time.sleep(0.1)
            
        traci.close()
        st.session_state['sumo_running'] = False
    except Exception as e:
        print(f"SUMO Error: {e}")
        st.session_state['sumo_running'] = False

# ===== INITIALIZE SESSION STATE =====
if 'sumo_started' not in st.session_state:
    st.session_state['sumo_started'] = False
    st.session_state['sumo_continue'] = True
    st.session_state['sumo_running'] = False

# Start SUMO only once
if not st.session_state['sumo_started']:
    data_store = SumoDataStore()
    sumo_thread = threading.Thread(target=run_sumo, daemon=True)
    sumo_thread.start()
    st.session_state['sumo_started'] = True
    st.session_state['data_store'] = data_store
else:
    data_store = st.session_state['data_store']

# ===== STREAMLIT UI =====
st.set_page_config(
    page_title="SUMO Traffic Dashboard - 5 Metrics",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-good { color: #00ff00; }
    .metric-bad { color: #ff0000; }
    .metric-neutral { color: #ffa500; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚦 SUMO Traffic Dashboard - PPO vs Fixed Metrics")
st.markdown("Monitoring **5 Key Metrics**: Delay, Queue, Throughput, Stops, Pressure")

# Show SUMO status
col_status, col_step, col_time = st.columns(3)
with col_status:
    if st.session_state.get('sumo_running', False):
        st.success("✅ SUMO Simulation Running")
    else:
        st.warning("⏳ Starting SUMO Simulation...")
with col_step:
    st.metric("Simulation Step", f"{data_store.current_step}")
with col_time:
    sim_time = data_store.current_step * 0.1
    st.metric("Sim Time", f"{sim_time:.1f} s")

st.markdown("---")

# ===== 5 KEY METRICS - MAIN DASHBOARD =====
st.subheader("📊 5 Key Performance Metrics (Current Values)")

# Create 5 metric cards in a row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_delay = data_store.global_metrics["delay"][-1] if data_store.global_metrics["delay"] else 0
    st.metric(
        "⏱️ DELAY", 
        f"{current_delay:.3f}",
        help="Normalized Delay (0=Full Speed, 1=Stopped) | LOWER IS BETTER"
    )
    st.caption("LOWER = Better")

with col2:
    current_queue = data_store.global_metrics["queue"][-1] if data_store.global_metrics["queue"] else 0
    st.metric(
        "📊 QUEUE", 
        f"{current_queue:.1f} veh",
        help="Average halting vehicles on incoming lanes | LOWER IS BETTER"
    )
    st.caption("LOWER = Better")

with col3:
    current_tp = data_store.global_metrics["throughput"][-1] if data_store.global_metrics["throughput"] else 0
    st.metric(
        "🚗 THROUGHPUT", 
        f"{current_tp:.0f} veh/step",
        help="Vehicles exiting per step | HIGHER IS BETTER"
    )
    st.caption("HIGHER = Better")

with col4:
    current_stops = data_store.global_metrics["stops"][-1] if data_store.global_metrics["stops"] else 0
    st.metric(
        "🛑 STOPS", 
        f"{current_stops:.3f}",
        help="Ratio of stopped vehicles (0=None, 1=All) | LOWER IS BETTER"
    )
    st.caption("LOWER = Better")

with col5:
    current_pressure = data_store.global_metrics["pressure"][-1] if data_store.global_metrics["pressure"] else 0
    st.metric(
        "⚡ PRESSURE", 
        f"{current_pressure:.0f}",
        help="Congestion: Incoming - Outgoing | LOWER IS BETTER"
    )
    st.caption("LOWER = Better")

st.markdown("---")

# ===== TREND CHARTS FOR ALL 5 METRICS =====
st.subheader("📈 Metric Trends Over Time")

# Create tabs for each metric type
metric_tabs = st.tabs(["⏱️ Delay", "📊 Queue", "🚗 Throughput", "🛑 Stops", "⚡ Pressure"])

# Metric configurations
metrics_config = {
    "delay": {"title": "Delay Trend", "ylabel": "Normalized Delay (0-1)", "color": "red", "lower_better": True},
    "queue": {"title": "Queue Trend", "ylabel": "Average Queue Length (vehicles)", "color": "orange", "lower_better": True},
    "throughput": {"title": "Throughput Trend", "ylabel": "Vehicles per Step", "color": "green", "lower_better": False},
    "stops": {"title": "Stop Ratio Trend", "ylabel": "Stopped Vehicles Ratio", "color": "purple", "lower_better": True},
    "pressure": {"title": "Pressure Trend", "ylabel": "Incoming - Outgoing", "color": "brown", "lower_better": True},
}

for idx, (metric_key, config) in enumerate(metrics_config.items()):
    with metric_tabs[idx]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = go.Figure()
            
            if data_store.time_history and data_store.global_metrics[metric_key]:
                values = list(data_store.global_metrics[metric_key])
                times = list(data_store.time_history)
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines',
                    name=config['title'],
                    line=dict(color=config['color'], width=2),
                    fill='tozeroy'
                ))
                
                # Add rolling average (window=10)
                if len(values) >= 10:
                    rolling_avg = pd.Series(values).rolling(window=10).mean()
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=rolling_avg,
                        mode='lines',
                        name='Rolling Avg (10 steps)',
                        line=dict(color='blue', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"{config['title']} - Current: {values[-1]:.3f}" if values else config['title'],
                    xaxis_title="Simulation Step",
                    yaxis_title=config['ylabel'],
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics box
            values = list(data_store.global_metrics[metric_key]) if data_store.global_metrics[metric_key] else []
            if values:
                st.markdown("**Statistics**")
                st.metric("Current", f"{values[-1]:.3f}")
                st.metric("Mean", f"{np.mean(values):.3f}")
                st.metric("Median", f"{np.median(values):.3f}")
                st.metric("Std Dev", f"{np.std(values):.3f}")
                st.metric("Min", f"{np.min(values):.3f}")
                st.metric("Max", f"{np.max(values):.3f}")
                
                # Performance indicator
                if config['lower_better']:
                    trend = values[-1] - values[0] if len(values) > 1 else 0
                    if trend < 0:
                        st.success(f"✅ Improving ({trend:.3f})")
                    else:
                        st.warning(f"⚠️ Worsening (+{trend:.3f})")
                else:
                    trend = values[-1] - values[0] if len(values) > 1 else 0
                    if trend > 0:
                        st.success(f"✅ Improving (+{trend:.3f})")
                    else:
                        st.warning(f"⚠️ Worsening ({trend:.3f})")

st.markdown("---")

# ===== PER-INTERSECTION DETAILS =====
st.subheader("🚥 Per-Intersection Details")

# Create tabs for each intersection
tl_tabs = st.tabs([f"🚦 {tl}" for tl in TL_IDS])

for idx, tl in enumerate(TL_IDS):
    with tl_tabs[idx]:
        # Get latest metrics for this intersection
        metrics_data = {}
        for metric in ["delay", "queue", "throughput", "stops", "pressure"]:
            metrics_data[metric] = data_store.intersections[tl][metric][-1] if data_store.intersections[tl][metric] else 0
        
        # Display metrics in 5 columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Delay", f"{metrics_data['delay']:.3f}")
        with col2:
            st.metric("Queue", f"{metrics_data['queue']:.1f}")
        with col3:
            st.metric("Throughput", f"{metrics_data['throughput']:.0f}")
        with col4:
            st.metric("Stops", f"{metrics_data['stops']:.3f}")
        with col5:
            st.metric("Pressure", f"{metrics_data['pressure']:.0f}")
        
        # Phase and occupancy
        col1, col2 = st.columns(2)
        with col1:
            current_phase = data_store.intersections[tl]["phase"][-1] if data_store.intersections[tl]["phase"] else 0
            phase_symbols = {0: "🔴 RED", 1: "🟡 YELLOW", 2: "🟢 GREEN", 3: "🟡 YELLOW"}
            st.metric("Traffic Light Phase", phase_symbols.get(current_phase, f"Phase {current_phase}"))
        
        with col2:
            current_occ = data_store.intersections[tl]["occupancy"][-1] if data_store.intersections[tl]["occupancy"] else 0
            st.metric("Lane Occupancy", f"{current_occ:.1f}%")
        
        # Mini trends for this intersection
        st.markdown("**Metric Trends**")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        metrics_to_plot = ["queue", "throughput", "pressure"]
        colors = ["orange", "green", "brown"]
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            values = list(data_store.intersections[tl][metric]) if data_store.intersections[tl][metric] else []
            times = list(data_store.time_history)
            
            if values and times:
                axes[i].plot(times, values, color=color, linewidth=2)
                axes[i].set_title(metric.capitalize())
                axes[i].set_xlabel("Step")
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ===== SIDE PANEL WITH SUMMARY STATISTICS =====
with st.sidebar:
    st.title("📈 Summary Statistics")
    
    st.markdown("### Overall Performance")
    
    # Calculate average metrics
    avg_delay = np.mean(list(data_store.global_metrics["delay"])) if data_store.global_metrics["delay"] else 0
    avg_queue = np.mean(list(data_store.global_metrics["queue"])) if data_store.global_metrics["queue"] else 0
    avg_tp = np.mean(list(data_store.global_metrics["throughput"])) if data_store.global_metrics["throughput"] else 0
    avg_stops = np.mean(list(data_store.global_metrics["stops"])) if data_store.global_metrics["stops"] else 0
    avg_pressure = np.mean(list(data_store.global_metrics["pressure"])) if data_store.global_metrics["pressure"] else 0
    
    st.metric("📊 Avg Delay", f"{avg_delay:.3f}")
    st.metric("📊 Avg Queue", f"{avg_queue:.1f}")
    st.metric("📊 Avg Throughput", f"{avg_tp:.1f}")
    st.metric("📊 Avg Stops", f"{avg_stops:.3f}")
    st.metric("📊 Avg Pressure", f"{avg_pressure:.1f}")
    
    st.markdown("---")
    
    # Performance scoring (lower is better for most metrics)
    st.markdown("### Performance Score")
    
    # Normalize metrics (0-100 scale, higher is better)
    delay_score = max(0, min(100, (1 - avg_delay) * 100))
    queue_score = max(0, min(100, 100 - (avg_queue * 10)))
    tp_score = max(0, min(100, (avg_tp / 5) * 100))
    stops_score = max(0, min(100, (1 - avg_stops) * 100))
    pressure_score = max(0, min(100, 100 - (abs(avg_pressure) * 5)))
    
    overall_score = (delay_score + queue_score + tp_score + stops_score + pressure_score) / 5
    
    st.metric("Overall Score", f"{overall_score:.1f}/100")
    
    # Progress bar
    st.progress(overall_score / 100)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **5 Key Metrics:**
    - **Delay**: Lower is better (0=full speed, 1=stopped)
    - **Queue**: Lower is better (halting vehicles)
    - **Throughput**: Higher is better (vehicles/step)
    - **Stops**: Lower is better (stopped ratio)
    - **Pressure**: Lower is better (incoming-outgoing)
    """)

# Auto-refresh
time.sleep(1)
st.rerun()