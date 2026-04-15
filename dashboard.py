"""
SUMO Multi-Intersection Dashboard for Streamlit
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
import atexit

# ===== CONFIGURATION =====
SUMO_CFG = "triple.sumocfg"
USE_GUI = True
MAX_HISTORY = 100

# Traffic light IDs
TL_IDS = [
    "6073919354",
    "6073919354_B", 
    "6073919354_C"
]

# Lane mappings
LANES = {
    "6073919354": [
        "-E5_0",
        "-465932558#1.34_0",
        "-465932558#1.34_1", 
        "-465932558#1.34_2",
        "470773638#1_0",
        "465932558#0_0",
        "465932558#0_1",
    ],
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

# ===== DATA STORAGE CLASS =====
class SumoDataStore:
    def __init__(self):
        self.time_history = deque(maxlen=MAX_HISTORY)
        self.global_queue = deque(maxlen=MAX_HISTORY)
        self.global_throughput = deque(maxlen=MAX_HISTORY)
        
        self.intersections = {
            tl: {
                "queue": deque(maxlen=MAX_HISTORY),
                "throughput": deque(maxlen=MAX_HISTORY),
                "phase": deque(maxlen=MAX_HISTORY),
                "occupancy": deque(maxlen=MAX_HISTORY),
                "waiting_time": deque(maxlen=MAX_HISTORY),
            }
            for tl in TL_IDS
        }
        
        self.running = True
        self.current_step = 0
        self.lane_queues = {}
        self.flow_periods = {}
        self.demand_level = 1.0
        
    def update(self):
        self.current_step += 1
        self.time_history.append(self.current_step)
        
        total_queue = 0
        total_throughput = 0
        
        for tl in TL_IDS:
            lanes = LANES.get(tl, [])
            
            queues = []
            throughputs = []
            occupancies = []
            waiting_times = []
            
            for lane in lanes:
                if lane:
                    try:
                        q = traci.lane.getLastStepHaltingNumber(lane)
                        t = traci.lane.getLastStepVehicleNumber(lane)
                        occ = traci.lane.getLastStepOccupancy(lane)
                        wt = traci.lane.getWaitingTime(lane)
                        
                        queues.append(q)
                        throughputs.append(t)
                        occupancies.append(occ)
                        waiting_times.append(wt)
                        
                        self.lane_queues[f"{tl}_{lane}"] = q
                    except Exception:
                        pass
            
            q_sum = sum(queues) if queues else 0
            t_sum = sum(throughputs) if throughputs else 0
            occ_avg = sum(occupancies) / len(occupancies) if occupancies else 0
            wt_sum = sum(waiting_times) if waiting_times else 0
            
            try:
                phase = traci.trafficlight.getPhase(tl)
            except:
                phase = 0
            
            self.intersections[tl]["queue"].append(q_sum)
            self.intersections[tl]["throughput"].append(t_sum)
            self.intersections[tl]["phase"].append(phase)
            self.intersections[tl]["occupancy"].append(occ_avg)
            self.intersections[tl]["waiting_time"].append(wt_sum)
            
            total_queue += q_sum
            total_throughput += t_sum
        
        self.global_queue.append(total_queue)
        self.global_throughput.append(total_throughput)
    
    def update_demand(self, demand_multiplier):
        """Update traffic demand by adjusting flow periods"""
        try:
            flows = traci.route.getFlowIDList()
            
            for flow_id in flows:
                # Store original period on first call
                if flow_id not in self.flow_periods:
                    try:
                        self.flow_periods[flow_id] = traci.flow.getFlowPeriod(flow_id)
                    except:
                        continue
                
                # Calculate new period based on demand
                original_period = self.flow_periods[flow_id]
                new_period = max(1, int(original_period / demand_multiplier))
                
                try:
                    traci.flow.setFlowPeriod(flow_id, new_period)
                except:
                    pass
                    
            self.demand_level = demand_multiplier
        except Exception as e:
            print(f"Error updating demand: {e}")

# ===== SUMO SIMULATION THREAD =====
def run_sumo():
    """Run SUMO simulation in background thread"""
    try:
        cmd = ["sumo-gui" if USE_GUI else "sumo", "-c", SUMO_CFG]
        traci.start(cmd)
        st.session_state['sumo_running'] = True
        
        while st.session_state.get('sumo_continue', True):
            traci.simulationStep()
            data_store.update()
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
    st.session_state['demand_level'] = 1.0

# Start SUMO only once
if not st.session_state['sumo_started']:
    data_store = SumoDataStore()
    sumo_thread = threading.Thread(target=run_sumo, daemon=True)
    sumo_thread.start()
    st.session_state['sumo_started'] = True
    st.session_state['data_store'] = data_store
else:
    data_store = st.session_state['data_store']

# Cleanup on exit
def cleanup():
    st.session_state['sumo_continue'] = False
    time.sleep(0.5)

atexit.register(cleanup)

# ===== STREAMLIT UI =====
st.set_page_config(
    page_title="SUMO Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Title
st.title("🚦 Multi-Intersection SUMO Dashboard")
st.markdown(f"Monitoring **{len(TL_IDS)}** traffic light intersections")

# Show SUMO status
if st.session_state.get('sumo_running', False):
    st.success("✅ SUMO Simulation Running")
else:
    st.warning("⏳ Starting SUMO Simulation...")

# Sidebar controls
with st.sidebar:
    st.subheader("⚙️ Controls")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_rate = st.slider("Refresh interval (seconds)", 1, 10, 2)
    
    st.markdown("---")
    
    # Traffic demand control
    st.subheader("🚗 Traffic Demand")
    
    demand_mode = st.radio(
        "Demand Mode",
        ["Manual", "Preset Profile"]
    )
    
    if demand_mode == "Manual":
        demand_level = st.slider(
            "Traffic Demand",
            min_value=0.1,
            max_value=2.5,
            value=st.session_state.get('demand_level', 1.0),
            step=0.1,
            help="0.5 = Light, 1.0 = Normal, 2.0+ = Heavy"
        )
        demand_display = f"{demand_level:.1f}x"
        
    else:  # Preset Profile
        profile = st.selectbox(
            "Select Profile",
            ["Light (0.5x)", "Normal (1.0x)", "Heavy (1.5x)", "Rush Hour (2.0x)", "Extreme (2.5x)"]
        )
        profile_map = {
            "Light (0.5x)": 0.5,
            "Normal (1.0x)": 1.0,
            "Heavy (1.5x)": 1.5,
            "Rush Hour (2.0x)": 2.0,
            "Extreme (2.5x)": 2.5,
        }
        demand_level = profile_map[profile]
        demand_display = profile.split("(")[1].strip(")")
    
    # Apply demand button
    if st.button("🔄 Apply Demand", key="apply_demand"):
        try:
            data_store.update_demand(demand_level)
            st.session_state['demand_level'] = demand_level
            st.success(f"✅ Demand set to {demand_display}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    
    # Stop button
    if st.button("⏹️ Stop SUMO"):
        st.session_state['sumo_continue'] = False
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Simulation Info")
    sim_step = st.empty()
    total_time = st.empty()
    current_demand_display = st.empty()

# Global metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    current_queue = data_store.global_queue[-1] if data_store.global_queue else 0
    st.metric("🌍 Global Queue", f"{current_queue} veh")
with col2:
    current_throughput = data_store.global_throughput[-1] if data_store.global_throughput else 0
    st.metric("🚗 Global Throughput", f"{current_throughput} veh/step")
with col3:
    total_vehicles = sum(data_store.global_throughput) if data_store.global_throughput else 0
    st.metric("📈 Total Vehicles", f"{total_vehicles}")
with col4:
    avg_queue = sum(data_store.global_queue) / len(data_store.global_queue) if data_store.global_queue else 0
    st.metric("📊 Avg Queue", f"{avg_queue:.1f} veh")

# Update sidebar info
sim_step.metric("Step", f"{data_store.current_step}")
total_time.metric("Sim Time", f"{data_store.current_step * 0.1:.1f} s")
current_demand_display.metric("🎯 Current Demand", f"{data_store.demand_level:.1f}x")

st.markdown("---")

# Global trend charts
st.subheader("📈 Global Trends")
col1, col2 = st.columns(2)

with col1:
    if data_store.time_history:
        fig_queue = go.Figure()
        fig_queue.add_trace(go.Scatter(
            x=list(data_store.time_history),
            y=list(data_store.global_queue),
            mode='lines',
            name='Queue Length',
            line=dict(color='#FF6B6B', width=2)
        ))
        fig_queue.update_layout(
            title="Total Queue Length Over Time",
            xaxis_title="Simulation Step",
            yaxis_title="Number of Vehicles",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_queue, use_container_width=True)

with col2:
    if data_store.time_history:
        fig_throughput = go.Figure()
        fig_throughput.add_trace(go.Scatter(
            x=list(data_store.time_history),
            y=list(data_store.global_throughput),
            mode='lines',
            name='Throughput',
            line=dict(color='#51CF66', width=2)
        ))
        fig_throughput.update_layout(
            title="Total Throughput Over Time",
            xaxis_title="Simulation Step",
            yaxis_title="Vehicles per Step",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_throughput, use_container_width=True)

st.markdown("---")

# Intersection details
st.subheader("🚥 Intersection Details")

# Create tabs for each intersection
tabs = st.tabs([f"🚦 {tl}" for tl in TL_IDS])

for idx, tl in enumerate(TL_IDS):
    with tabs[idx]:
        st.write(f"**Intersection ID:** `{tl}`")
        st.write(f"**Lanes:** {len(LANES.get(tl, []))}")
        
        # Get latest data with safety checks
        int_data = data_store.intersections[tl]
        latest_queue = int_data["queue"][-1] if len(int_data["queue"]) > 0 else 0
        latest_throughput = int_data["throughput"][-1] if len(int_data["throughput"]) > 0 else 0
        latest_phase = int_data["phase"][-1] if len(int_data["phase"]) > 0 else 0
        latest_occ = int_data["occupancy"][-1] if len(int_data["occupancy"]) > 0 else 0
        latest_wait = int_data["waiting_time"][-1] if len(int_data["waiting_time"]) > 0 else 0
        
        # Phase display
        phase_symbols = {0: "🔴 RED", 1: "🟡 YELLOW", 2: "🟢 GREEN", 3: "🟡 YELLOW"}
        phase_display = phase_symbols.get(int(latest_phase), f"Phase {int(latest_phase)}")
        
        # Metrics row - 5 metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🚗 Queue", f"{int(latest_queue)} veh")
        with col2:
            st.metric("📊 Throughput", f"{int(latest_throughput)} veh/step")
        with col3:
            st.metric("🎯 Phase", phase_display)
        with col4:
            st.metric("📈 Occupancy", f"{latest_occ:.2f}%")
        with col5:
            st.metric("⏱️ Wait Time", f"{latest_wait:.1f}s")
        
        st.markdown("---")
        
        # Charts - Queue and Throughput
        col1, col2 = st.columns(2)
        
        with col1:
            if len(int_data["queue"]) > 0:
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(
                    x=list(data_store.time_history),
                    y=list(int_data["queue"]),
                    mode='lines+markers',
                    name='Queue',
                    line=dict(color='#FF6B6B', width=2),
                    marker=dict(size=4)
                ))
                fig_q.update_layout(
                    title=f"Queue Length - {tl}",
                    xaxis_title="Time Step",
                    yaxis_title="Vehicles",
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_q, use_container_width=True)
            else:
                st.info("Waiting for data...")
        
        with col2:
            if len(int_data["throughput"]) > 0:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(
                    x=list(data_store.time_history),
                    y=list(int_data["throughput"]),
                    mode='lines+markers',
                    name='Throughput',
                    line=dict(color='#51CF66', width=2),
                    marker=dict(size=4)
                ))
                fig_t.update_layout(
                    title=f"Throughput - {tl}",
                    xaxis_title="Time Step",
                    yaxis_title="Vehicles/Step",
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Waiting for data...")
        
        # Charts - Occupancy and Waiting Time
        col1, col2 = st.columns(2)
        
        with col1:
            if len(int_data["occupancy"]) > 0:
                fig_occ = go.Figure()
                fig_occ.add_trace(go.Scatter(
                    x=list(data_store.time_history),
                    y=list(int_data["occupancy"]),
                    mode='lines',
                    name='Occupancy',
                    line=dict(color='#4C6EF5', width=2),
                    fill='tozeroy'
                ))
                fig_occ.update_layout(
                    title=f"Lane Occupancy - {tl}",
                    xaxis_title="Time Step",
                    yaxis_title="Occupancy %",
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_occ, use_container_width=True)
        
        with col2:
            if len(int_data["waiting_time"]) > 0:
                fig_wait = go.Figure()
                fig_wait.add_trace(go.Scatter(
                    x=list(data_store.time_history),
                    y=list(int_data["waiting_time"]),
                    mode='lines',
                    name='Waiting Time',
                    line=dict(color='#FFD43B', width=2),
                    fill='tozeroy'
                ))
                fig_wait.update_layout(
                    title=f"Avg Waiting Time - {tl}",
                    xaxis_title="Time Step",
                    yaxis_title="Seconds",
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_wait, use_container_width=True)

st.markdown("---")

# Auto-refresh without flickering
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()