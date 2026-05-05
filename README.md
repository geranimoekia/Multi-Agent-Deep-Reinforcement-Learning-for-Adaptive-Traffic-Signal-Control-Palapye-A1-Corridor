# DRL Traffic Signal Control

Deep RL traffic signal control using PPO + SUMO. An agent learns when to switch the light green across 3 coordinated intersections — curriculum training, dynamic demand injection, live Streamlit dashboard.

---

## Overview

The agent observes queue lengths, cumulative waiting times, and outgoing flow at each intersection every 3 simulation seconds. It then decides which green phase to hold or switch to. Traffic is injected dynamically according to configurable demand scenarios (rush hour, holiday, incident, etc.).

**Network:** Triple-intersection layout based on a real OSM network  
**Agent:** PPO (Proximal Policy Optimization) via Stable Baselines3  
**Simulator:** SUMO 1.24+ controlled via TraCI  
**Dashboard:** Streamlit live monitor with model toggle

---

## Project Structure

```
DRL-TRAFFIC/
├── network/                        # SUMO simulation files
│   ├── triple.sumocfg              # Simulation config
│   ├── network_tripled.net.xml     # Road network
│   ├── triple_routes_flows.rou.xml # OD pair definitions
│   ├── empty.rou.xml               # Empty routes (traffic injected dynamically)
│   └── vtypes.add.xml              # Vehicle type definitions
│
├── sumo_env.py                     # Gymnasium RL environment
├── train_ppo.py                    # PPO training script
├── dashboard.py                    # Streamlit live dashboard
├── traffic_injector.py             # Dynamic vehicle injection
├── traffic_scenario.py             # Demand scenario profiles
├── green_wave.py                   # Green wave coordination (rule-based)
├── tl_programs.py                  # Traffic light phase programs
│
├── ppo_models/                     # Saved model checkpoints (gitignored)
├── ppo_logs/                       # TensorBoard training logs (gitignored)
├── v1.0/                           # Archive of initial version
└── .gitignore
```

---

## Setup

**Requirements:** Python 3.10+, SUMO 1.24+

```bash
# Install dependencies
pip install stable-baselines3 gymnasium streamlit plotly traci

# Set SUMO_HOME if not set (Windows example)
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

---

## Training

```bash
python train_ppo.py
```

- Runs **2 parallel environments** via `SubprocVecEnv`
- Trains for **5 million timesteps**
- Saves checkpoints every 10 000 steps to `ppo_models/`
- Evaluates every 20 000 steps and saves the best model as `ppo_models/best_model.zip`
- View training progress: `tensorboard --logdir ppo_logs`

### Curriculum

| Episodes | Scenarios active |
|---|---|
| 0–29 | `low`, `normal` |
| 30–79 | `normal`, `rush_hour_am`, `rush_hour_pm`, `holiday` |
| 80+ | All scenarios (full random) |

---

## Dashboard

```bash
streamlit run dashboard.py
```

Opens a live Streamlit interface with:

- **Live Monitor** — real-time queue length and throughput charts
- **Traffic Analysis** — origin-destination flow heatmap
- **Control & Configuration** — scenario selector, green wave toggle, PPO model toggle

To activate the PPO agent: sidebar → select `best_model.zip` → toggle **Enable Model Control**.

---

## RL Design

| Component | Detail |
|---|---|
| **Observation** | 66 features per step: halting count (log-scaled) + waiting time (log-scaled) + outgoing vehicle count + current phase + phase state, for each of 3 TLs |
| **Action** | `MultiDiscrete([2, 2, 3])` — green phase selection per TL |
| **Reward** | `−tanh(avg_wait / 60s)` − 0.1 × queue ratio − collision/teleport penalties |
| **Episode length** | 1 200 simulation steps (~60 min simulated time) |
| **Min green** | 5 RL steps (15 sim seconds) before a phase switch is allowed |

---

## Traffic Scenarios

| Scenario | Description |
|---|---|
| `low` | Off-peak, sparse traffic |
| `normal` | Weekday mixed flow |
| `rush_hour_am` | Heavy inbound morning traffic |
| `rush_hour_pm` | Heavy outbound evening traffic |
| `holiday` | Transit traffic through town |
| `incident` | Two entry points blocked |
