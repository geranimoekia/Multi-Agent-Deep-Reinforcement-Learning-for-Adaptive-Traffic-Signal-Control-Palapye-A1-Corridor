"""
SUMO Multi-Intersection Dashboard
Run: python app.py
Open: http://localhost:5050

pip install flask traci
"""

import threading
import time
import os
import atexit
from collections import deque
from flask import Flask, jsonify, send_from_directory, request

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("WARNING: traci not found — running in demo mode with fake data")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SUMO_CFG  = "triple.sumocfg"
USE_GUI   = True
MAX_HIST  = 120
PORT      = 5050

TL_IDS = ["6073919354", "6073919354_B", "6073919354_C"]

LANES = {
    "6073919354": [
        "-E5_0", "-465932558#1.34_0", "-465932558#1.34_1",
        "-465932558#1.34_2", "470773638#1_0", "465932558#0_0", "465932558#0_1",
    ],
    "6073919354_B": [
        "-E5_B_0", "-465932558#1.34_B_0", "-465932558#1.34_B_1",
        "-465932558#1.34_B_2", "470773638#1_B_0", "465932558#0_B_0", "465932558#0_B_1",
    ],
    "6073919354_C": ["-465932558#2_C_0", "470773638#1_C_0", "E0_C_0"],
}

PHASE_LABELS = {0: "RED", 1: "YELLOW", 2: "GREEN", 3: "YELLOW"}


# ──────────────────────────────────────────────
# DATA STORE
# ──────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self.lock         = threading.Lock()
        self.step         = 0
        self.demand       = 1.0
        self.running      = False
        self.g_time       = deque(maxlen=MAX_HIST)
        self.g_queue      = deque(maxlen=MAX_HIST)
        self.g_throughput = deque(maxlen=MAX_HIST)
        self.flow_periods = {}
        self.tl = {
            tl: {
                k: deque(maxlen=MAX_HIST)
                for k in ("queue","throughput","occupancy","waiting_time","phase")
            } | {"latest": {}}
            for tl in TL_IDS
        }

    # called from SUMO thread — no lock needed (single writer)
    def update(self):
        self.step += 1
        gq = gt = 0

        for tl in TL_IDS:
            qs, ts, os_, ws = [], [], [], []
            for lane in LANES.get(tl, []):
                try:
                    qs.append(traci.lane.getLastStepHaltingNumber(lane))
                    ts.append(traci.lane.getLastStepVehicleNumber(lane))
                    os_.append(traci.lane.getLastStepOccupancy(lane))
                    ws.append(traci.lane.getWaitingTime(lane))
                except Exception:
                    pass

            q   = sum(qs)  if qs  else 0
            t   = sum(ts)  if ts  else 0
            occ = sum(os_) / len(os_) if os_ else 0
            wt  = sum(ws)  if ws  else 0
            try:    phase = traci.trafficlight.getPhase(tl)
            except: phase = 0

            d = self.tl[tl]
            d["queue"].append(q);        d["throughput"].append(t)
            d["occupancy"].append(occ);  d["waiting_time"].append(wt)
            d["phase"].append(phase)
            d["latest"] = {
                "queue": q, "throughput": t,
                "occupancy": round(occ, 2), "waiting_time": round(wt, 1),
                "phase": phase,
                "phase_label": PHASE_LABELS.get(int(phase), f"P{phase}"),
            }
            gq += q;  gt += t

        with self.lock:
            self.g_time.append(self.step)
            self.g_queue.append(gq)
            self.g_throughput.append(gt)

    # called from Flask thread — needs lock for global deques
    def snapshot(self):
        with self.lock:
            times = list(self.g_time)
            gq    = list(self.g_queue)
            gt    = list(self.g_throughput)

        tl_out = {}
        for tl in TL_IDS:
            d = self.tl[tl]
            tl_out[tl] = {
                "latest":      d["latest"],
                "queue":       list(d["queue"]),
                "throughput":  list(d["throughput"]),
                "occupancy":   [round(v,2) for v in d["occupancy"]],
                "waiting_time":[round(v,1) for v in d["waiting_time"]],
            }

        return {
            "step":        self.step,
            "sim_time":    round(self.step * 0.1, 1),
            "demand":      self.demand,
            "running":     self.running,
            "time":        times,
            "g_queue":     gq,
            "g_throughput":gt,
            "cur_queue":   gq[-1] if gq else 0,
            "cur_tput":    gt[-1] if gt else 0,
            "avg_queue":   round(sum(gq)/len(gq), 1) if gq else 0,
            "total_veh":   sum(gt),
            "intersections": tl_out,
        }

    def set_demand(self, mult):
        try:
            for fid in traci.route.getFlowIDList():
                if fid not in self.flow_periods:
                    try: self.flow_periods[fid] = traci.flow.getFlowPeriod(fid)
                    except: continue
                try: traci.flow.setFlowPeriod(fid, max(1, int(self.flow_periods[fid]/mult)))
                except: pass
        except Exception as e:
            print(f"demand error: {e}")
        self.demand = mult


# ──────────────────────────────────────────────
# SIMULATION THREADS
# ──────────────────────────────────────────────
store      = DataStore()
stop_event = threading.Event()


def run_sumo():
    if not TRACI_AVAILABLE:
        _run_demo()
        return
    try:
        traci.start(["sumo-gui" if USE_GUI else "sumo", "-c", SUMO_CFG])
        store.running = True
        while not stop_event.is_set():
            traci.simulationStep()
            store.update()
            time.sleep(0.05)
        traci.close()
    except Exception as e:
        print(f"SUMO error: {e}")
    finally:
        store.running = False


def _run_demo():
    """Fake data so the dashboard works without SUMO installed."""
    import math, random
    store.running = True
    while not stop_event.is_set():
        store.step += 1
        s = store.step
        gq = gt = 0
        for tl in TL_IDS:
            q  = int(5 + 4*math.sin(s/20) + random.random()*2)
            t  = int(8 + 3*math.cos(s/15) + random.random()*2)
            oc = round(20 + 10*math.sin(s/25) + random.random()*3, 2)
            wt = round(15 + 8*math.sin(s/30) + random.random()*2, 1)
            ph = (s // 15) % 4
            d  = store.tl[tl]
            d["queue"].append(q);       d["throughput"].append(t)
            d["occupancy"].append(oc);  d["waiting_time"].append(wt)
            d["phase"].append(ph)
            d["latest"] = {"queue":q,"throughput":t,"occupancy":oc,
                           "waiting_time":wt,"phase":ph,
                           "phase_label":PHASE_LABELS.get(ph,f"P{ph}")}
            gq += q;  gt += t
        with store.lock:
            store.g_time.append(s)
            store.g_queue.append(gq)
            store.g_throughput.append(gt)
        time.sleep(0.1)
    store.running = False


threading.Thread(target=run_sumo, daemon=True).start()
atexit.register(lambda: stop_event.set())


# ──────────────────────────────────────────────
# FLASK
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder=".")

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/data")
def api_data():
    return jsonify(store.snapshot())

@app.route("/api/demand", methods=["POST"])
def api_demand():
    mult = float(request.json.get("level", 1.0))
    mult = max(0.1, min(3.0, mult))
    store.set_demand(mult)
    return jsonify({"ok": True, "level": mult})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    stop_event.set()
    return jsonify({"ok": True})

if __name__ == "__main__":
    print(f"\n  Dashboard → http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)