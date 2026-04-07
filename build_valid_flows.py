"""
build_valid_flows.py
--------------------
Run this from C:\\DRL_TRAFFIC\\ :
    python build_valid_flows.py

It will:
1. Test every IN x OUT edge pair using duarouter
2. Print a table of valid / invalid pairs
3. Write triple_routes_flows.rou.xml using ONLY valid pairs
4. Write triple.sumocfg with the correct ignore flag
"""

import subprocess
import os
import sys
import tempfile
import itertools

# ── configuration ────────────────────────────────────────────────────────────
NET_FILE   = "network_tripled.net.xml"
OUT_ROUTES = "triple_routes_flows.rou.xml"
OUT_CFG    = "triple.sumocfg"
VTYPES     = "vtypes.add.xml"
SIM_END    = 5000

IN_EDGES = [
    "-465932558#2_C",
    "470773638#0_C",
    "-E5_B",
    "470773638#0_B",
    "-E5",
    "470773638#0",
    "-470773638#0",
    "E0",
]

OUT_EDGES = [
    "465932558#2_C",
    "-470773638#0_C",
    "E5_B",
    "-470773638#0_B",
    "E5",
    "E6",
]

# Traffic phases: (begin, end, period_seconds)
PHASES = [
    (0,    1000, 60),
    (1000, 2000, 30),
    (2000, 3000, 15),
    (3000, 4000,  8),
    (4000, 5000,  4),
]

# Vehicle types to cycle through
VTYPES_LIST = [
    "car_normal", "car_small", "car_suv", "car_sport",
    "moto", "truck", "bus_city", "car_normal", "car_small",
]
# ─────────────────────────────────────────────────────────────────────────────


def find_sumo_home():
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        return sumo_home
    # common Windows install paths
    for base in [r"C:\Program Files (x86)\Eclipse\Sumo",
                 r"C:\Program Files\Eclipse\Sumo",
                 r"C:\Sumo"]:
        if os.path.isdir(base):
            return base
    return None


def test_pair(duarouter_exe, net_file, from_edge, to_edge):
    """Return True if duarouter can find a route from_edge -> to_edge."""
    with tempfile.TemporaryDirectory() as tmp:
        trip_file  = os.path.join(tmp, "test.trips.xml")
        route_file = os.path.join(tmp, "test.rou.xml")

        with open(trip_file, "w") as f:
            f.write('<?xml version="1.0"?>\n<trips>\n')
            f.write(f'  <trip id="t0" from="{from_edge}" to="{to_edge}" depart="0"/>\n')
            f.write('</trips>\n')

        result = subprocess.run(
            [duarouter_exe,
             "--net-file",   net_file,
             "--trip-files", trip_file,
             "--output-file", route_file,
             "--ignore-errors", "true",
             "--no-warnings", "true",
             "--no-step-log",  "true"],
            capture_output=True, text=True
        )
        # If the route file exists and contains a <vehicle or <route tag, it worked
        if os.path.isfile(route_file):
            content = open(route_file).read()
            return "<vehicle" in content or "<route" in content
        return False


def main():
    if not os.path.isfile(NET_FILE):
        sys.exit(f"ERROR: {NET_FILE} not found. Run this script from C:\\DRL_TRAFFIC\\")

    sumo_home = find_sumo_home()
    if not sumo_home:
        sys.exit("ERROR: SUMO_HOME not set and SUMO not found in default paths.\n"
                 "Set it with:  set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")

    duarouter = os.path.join(sumo_home, "bin", "duarouter.exe")
    if not os.path.isfile(duarouter):
        sys.exit(f"ERROR: duarouter.exe not found at {duarouter}")

    print(f"Using SUMO from: {sumo_home}")
    print(f"Testing {len(IN_EDGES)} x {len(OUT_EDGES)} = {len(IN_EDGES)*len(OUT_EDGES)} edge pairs...\n")

    valid_pairs = []
    for frm, to in itertools.product(IN_EDGES, OUT_EDGES):
        ok = test_pair(duarouter, NET_FILE, frm, to)
        symbol = "✓" if ok else "✗"
        print(f"  {symbol}  {frm:30s}  ->  {to}")
        if ok:
            valid_pairs.append((frm, to))

    print(f"\n{len(valid_pairs)} valid pairs found out of {len(IN_EDGES)*len(OUT_EDGES)}\n")

    if not valid_pairs:
        sys.exit("ERROR: No valid pairs found at all. Check your network file.")

    # ── write route file ──────────────────────────────────────────────────
    vtype_cycle = itertools.cycle(VTYPES_LIST)
    flow_id     = 0
    lines       = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    lines.append('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')

    for phase_idx, (begin, end, period) in enumerate(PHASES, 1):
        lines.append(f'    <!-- Phase {phase_idx}: {begin}-{end}s  period={period}s -->')
        for frm, to in valid_pairs:
            vtype = next(vtype_cycle)
            lines.append(
                f'    <flow id="f_p{phase_idx}_{flow_id:04d}" '
                f'from="{frm}" to="{to}" '
                f'begin="{begin}" end="{end}" '
                f'period="{period}" type="{vtype}"/>'
            )
            flow_id += 1
        lines.append('')

    lines.append('</routes>')
    lines.append('')

    with open(OUT_ROUTES, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {OUT_ROUTES}  ({flow_id} flow entries)")

    # ── write sumocfg ─────────────────────────────────────────────────────
    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{NET_FILE}"/>
        <route-files value="{OUT_ROUTES}"/>
        <additional-files value="{VTYPES}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{SIM_END}"/>
    </time>

    <processing>
        <ignore-route-errors value="true"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="false"/>
    </report>

</configuration>
"""
    with open(OUT_CFG, "w") as f:
        f.write(cfg)
    print(f"Written: {OUT_CFG}")
    print("\nDone! Now run:  sumo-gui -c triple.sumocfg")


if __name__ == "__main__":
    main()
