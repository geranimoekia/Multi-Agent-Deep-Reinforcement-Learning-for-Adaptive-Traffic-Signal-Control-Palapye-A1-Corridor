import xml.etree.ElementTree as ET
import random
import traci
from traffic_scenario import (
    TrafficScenario,
    HIGHWAY_ENTRIES, LOCAL_DESTS,
    LOCAL_ORIGINS, HIGHWAY_EXITS,
    TRANSIT_ENTRIES, TRANSIT_EXITS,
)

FLOW_FILE = "triple_routes_flows.rou.xml"
TYPES = ["car_small", "car_normal", "car_suv", "car_sport", "moto", "truck", "bus_city"]

# Built once at init() from the live network — never changes mid-episode.
_all_od_pairs: list[tuple[str, str]] = []

# Directional subsets — rebuilt cheaply when scenario changes.
_inbound_pairs: list[tuple[str, str]] = []   # highway entry  → local dest
_outbound_pairs: list[tuple[str, str]] = []  # local origin   → highway exit
_transit_pairs: list[tuple[str, str]] = []   # E0/local entry ↔ 465932558#2_C
_active_pairs: list[tuple[str, str]] = []    # pool used by inject()

_scenario: TrafficScenario | None = None
_veh_count = 0
_scenario_step = 0   # resets on scenario change — keeps demand within training distribution
injection_log: list[dict] = []


def init(scenario: TrafficScenario | None = None):
    """
    Call once after traci.start().
    Validates all OD pairs against the live SUMO network, then applies the scenario.
    Resets vehicle counter and log — safe to call again on episode reset.
    """
    global _all_od_pairs, _veh_count, _scenario_step, injection_log

    _veh_count = 0
    _scenario_step = 0
    injection_log.clear()

    if scenario is None:
        scenario = TrafficScenario("normal")

    # Parse raw OD pairs from the reference flow file
    raw: list[tuple[str, str]] = []
    try:
        root = ET.parse(FLOW_FILE).getroot()
        for flow in root.findall("flow"):
            frm, to = flow.get("from"), flow.get("to")
            if frm and to:
                raw.append((frm, to))
    except Exception as e:
        print(f"[INJECTOR] Could not parse {FLOW_FILE}: {e}")
        return

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    unique_raw = [p for p in raw if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    # Test each pair against the live network
    live_edges = set(traci.edge.getIDList())
    valid: list[tuple[str, str]] = []
    skipped_edge = skipped_route = 0

    for frm, to in unique_raw:
        if frm not in live_edges or to not in live_edges:
            skipped_edge += 1
            continue
        r = traci.simulation.findRoute(frm, to)
        if r and r.edges:
            valid.append((frm, to))
        else:
            skipped_route += 1

    _all_od_pairs = valid
    print(
        f"[INJECTOR] {len(_all_od_pairs)} valid OD pairs "
        f"(dropped {skipped_edge} bad-edge, {skipped_route} no-route)"
    )

    _apply_scenario(scenario)


def set_scenario(scenario: TrafficScenario):
    """Hot-swap the scenario without restarting SUMO."""
    global _scenario_step
    _scenario_step = 0
    _apply_scenario(scenario)


def _apply_scenario(scenario: TrafficScenario):
    global _inbound_pairs, _outbound_pairs, _transit_pairs, _active_pairs, _scenario
    _scenario = scenario
    blocked = scenario.blocked_origins

    # Remove blocked origins
    available = [(o, d) for o, d in _all_od_pairs if o not in blocked]

    _inbound_pairs  = [(o, d) for o, d in available if o in HIGHWAY_ENTRIES and d in LOCAL_DESTS]
    _outbound_pairs = [(o, d) for o, d in available if o in LOCAL_ORIGINS   and d in HIGHWAY_EXITS]
    _transit_pairs  = [(o, d) for o, d in available
                       if (o in TRANSIT_ENTRIES and d in TRANSIT_EXITS)
                       or (o in TRANSIT_EXITS   and d in HIGHWAY_EXITS)]

    mode = scenario.direction_mode
    if mode == "inbound":
        pool = _inbound_pairs
    elif mode == "outbound":
        pool = _outbound_pairs
    elif mode == "transit":
        pool = _transit_pairs
    else:
        pool = available

    # Fallback to full available set if the directional pool is empty
    _active_pairs = pool if pool else available

    print(
        f"[INJECTOR] '{scenario.name}' | mode={mode} | "
        f"pool={len(_active_pairs)} pairs "
        f"(inbound={len(_inbound_pairs)}, outbound={len(_outbound_pairs)}, "
        f"transit={len(_transit_pairs)}, blocked={len(_all_od_pairs)-len(available)})"
    )


def inject(step: int):
    """Inject one vehicle per call according to the active scenario."""
    global _veh_count, _scenario_step

    if not _active_pairs or _scenario is None:
        return

    rate = _scenario.demand_rate(_scenario_step)
    _scenario_step += 1
    # When rate >= 1.0, always inject; otherwise inject with probability = rate
    if rate < 1.0 and random.random() > rate:
        return

    origin, dest = random.choice(_active_pairs)

    route_obj = traci.simulation.findRoute(origin, dest)
    if not route_obj or not route_obj.edges:
        return

    veh_id = f"veh_{_veh_count}"
    route_id = f"route_{veh_id}"
    vtype = random.choice(TYPES)

    try:
        traci.route.add(route_id, route_obj.edges)
        traci.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=vtype,
            depart="now",
        )
        _veh_count += 1
        injection_log.append({
            "step": step,
            "veh_id": veh_id,
            "origin": origin,
            "destination": dest,
            "route_len": len(route_obj.edges),
            "type": vtype,
            "scenario": _scenario.name,
        })
    except Exception as e:
        print(f"[INJECT FAIL] {veh_id}: {e}")


def current_scenario() -> TrafficScenario | None:
    return _scenario


def vehicle_count() -> int:
    return _veh_count
