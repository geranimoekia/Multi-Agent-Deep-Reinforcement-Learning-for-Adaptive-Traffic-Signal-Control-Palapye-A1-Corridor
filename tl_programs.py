"""
Traffic light program definitions — only depends on traci.
Isolated here so it can be imported by both dashboard.py and sumo_env.py
without pulling in gymnasium or numpy.
"""

import traci


def _apply_2phase(tl_id: str, n_links: int):
    """
    Apply a generic 2-phase program to a standard cross-intersection.
    Phase 0 (index 0): first half of links green  → yellow at index 1
    Phase 4 (index 4): second half of links green → yellow at index 5
    This guarantees the PPO agent's expected phase indices (0, 4) exist.
    """
    half = n_links // 2
    # Phase 0: first half G, second half r
    p0_state = "G" * half + "r" * (n_links - half)
    p0_yell  = "y" * half + "r" * (n_links - half)
    # Phase 4: first half r, second half G
    p4_state = "r" * half + "G" * (n_links - half)
    p4_yell  = "r" * half + "y" * (n_links - half)

    phases = [
        traci.trafficlight.Phase(41, p0_state),  # phase 0 — green A
        traci.trafficlight.Phase(4,  p0_yell),   # phase 1 — yellow A
        traci.trafficlight.Phase(2,  "r" * n_links),  # phase 2 — all-red
        traci.trafficlight.Phase(2,  "r" * n_links),  # phase 3 — all-red
        traci.trafficlight.Phase(41, p4_state),  # phase 4 — green B
        traci.trafficlight.Phase(4,  p4_yell),   # phase 5 — yellow B
    ]
    logic = traci.trafficlight.Logic("ppo_2phase", 0, 0, phases)
    traci.trafficlight.setProgramLogic(tl_id, logic)
    traci.trafficlight.setProgram(tl_id, "ppo_2phase")


def apply_tl_programs():
    """
    Apply known conflict-free programs to all three TLs so that the PPO
    agent's expected phase indices (0 and 4 as major greens, 1 and 5 as
    yellows) are guaranteed to exist regardless of the SUMO default program.
    """
    # ── TL_A and TL_B: standard 2-phase cross-intersection ──────────────────
    for tl_id in ("6073919354", "6073919354_B"):
        try:
            n_links = len(traci.trafficlight.getControlledLinks(tl_id))
            if n_links < 2:
                print(f"[TL CONFIG] {tl_id}: too few links ({n_links}), skipping")
                continue
            _apply_2phase(tl_id, n_links)
            print(f"[TL CONFIG] {tl_id}: 2-phase program applied ({n_links} links)")
        except Exception as e:
            print(f"[TL CONFIG] Failed to apply {tl_id} program: {e}")

    # ── TL_C: T-junction needs strict one-approach-at-a-time program ────────
    tl_id = "6073919354_C"
    try:
        #            link:  0  1  2  3  4  5
        phases = [
            traci.trafficlight.Phase(41, "GGrrrr"),  # approach A open
            traci.trafficlight.Phase(4,  "yyrrrr"),  # yellow A
            traci.trafficlight.Phase(41, "rrGGrr"),  # approach B open
            traci.trafficlight.Phase(4,  "rryyrr"),  # yellow B
            traci.trafficlight.Phase(41, "rrrrGG"),  # approach C (E0) open
            traci.trafficlight.Phase(4,  "rrrryy"),  # yellow C
        ]
        logic = traci.trafficlight.Logic("tjunc_3phase", 0, 0, phases)
        traci.trafficlight.setProgramLogic(tl_id, logic)
        traci.trafficlight.setProgram(tl_id, "tjunc_3phase")
        print(f"[TL CONFIG] {tl_id}: 3-phase conflict-free program applied")
    except Exception as e:
        print(f"[TL CONFIG] Failed to apply {tl_id} program: {e}")
