"""
Traffic light program definitions — only depends on traci.
Isolated here so it can be imported by both dashboard.py and sumo_env.py
without pulling in gymnasium or numpy.
"""

import traci


def apply_tl_programs():
    """
    Override TL C with a strictly conflict-free 3-phase program.

    6073919354_C is a T-junction with 3 approaches that ALL conflict:
      - Approaches A+C share the north exit  → merge conflict
      - Approaches B+C cross on the north road → head-on conflict
      - Approaches A+B share the E6 exit      → merge conflict

    Solution: one approach at a time.

      Phase 0 (41s): Approach A only  (-465932558#2_C)
        link 0  -465932558#2_C → E6_C           G
        link 1  -465932558#2_C → -470773638#1_C G
        all others: r

      Phase 2 (41s): Approach B only  (470773638#1_C)
        link 2  470773638#1_C → 465932558#2_C   G
        link 3  470773638#1_C → E6_C            G
        all others: r

      Phase 4 (41s): Approach C only  (E0_C)
        link 4  E0_C → -470773638#1_C           G
        link 5  E0_C → 465932558#2_C            G
        all others: r
    """
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
