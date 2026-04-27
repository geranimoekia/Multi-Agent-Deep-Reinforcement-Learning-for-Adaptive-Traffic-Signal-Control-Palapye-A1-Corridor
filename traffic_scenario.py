import random

# ── Edge classification for this network ─────────────────────────────────────
# "Inbound"  = coming from the highway into town
HIGHWAY_ENTRIES: set[str] = {"E0", "E0_B", "E0_C", "-E5", "-E5_B"}
LOCAL_DESTS: set[str]     = {"465932558#2_C", "-470773638#0_B", "-470773638#0_C"}

# "Outbound" = leaving town onto the highway
LOCAL_ORIGINS: set[str]   = {"-465932558#2_C", "470773638#0", "470773638#0_B", "470773638#0_C"}
HIGHWAY_EXITS: set[str]   = {"E5", "E5_B", "E6", "E6_B", "E6_C"}

# "Transit"  = entering/leaving town via the main corridor (E0 ↔ 465932558#2_C)
TRANSIT_ENTRIES: set[str] = {"E0", "E0_B", "E0_C"}
TRANSIT_EXITS: set[str]   = {"465932558#2_C", "-465932558#2_C"}

# ── Scenario profiles ─────────────────────────────────────────────────────────
# direction_mode: which OD subset inject() draws from
#   "any"      → all active pairs (background traffic)
#   "inbound"  → highway entry → local dest  (morning rush, people going to work)
#   "outbound" → local origin → highway exit (evening rush, people going home)
#   "transit"  → E0 / 465932558#2_C corridor (holiday through-traffic)
#
# phase_mults: per-1000-step multipliers [0-1k, 1-2k, 2-3k, 3-4k, 4-5k]
# base_rate  × phase_mult = injection probability per simulation step (capped at 1.0)

PROFILES: dict[str, dict] = {
    "low": {
        "base_rate": 0.10,
        "phase_mults": [0.3, 0.5, 0.7, 0.9, 1.0],
        "blocked_origins": [],
        "direction_mode": "any",
        "description": "Light off-peak traffic, dispersed routes",
    },
    "normal": {
        "base_rate": 0.20,
        "phase_mults": [0.3, 0.6, 1.2, 2.0, 3.5],
        "blocked_origins": [],
        "direction_mode": "any",
        "description": "Typical weekday mixed demand",
    },
    "rush_hour_am": {
        # Morning: peaks early (people flooding in), then eases off
        "base_rate": 0.45,
        "phase_mults": [5.0, 4.0, 2.5, 1.0, 0.5],
        "blocked_origins": [],
        "direction_mode": "inbound",
        "description": "Morning rush — high inbound flow, people going to work",
    },
    "rush_hour_pm": {
        # Evening: builds up late (people leaving work), peaks near end
        "base_rate": 0.45,
        "phase_mults": [0.5, 1.0, 2.5, 4.0, 5.0],
        "blocked_origins": [],
        "direction_mode": "outbound",
        "description": "Evening rush — high outbound flow, people going home",
    },
    "holiday": {
        # Transit area: sustained high volume all day, both directions through town
        "base_rate": 0.40,
        "phase_mults": [2.5, 2.8, 3.0, 2.8, 2.5],
        "blocked_origins": [],
        "direction_mode": "transit",
        "description": "Holiday — sustained transit traffic through town (E0 ↔ 465932558#2_C)",
    },
    "incident": {
        "base_rate": 0.20,
        "phase_mults": [0.3, 0.6, 1.2, 2.0, 3.5],
        "blocked_origins": ["-E5_B", "-E5"],
        "direction_mode": "any",
        "description": "Normal demand but two highway entries closed",
    },
}

PROFILE_NAMES: list[str] = list(PROFILES.keys())


class TrafficScenario:
    def __init__(self, name: str | None = None):
        if name is None:
            name = random.choice(PROFILE_NAMES)
        if name not in PROFILES:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {PROFILE_NAMES}")
        self.name = name
        self._p = PROFILES[name]

    def demand_rate(self, step: int) -> float:
        """Injection probability for this step — may exceed 1.0 (always inject)."""
        phase_idx = min(step // 1000, len(self._p["phase_mults"]) - 1)
        return self._p["base_rate"] * self._p["phase_mults"][phase_idx]

    @property
    def direction_mode(self) -> str:
        return self._p["direction_mode"]

    @property
    def blocked_origins(self) -> set[str]:
        return set(self._p["blocked_origins"])

    @property
    def description(self) -> str:
        return self._p["description"]

    @staticmethod
    def random() -> "TrafficScenario":
        return TrafficScenario(random.choice(PROFILE_NAMES))

    def __repr__(self) -> str:
        return f"TrafficScenario('{self.name}')"
