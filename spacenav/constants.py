"""Game-scale units and tunables.

Units are arbitrary "game units" (u) and game seconds (s) with G = 1, so every
mass below is really a gravitational parameter GM.  Scales are chosen for play,
not realism: a typical level is ~1000-3000 u across, planets move visibly
during an episode, and the ship's thrust is comparable to local gravity so that
orbital mechanics matter but never make a level unwinnable.

Everything that shapes game feel lives here so tuning happens in one place.
"""

# --- time ------------------------------------------------------------------
PHYS_DT = 1.0 / 60.0          # physics substep (s); one render frame at 60 fps
SUBSTEPS = 4                  # physics substeps per control tick
CTRL_DT = PHYS_DT * SUBSTEPS  # agent / player decision interval (15 Hz)
MAX_EPISODE_TIME = 150.0      # s (tours on a tight fuel budget coast a long way)
MAX_EPISODE_TICKS = int(round(MAX_EPISODE_TIME / CTRL_DT))

# --- padding ---------------------------------------------------------------
MAX_BODIES = 32               # bodies incl. massless tracers (stations, cloud centres)
MAX_ZONES = 12
SNAPSHOT_INTERVAL = 20.0      # s between stored level snapshots (episode start epochs)
LEVEL_HORIZON = 300.0         # s a level must stay well-behaved for

# --- ship ------------------------------------------------------------------
# Thrust and fuel are per-mission (see env.sample_loadout): a ship whose delta-v
# dwarfs local orbital speeds can ignore gravity and fly straight at the target,
# which is exactly the degenerate strategy these ranges are meant to prevent.
SHIP_ACCEL = 4.0              # u/s^2 at full throttle (default / upper end)
SHIP_ACCEL_RANGE = (0.6, 3.0)
SHIP_TURN_RATE = 3.5          # rad/s at full stick (arcade: no angular inertia)
SHIP_FUEL = 30.0              # s of full-throttle burn (default)
SHIP_FUEL_LIMITS = (3.0, 60.0)
DELTAV_BUDGET_RANGE = (0.55, 1.35)   # fuel as a multiple of the direct-flight estimate
SHIP_HEALTH = 100.0

# --- waypoint tours --------------------------------------------------------
MAX_WAYPOINTS = 6          # a Level-0 coast tour wants a long chain of targets

# --- hazards ---------------------------------------------------------------
ATMO_SCALE_FRAC = 0.25        # density e-folds every (atmo thickness * this)
ATMO_DRAG = 0.05              # drag accel = k * rho * |v_rel|^2
ATMO_HEAT = 0.004             # damage/s  = k * rho * |v_rel|^3
RADIATION_K = 1.0             # damage/s  = k * lum / r^2
DEBRIS_K = 0.35               # damage/s  = k * strength * |v_rel_to_debris_flow|
ZONE_EDGE = 0.15              # soft-edge width as a fraction of zone thickness
OUT_OF_BOUNDS_FACTOR = 1.35   # dead if |pos| > level_radius * this

# --- targets ---------------------------------------------------------------
TARGET_RADIUS = 12.0
RENDEZVOUS_V_TOL = 3.0        # u/s relative speed allowed for a rendezvous
FLYTHROUGH_V_TOL = 1e9

# --- body kinds ------------------------------------------------------------
KIND_NONE = 0
KIND_STAR = 1
KIND_ROCKY = 2
KIND_GIANT = 3
KIND_MOON = 4
KIND_BLACK_HOLE = 5
KIND_STATION = 6              # massless tracer, rendezvous target
KIND_ASTEROID = 7
KIND_TRACER = 8               # massless, invisible; anchors a moving zone
KIND_NEUTRON = 9

KIND_NAMES = {
    KIND_NONE: "none", KIND_STAR: "star", KIND_ROCKY: "rocky", KIND_GIANT: "giant",
    KIND_MOON: "moon", KIND_BLACK_HOLE: "black_hole", KIND_STATION: "station",
    KIND_ASTEROID: "asteroid", KIND_TRACER: "tracer", KIND_NEUTRON: "neutron_star",
}

# --- zone kinds ------------------------------------------------------------
ZONE_NONE = 0
ZONE_DEBRIS = 1               # damage scales with speed relative to the debris flow
ZONE_RADIATION = 2            # flat damage rate while inside (radiation belts, jets)
ZONE_SENSOR = 3               # harmless; only matters for the exposure objective

ZONE_NAMES = {ZONE_NONE: "none", ZONE_DEBRIS: "debris",
              ZONE_RADIATION: "radiation", ZONE_SENSOR: "sensor"}

# --- objectives ------------------------------------------------------------
OBJECTIVES = ("time", "path", "fuel", "exposure")
N_OBJECTIVES = len(OBJECTIVES)
