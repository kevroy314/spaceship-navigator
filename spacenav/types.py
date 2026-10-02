"""Pytrees shared by the simulator, environment and exporters.

All arrays are fixed-size (padded to MAX_BODIES / MAX_ZONES with `active`
masks) so a batch of wildly different levels can be vmapped without
recompilation.
"""

from typing import NamedTuple

import jax.numpy as jnp


class Level(NamedTuple):
    """Everything static about a level, plus snapshots of its body state.

    Body "mass" is GM (G = 1).  Massless bodies (stations, tracers) are carried
    along by gravity but exert none.
    """
    mass: jnp.ndarray        # (N,)
    radius: jnp.ndarray      # (N,)   collision radius
    atmo_h: jnp.ndarray      # (N,)   atmosphere thickness above the surface (0 = none)
    atmo_rho: jnp.ndarray    # (N,)   atmosphere density at the surface
    lum: jnp.ndarray         # (N,)   radiation luminosity
    kind: jnp.ndarray        # (N,)   int, see constants.KIND_*
    parent: jnp.ndarray      # (N,)   int, index of the body it orbits (-1 = root)
    active: jnp.ndarray      # (N,)   bool
    zone_anchor: jnp.ndarray   # (Z,) int body index the zone rides with (-1 = origin)
    zone_r_in: jnp.ndarray     # (Z,)
    zone_r_out: jnp.ndarray    # (Z,)
    zone_kind: jnp.ndarray     # (Z,) int, see constants.ZONE_*
    zone_strength: jnp.ndarray # (Z,)
    zone_spin: jnp.ndarray     # (Z,) +1/-1: debris circulates around the anchor; 0: moves with it
    zone_active: jnp.ndarray   # (Z,) bool
    snap_pos: jnp.ndarray    # (S, N, 2) body positions at t = k * SNAPSHOT_INTERVAL
    snap_vel: jnp.ndarray    # (S, N, 2)
    level_radius: jnp.ndarray  # ()  play-area radius around the origin (barycentre)
    family: jnp.ndarray        # ()  int family id


class Task(NamedTuple):
    """One mission: a tour of waypoints, flown with a given ship loadout.

    A single-target mission is a tour of length one.  Waypoint arrays are padded
    to MAX_WAYPOINTS with `wp_active`.
    """
    snapshot: jnp.ndarray      # ()  int, which level snapshot the episode starts from
    start_pos: jnp.ndarray     # (2,)
    start_vel: jnp.ndarray     # (2,)
    start_angle: jnp.ndarray   # ()
    wp_anchor: jnp.ndarray     # (K,) int body index for a moving waypoint, -1 for a fixed point
    wp_pos: jnp.ndarray        # (K, 2) fixed waypoint positions (ignored when anchored)
    wp_offset: jnp.ndarray     # (K, 2) offset from the anchor body, so a waypoint can ride
                               #        a planet or moon without sitting inside it
    wp_radius: jnp.ndarray     # (K,)
    wp_v_tol: jnp.ndarray      # (K,) max relative speed on arrival (huge = fly-through)
    wp_active: jnp.ndarray     # (K,) bool
    order_free: jnp.ndarray    # ()  bool: visit in any order, else in sequence
    accel: jnp.ndarray         # ()  u/s^2 at full throttle
    fuel: jnp.ndarray          # ()  seconds of full-throttle burn carried
    weights: jnp.ndarray       # (N_OBJECTIVES,) time, path, fuel, exposure
    exposure_mask: jnp.ndarray # (Z,) bool, zones that count as "exposure"
    ref_scale: jnp.ndarray     # (N_OBJECTIVES,) normalisers for each cost term


class Ship(NamedTuple):
    pos: jnp.ndarray     # (2,)
    vel: jnp.ndarray     # (2,)
    angle: jnp.ndarray   # ()
    fuel: jnp.ndarray    # ()  seconds of full-throttle burn left
    health: jnp.ndarray  # ()


class Costs(NamedTuple):
    """Running totals of every quantity an objective can care about."""
    time: jnp.ndarray
    path: jnp.ndarray
    fuel: jnp.ndarray
    exposure: jnp.ndarray
    damage: jnp.ndarray

    def vector(self):
        return jnp.stack([self.time, self.path, self.fuel, self.exposure])


# termination reasons
RUNNING, ARRIVED, CRASHED, DESTROYED, LOST, TIMEOUT = 0, 1, 2, 3, 4, 5
REASONS = {RUNNING: "running", ARRIVED: "arrived", CRASHED: "crashed",
           DESTROYED: "destroyed", LOST: "lost", TIMEOUT: "timeout"}


class EnvState(NamedTuple):
    t: jnp.ndarray          # () game seconds since episode start
    tick: jnp.ndarray       # () control ticks since episode start
    body_pos: jnp.ndarray   # (N, 2)
    body_vel: jnp.ndarray   # (N, 2)
    ship: Ship
    costs: Costs
    status: jnp.ndarray     # () int, RUNNING or a termination reason
    thrust: jnp.ndarray     # () last applied throttle (for rendering)
    visited: jnp.ndarray    # (K,) bool, waypoints reached so far
    leg: jnp.ndarray        # () int, index of the waypoint being flown to
