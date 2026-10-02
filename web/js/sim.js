// Ship physics, a line-for-line port of spacenav/physics.py + env._substep.
// Bodies are not integrated here: they come from the JAX ephemeris, one frame
// per physics substep, so the browser plays exactly the universe the agent saw.
// No DOM access: this module also runs under node for the parity test.

export const RUNNING = 0, ARRIVED = 1, CRASHED = 2, DESTROYED = 3, LOST = 4, TIMEOUT = 5;
export const STATUS_NAMES = ["running", "arrived", "crashed", "destroyed", "lost", "timeout"];
const EPS = 1e-6;
const KIND_STATION = 6, KIND_TRACER = 8;
const ZONE_DEBRIS = 1, ZONE_RADIATION = 2;

export class Episode {
  // With no ephemBuffer the body ephemeris is integrated here from desc.init, with
  // the same kick-drift-kick leapfrog as physics.bodies_step (float64 in the browser,
  // float32 in JAX: bodies agree to ~0.01-0.2 u over a full 120 s episode).
  constructor(desc, ephemBuffer = null) {
    this.desc = desc;
    this.K = desc.constants;
    this.bodies = desc.bodies;
    this.zones = desc.zones;
    this.task = desc.task;
    this.n = desc.ephemeris.bodies;
    this.frames = desc.ephemeris.frames;
    this.eph = ephemBuffer ? new Float32Array(ephemBuffer) : integrateBodies(desc, this.frames);
    if (this.eph.length !== this.frames * this.n * 4) throw new Error("ephemeris size mismatch");
    this.solid = this.bodies.map(b => b.radius > 0 && b.kind !== KIND_STATION && b.kind !== KIND_TRACER);
  }
  clampFrame(k) { return Math.max(0, Math.min(this.frames - 1, k)); }
  // body i at frame k -> [px, py, vx, vy]
  body(k, i) {
    const o = (this.clampFrame(k) * this.n + i) * 4;
    return [this.eph[o], this.eph[o + 1], this.eph[o + 2], this.eph[o + 3]];
  }
  // position and velocity of waypoint w at frame k
  waypoint(k, w) {
    const wp = this.task.waypoints[w];
    if (wp.anchor >= 0) {
      const b = this.body(k, wp.anchor), o = wp.offset || [0, 0];
      return [b[0] + o[0], b[1] + o[1], b[2], b[3]];
    }
    return [wp.pos[0], wp.pos[1], 0, 0];
  }
}

function integrateBodies(desc, frames) {
  const n = desc.ephemeris.bodies, h = desc.constants.PHYS_DT;
  const m = desc.bodies.map(b => b.mass), R = desc.bodies.map(b => b.radius);
  const px = new Float64Array(n), py = new Float64Array(n), vx = new Float64Array(n), vy = new Float64Array(n);
  desc.init.forEach(([x, y, u, w], i) => { px[i] = x; py[i] = y; vx[i] = u; vy[i] = w; });
  const ax = new Float64Array(n), ay = new Float64Array(n);
  const accel = () => {
    for (let i = 0; i < n; i++) {
      let gx = 0, gy = 0;
      for (let j = 0; j < n; j++) {
        if (m[j] === 0) continue;
        const dx = px[j] - px[i], dy = py[j] - py[i];
        const s = Math.max(Math.max(Math.sqrt(dx * dx + dy * dy), R[j]), EPS);
        const f = m[j] / (s * s * s);
        gx += f * dx; gy += f * dy;
      }
      ax[i] = gx; ay[i] = gy;
    }
  };
  const eph = new Float64Array(frames * n * 4);
  const store = (k) => { for (let i = 0; i < n; i++) { const o = (k * n + i) * 4; eph[o] = px[i]; eph[o + 1] = py[i]; eph[o + 2] = vx[i]; eph[o + 3] = vy[i]; } };
  store(0);
  for (let k = 1; k < frames; k++) {
    accel();
    for (let i = 0; i < n; i++) { vx[i] += 0.5 * h * ax[i]; vy[i] += 0.5 * h * ay[i]; px[i] += h * vx[i]; py[i] += h * vy[i]; }
    accel();
    for (let i = 0; i < n; i++) { vx[i] += 0.5 * h * ax[i]; vy[i] += 0.5 * h * ay[i]; }
    store(k);
  }
  return eph;
}

export function gravityAt(ep, k, x, y) {
  let ax = 0, ay = 0;
  for (let i = 0; i < ep.n; i++) {
    const b = ep.bodies[i];
    if (b.mass === 0) continue;
    const [px, py] = ep.body(k, i);
    const dx = px - x, dy = py - y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const soft = Math.max(Math.max(r, b.radius), EPS);
    const f = b.mass / (soft * soft * soft);
    ax += f * dx; ay += f * dy;
  }
  return [ax, ay];
}

export function atmosphere(ep, k, x, y, vx, vy) {
  let dx = 0, dy = 0, heat = 0;
  const K = ep.K;
  for (let i = 0; i < ep.n; i++) {
    const b = ep.bodies[i];
    if (!(b.atmo_h > 0)) continue;
    const [px, py, bvx, bvy] = ep.body(k, i);
    const r = Math.hypot(x - px, y - py);
    const alt = Math.max(r - b.radius, 0);
    if (alt >= b.atmo_h) continue;
    const scale = Math.max(b.atmo_h * K.ATMO_SCALE_FRAC, EPS);
    const rho = b.atmo_rho * Math.exp(-alt / scale);
    const rx = vx - bvx, ry = vy - bvy;
    const speed = Math.sqrt(rx * rx + ry * ry + EPS);
    dx -= K.ATMO_DRAG * rho * speed * rx;
    dy -= K.ATMO_DRAG * rho * speed * ry;
    heat += K.ATMO_HEAT * rho * speed * speed * speed;
  }
  return [dx, dy, heat];
}

export function zoneWeight(ep, k, z, x, y) {
  let cx = 0, cy = 0;
  if (z.anchor >= 0) { const b = ep.body(k, z.anchor); cx = b[0]; cy = b[1]; }
  const dx = x - cx, dy = y - cy;
  const r = Math.sqrt(dx * dx + dy * dy + EPS);
  const edge = Math.max(ep.K.ZONE_EDGE * (z.r_out - z.r_in), EPS);
  const inner = z.r_in > 0 ? Math.min(Math.max((r - z.r_in) / edge, 0), 1) : 1;
  const outer = Math.min(Math.max((z.r_out - r) / edge, 0), 1);
  return { w: inner * outer, dx, dy, r };
}

export function hazards(ep, k, x, y, vx, vy) {
  const K = ep.K;
  let radiation = 0, crashed = false;
  for (let i = 0; i < ep.n; i++) {
    const b = ep.bodies[i];
    const [px, py] = ep.body(k, i);
    const r = Math.hypot(x - px, y - py);
    if (b.lum > 0) {
      const s = Math.max(r, Math.max(b.radius, 1));
      radiation += K.RADIATION_K * b.lum / (s * s);
    }
    if (ep.solid[i] && r < b.radius) crashed = true;
  }
  const heat = atmosphere(ep, k, x, y, vx, vy)[2];
  let debris = 0, belts = 0, exposure = 0;
  for (const z of ep.zones) {
    const { w, dx, dy, r } = zoneWeight(ep, k, z, x, y);
    if (z.exposure) exposure += w;
    if (w === 0) continue;
    if (z.kind === ZONE_DEBRIS) {
      let fx = 0, fy = 0, m = 0;
      if (z.anchor >= 0) { const b = ep.body(k, z.anchor); fx = b[2]; fy = b[3]; m = ep.bodies[z.anchor].mass; }
      const circ = Math.sqrt(m / r) * z.spin;
      fx += circ * (-dy / r); fy += circ * (dx / r);
      const rel = Math.sqrt((vx - fx) ** 2 + (vy - fy) ** 2 + EPS);
      debris += K.DEBRIS_K * z.strength * rel * w;
    } else if (z.kind === ZONE_RADIATION) {
      belts += z.strength * w;
    }
  }
  exposure = Math.min(exposure, 1);
  return { damage: radiation + heat + debris + belts, exposure, crashed,
           parts: { radiation, heat, debris, belts } };
}

function shipAccel(ep, k, x, y, vx, vy, tax, tay) {
  const [gx, gy] = gravityAt(ep, k, x, y);
  const [dx, dy] = atmosphere(ep, k, x, y, vx, vy);
  return [gx + dx + tax, gy + dy + tay];
}

export class ShipSim {
  constructor(ep) { this.ep = ep; this.reset(); }

  reset() {
    const t = this.ep.task, K = this.ep.K;
    this.x = t.start_pos[0]; this.y = t.start_pos[1];
    this.vx = t.start_vel[0]; this.vy = t.start_vel[1];
    this.angle = t.start_angle;
    this.fuel = t.fuel; this.health = K.SHIP_HEALTH;
    this.visited = this.ep.task.waypoints.map(() => false);
    this.leg = 0;
    this.status = RUNNING;
    this.frame = 0;            // physics substep index == ephemeris frame
    this.tick = 0;
    this.throttle = 0; this.turn = 0;
    this.costs = { time: 0, path: 0, fuel: 0, exposure: 0, damage: 0 };
    this.lastHazard = { damage: 0, exposure: 0, parts: { radiation: 0, heat: 0, debris: 0, belts: 0 } };
    this.actions = [];
    this.trajectory = [[this.x, this.y, this.angle, 0, this.health, this.fuel]];
    this.reached = 0;
    this.history = [this._snapshot()];   // one entry per completed tick, for rewind
    this.rewinds = 0;
  }

  _snapshot() {
    return { x: this.x, y: this.y, vx: this.vx, vy: this.vy, angle: this.angle, fuel: this.fuel,
             health: this.health, status: this.status, frame: this.frame, tick: this.tick,
             costs: { ...this.costs }, lastHazard: this.lastHazard,
             visited: this.visited.slice(), leg: this.leg, reached: this.reached };
  }

  // Restore the state at the start of control tick `tick`.  Actions and the
  // trajectory are truncated so the run stays a valid action sequence from t=0.
  rewindTo(tick) {
    tick = Math.max(0, Math.min(tick, this.history.length - 1));
    const s = this.history[tick];
    Object.assign(this, { ...s, costs: { ...s.costs }, visited: s.visited.slice() });
    this.status = RUNNING;
    this.history.length = tick + 1;
    this.actions.length = tick;
    this.trajectory.length = tick + 1;
    this.throttle = 0; this.turn = 0;
  }

  get running() { return this.status === RUNNING; }

  // which waypoint the ship is flying to right now
  currentLeg(k, x, y) {
    const t = this.ep.task;
    let best = -1, bestD = Infinity;
    for (let w = 0; w < t.waypoints.length; w++) {
      if (this.visited[w]) continue;
      if (!t.order_free) return w;                       // next in sequence
      const [wx, wy] = this.ep.waypoint(k, w);
      const d = (x - wx) ** 2 + (y - wy) ** 2;
      if (d < bestD) { bestD = d; best = w; }
    }
    return best < 0 ? Math.max(t.waypoints.length - 1, 0) : best;
  }

  target(k) { return this.ep.waypoint(k, this.leg); }

  // Latch an action for the next control tick (call when frame % SUBSTEPS === 0).
  setAction(turn, throttle) {
    this.turn = Math.max(-1, Math.min(1, turn));
    this.throttle = Math.max(0, Math.min(1, throttle));
    this.actions.push([this.turn, this.throttle]);
  }

  substep() {
    if (!this.running) return;
    const ep = this.ep, K = ep.K, h = K.PHYS_DT, k = this.frame;
    const angle = this.angle + this.turn * K.SHIP_TURN_RATE * h;
    const burn = Math.min(this.throttle * h, this.fuel);
    const a = (burn / h) * this.ep.task.accel;
    const tax = a * Math.cos(angle), tay = a * Math.sin(angle);
    const [ax0, ay0] = shipAccel(ep, k, this.x, this.y, this.vx, this.vy, tax, tay);
    const hvx = this.vx + 0.5 * h * ax0, hvy = this.vy + 0.5 * h * ay0;
    const x1 = this.x + h * hvx, y1 = this.y + h * hvy;
    const [ax1, ay1] = shipAccel(ep, k + 1, x1, y1, hvx, hvy, tax, tay);
    const vx1 = hvx + 0.5 * h * ax1, vy1 = hvy + 0.5 * h * ay1;
    const hz = hazards(ep, k + 1, x1, y1, vx1, vy1);
    const health = this.health - hz.damage * h;
    const t = ep.task;
    // waypoints: in sequence, or any outstanding one when the tour is free-order
    this.leg = this.currentLeg(k + 1, x1, y1);
    for (let w = 0; w < t.waypoints.length; w++) {
      if (this.visited[w] || (!t.order_free && w !== this.leg)) continue;
      const wp = t.waypoints[w];
      const [wx, wy, wvx, wvy] = ep.waypoint(k + 1, w);
      if ((x1 - wx) ** 2 + (y1 - wy) ** 2 < wp.radius ** 2 &&
          (vx1 - wvx) ** 2 + (vy1 - wvy) ** 2 < wp.v_tol ** 2) { this.visited[w] = true; this.reached++; }
    }
    this.leg = this.currentLeg(k + 1, x1, y1);
    const arrived = this.visited.every(Boolean);
    const lost = x1 * x1 + y1 * y1 > (ep.desc.level_radius * K.OUT_OF_BOUNDS_FACTOR) ** 2;
    const c = this.costs;
    c.time += h;
    c.path += Math.hypot(x1 - this.x, y1 - this.y);
    c.fuel += burn;
    c.exposure += hz.exposure * h;
    c.damage += hz.damage * h;
    this.x = x1; this.y = y1; this.vx = vx1; this.vy = vy1; this.angle = angle;
    this.fuel -= burn;
    this.health = Math.max(health, 0);
    this.lastHazard = hz;
    this.frame += 1;
    this.status = hz.crashed ? CRASHED : health <= 0 ? DESTROYED : arrived ? ARRIVED : lost ? LOST : RUNNING;
    // a run that ends mid-tick still closes the tick so trajectories stay tick-aligned
    if (this.frame % K.SUBSTEPS === 0 || this.status !== RUNNING) this._endTick();
  }

  _endTick() {
    const K = this.ep.K;
    this.tick += 1;
    if (this.status === RUNNING && this.tick >= K.MAX_EPISODE_TICKS) this.status = TIMEOUT;
    this.trajectory.push([this.x, this.y, this.angle, this.throttle, this.health, this.fuel]);
    if (this.status === RUNNING) this.history.push(this._snapshot());
  }

  // Run one full control tick with the given action (used by tests and replays).
  stepTick(turn, throttle) {
    this.setAction(turn, throttle);
    const n = this.ep.K.SUBSTEPS;
    for (let i = 0; i < n && this.running; i++) this.substep();
  }

  objective() {
    const t = this.ep.task, c = this.costs;
    const v = [c.time, c.path, c.fuel, c.exposure];
    return v.reduce((s, x, i) => s + t.weights[i] * x / t.ref_scale[i], 0);
  }
}

// Ballistic prediction from the current state: no thrust, gravity + drag.
//
// One line is a lie in a chaotic field. `predictFan` runs a small ensemble of
// neighbouring states as well, so the display can show where the futures
// separate: while the cone is tight the prediction means something, and where it
// opens, no amount of care tells you which branch you are on. That spread is the
// honest version of a trajectory forecast in an n-body system.
// Returns {points: [[x, y, danger], ...], end: status or null}.
export function predictFan(sim, seconds, { members = 8, dv = null, stride = 6 } = {}) {
  // The perturbation is the pilot's OWN execution error, not an infinitesimal.
  // An infinitesimal nudge answers "is this formally chaotic", which is not the
  // question in the cockpit; one control tick of thrust is the finest correction
  // anybody can actually make at 15 Hz, so the cone shows where the flight may
  // end up given how precisely it can be flown. That also makes the cone
  // directly comparable to the measured human limits (~20 ms of timing jitter
  // against a 67 ms tick).
  const ep = sim.ep, K = ep.K;
  const step = dv === null ? ep.task.accel * K.CTRL_DT : dv;
  const base = predict(sim, seconds, stride);
  const arms = [];
  for (let i = 0; i < members; i++) {
    const th = (2 * Math.PI * i) / members;
    const ghost = Object.create(Object.getPrototypeOf(sim));
    Object.assign(ghost, sim, {
      vx: sim.vx + step * Math.cos(th), vy: sim.vy + step * Math.sin(th),
    });
    arms.push(predict(ghost, seconds, stride).points);
  }
  // half-width of the ensemble at each sampled time: the cone's radius
  const n = Math.min(base.points.length, ...arms.map((a) => a.length));
  const width = [];
  for (let j = 0; j < n; j++) {
    const [bx, by] = base.points[j];
    let m = 0;
    for (const a of arms) m = Math.max(m, Math.hypot(a[j][0] - bx, a[j][1] - by));
    width.push(m);
  }
  // Prediction has expired once the spread exceeds the target you are aiming at:
  // beyond that point the forecast cannot tell you whether you arrive. Measured
  // against the CURRENT waypoint's radius, not a global constant, because that
  // is the tolerance that actually has to be met.
  const wp = (ep.task.waypoints || [])[sim.leg] || {};
  const tol = wp.radius || K.TARGET_RADIUS;
  let horizon = n - 1;
  for (let j = 0; j < n; j++) if (width[j] > tol) { horizon = j; break; }
  // predict() stores pts[0] at t=0 and pts[j>=1] after ((j-1)*stride + 1) steps
  const tOf = (j) => (j < 1 ? 0 : ((j - 1) * stride + 1) * K.PHYS_DT);
  return { ...base, arms, width, horizon, horizonSeconds: tOf(horizon),
           expired: horizon < n - 1, tol, spread: step, stride };
}

export function predict(sim, seconds, stride = 4) {
  const ep = sim.ep, K = ep.K, h = K.PHYS_DT;
  let x = sim.x, y = sim.y, vx = sim.vx, vy = sim.vy, k = sim.frame;
  const steps = Math.min(Math.round(seconds / h), ep.frames - 1 - k);
  const pts = [[x, y, 0]];
  let end = null, closest = Infinity;
  for (let s = 0; s < steps; s++) {
    const [ax0, ay0] = shipAccel(ep, k, x, y, vx, vy, 0, 0);
    const hvx = vx + 0.5 * h * ax0, hvy = vy + 0.5 * h * ay0;
    x += h * hvx; y += h * hvy;
    const [ax1, ay1] = shipAccel(ep, k + 1, x, y, hvx, hvy, 0, 0);
    vx = hvx + 0.5 * h * ax1; vy = hvy + 0.5 * h * ay1;
    k += 1;
    const [tx, ty] = ep.waypoint(k, sim.leg);
    closest = Math.min(closest, Math.hypot(x - tx, y - ty));
    if (s % stride === 0 || s === steps - 1) {
      const hz = hazards(ep, k, x, y, vx, vy);
      pts.push([x, y, hz.damage]);
      if (hz.crashed) { end = CRASHED; break; }
    }
  }
  return { points: pts, end, closest };
}

// ---------------------------------------------------------------------------
// instruments: a line-for-line port of spacenav/env.py:instrument_panel
// ---------------------------------------------------------------------------
//
//   eta      the share of the local field that is *not* the single dominant
//            body.  Near 0 a one-body mental model is accurate; near 1 it is
//            actively misleading.
//   A        the tank measured in units of the impulse gravity will deliver
//            over the rest of the flight.  Below 1 you cannot power through.
//   D        eta / A: the fraction of the tank that flying the wrong model
//            would cost.
//   stretch  sqrt of the tidal gradient: a local divergence rate (1/s), so how
//            soon prediction stops being worth anything.
//
// The Python side feeds the agent `[eta, symlog(A), symlog(D), stretch*10]`;
// `panel` below is that same vector, so a parity test can compare directly,
// while the HUD reads the raw numbers the lessons are named after.

function symlog(x) { return Math.sign(x) * Math.log1p(Math.abs(x)); }

// The field the single strongest attractor would produce on its own --
// physics.gravity_at(..., topk=1), the "patched conics" model a person reasons
// with.  Ties are kept, exactly as jax.lax.top_k + (pull >= cut) does.
export function dominantGravity(ep, k, x, y) {
  let cut = 0;                       // pull is never negative, and all-zero
  for (let i = 0; i < ep.n; i++) {   // masses must leave the field at zero
    const b = ep.bodies[i];
    if (b.mass === 0) continue;
    const [px, py] = ep.body(k, i);
    const dx = px - x, dy = py - y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const soft = Math.max(Math.max(r, b.radius), EPS);
    const pull = (b.mass / (soft * soft * soft)) * r;     // |a| from this body
    if (pull > cut) cut = pull;
  }
  let ax = 0, ay = 0;
  for (let i = 0; i < ep.n; i++) {
    const b = ep.bodies[i];
    if (b.mass === 0) continue;
    const [px, py] = ep.body(k, i);
    const dx = px - x, dy = py - y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const soft = Math.max(Math.max(r, b.radius), EPS);
    const w = b.mass / (soft * soft * soft);
    if (w * r < cut) continue;       // discarded by the top-k cut
    ax += w * dx; ay += w * dy;
  }
  return [ax, ay];
}

// `horizonSeconds` is normally handed in from predictFan (the measured spread).
// With nothing passed, `stretchHorizon` is used instead: the analytic time for
// a nudge of the fan's own size to grow to the arrival radius, capped by the
// time left in the episode.
export function instruments(sim, { horizonSeconds = null, spread = 0.05 } = {}) {
  const ep = sim.ep, K = ep.K, k = sim.frame;
  const [gx, gy] = gravityAt(ep, k, sim.x, sim.y);
  const [dgx, dgy] = dominantGravity(ep, k, sim.x, sim.y);
  const gmag = Math.sqrt(gx * gx + gy * gy + 1e-12);
  const eta = Math.sqrt((gx - dgx) ** 2 + (gy - dgy) ** 2 + 1e-18) / gmag;

  const tLeft = Math.max(K.MAX_EPISODE_TIME - k * K.PHYS_DT, K.CTRL_DT);
  const A = (sim.fuel * ep.task.accel) / Math.max(gmag * tLeft, 1e-6);
  const D = eta / Math.max(A, 1e-6);

  // tidal gradient: sum of 2*GM/r^3 over the bodies that matter.  Note the
  // softening here is max(r, max(radius, 1)), not the field's max(r, radius).
  let tidal = 0;
  for (let i = 0; i < ep.n; i++) {
    const b = ep.bodies[i];
    if (b.mass === 0) continue;
    const [px, py] = ep.body(k, i);
    const dx = px - sim.x, dy = py - sim.y;
    const r = Math.max(Math.sqrt(dx * dx + dy * dy), Math.max(b.radius, 1.0));
    tidal += 2.0 * b.mass / (r * r * r);
  }
  const stretch = Math.sqrt(tidal);

  const grow = Math.log(Math.max(K.TARGET_RADIUS / Math.max(spread, 1e-9), Math.E));
  const stretchHorizon = Math.min(stretch > 1e-9 ? grow / stretch : Infinity, tLeft);
  return {
    eta, A, D, stretch, stretchHorizon,
    horizonSeconds: horizonSeconds == null ? stretchHorizon : horizonSeconds,
    panel: [eta, symlog(A), symlog(D), stretch * 10.0],
  };
}
