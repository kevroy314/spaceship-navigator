// Canvas2D renderer.  World y points up; screen y points down.

const KIND = { STAR: 1, ROCKY: 2, GIANT: 3, MOON: 4, BH: 5, STATION: 6, ASTEROID: 7, TRACER: 8, NEUTRON: 9 };
const ZONE = { DEBRIS: 1, RADIATION: 2, SENSOR: 3 };
export const OVERLAY_COLORS = ["#5dd6ff", "#ff7ad9", "#ffd166", "#9dff5d", "#b18cff", "#ff9f5d", "#5dffd6", "#ff5d6c"];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GIANT_COLORS = [["#d9a066", "#b9804a"], ["#9fb7d9", "#7f97c0"], ["#e0c080", "#c0a060"], ["#b0a0d0", "#9080b0"], ["#d99a80", "#b87a60"]];
const ROCKY_COLORS = ["#8a7f73", "#6f8fa6", "#a5704f", "#7d9a6a", "#9c8f86"];

export class Renderer {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.cam = { x: 0, y: 0, s: 0.5 };
    this.follow = false;
    this.showPrediction = true;
    this.showFan = true;       // the predictability cone, once the level grants it
    // screen area covered by UI, so framing and indicators stay in the clear part
    this.insets = { left: 300, right: 300, top: 20, bottom: 20 };
    this.scaleAtTop = false;   // mobile: bottom corners belong to the touch pads
    this.resize();
    window.addEventListener("resize", () => this.resize());
    const rng = mulberry32(7);
    this.stars = Array.from({ length: 700 }, () => [rng(), rng(), rng() ** 3, rng()]);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.dpr = dpr;
    this.W = window.innerWidth; this.H = window.innerHeight;
    this.cv.width = this.W * dpr; this.cv.height = this.H * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  sx(x) { return this.W / 2 + (x - this.cam.x) * this.cam.s; }
  sy(y) { return this.H / 2 - (y - this.cam.y) * this.cam.s; }
  toWorld(px, py) { return [(px - this.W / 2) / this.cam.s + this.cam.x, -(py - this.H / 2) / this.cam.s + this.cam.y]; }

  zoomAt(px, py, factor) {
    const [wx, wy] = this.toWorld(px, py);
    this.cam.s = Math.min(40, Math.max(0.02, this.cam.s * factor));
    const [nx, ny] = this.toWorld(px, py);
    this.cam.x += wx - nx; this.cam.y += wy - ny;
  }

  fit(points, margin = 1.3) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of points) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const ins = this.insets;
    const aw = Math.max(this.W - ins.left - ins.right, 120), ah = Math.max(this.H - ins.top - ins.bottom, 120);
    const w = Math.max(x1 - x0, 50) * margin, h = Math.max(y1 - y0, 50) * margin;
    this.cam.s = Math.max(Math.min(aw / w, ah / h, 8), 0.02);
    // put the box centre at the centre of the unobstructed area
    const cxs = ins.left + aw / 2, cys = ins.top + ah / 2;
    this.cam.x = (x0 + x1) / 2 - (cxs - this.W / 2) / this.cam.s;
    this.cam.y = (y0 + y1) / 2 + (cys - this.H / 2) / this.cam.s;
  }

  setEpisode(ep) {
    this.ep = ep;
    const rng = mulberry32(ep.desc.index * 7919 + 13);
    this.bodyStyle = ep.bodies.map((b, i) => {
      const r = rng();
      if (b.kind === KIND.GIANT) return { colors: GIANT_COLORS[Math.floor(r * GIANT_COLORS.length)], bands: 3 + Math.floor(rng() * 4), tilt: rng() * 0.6 - 0.3 };
      if (b.kind === KIND.ROCKY) return { color: ROCKY_COLORS[Math.floor(r * ROCKY_COLORS.length)] };
      if (b.kind === KIND.MOON) { const g = 130 + Math.floor(r * 70); return { color: `rgb(${g},${g - 5},${g - 12})` }; }
      if (b.kind === KIND.ASTEROID) return { poly: Array.from({ length: 9 }, () => 0.7 + rng() * 0.5) };
      if (b.kind === KIND.STAR) {
        const hot = Math.min(1, b.mass / 8e4);
        return { core: "#fffbe8", mid: hot > 0.6 ? "#ffe9a8" : "#ffc861", edge: hot > 0.6 ? "rgba(255,220,150,0)" : "rgba(255,150,60,0)" };
      }
      return {};
    });
    // debris particles, stored in the zone's local frame
    this.zoneParticles = ep.zones.map((z) => {
      const area = Math.PI * (z.r_out ** 2 - z.r_in ** 2);
      const n = z.kind === ZONE.DEBRIS ? Math.min(900, Math.floor(area * 0.02 * Math.max(z.strength, 0.3)) + 40) : 0;
      return Array.from({ length: n }, () => {
        const r = Math.sqrt(z.r_in ** 2 + rng() * (z.r_out ** 2 - z.r_in ** 2));
        return [r, rng() * Math.PI * 2, 0.6 + rng() * 1.1, 0.25 + rng() * 0.55];
      });
    });
    // orbit paths relative to the parent body
    const stride = 12;
    this.orbits = ep.bodies.map((b, i) => {
      if (b.kind === KIND.TRACER) return null;
      const pts = [];
      for (let k = 0; k < ep.frames; k += stride) {
        const p = ep.body(k, i);
        if (b.parent >= 0) { const q = ep.body(k, b.parent); pts.push([p[0] - q[0], p[1] - q[1]]); }
        else pts.push([p[0], p[1]]);
      }
      return pts;
    });
  }

  clear() {
    const c = this.ctx;
    c.fillStyle = "#04060c";
    c.fillRect(0, 0, this.W, this.H);
    // parallax starfield
    for (const [u, v, b, tw] of this.stars) {
      const px = ((u * this.W * 1.5 - this.cam.x * this.cam.s * 0.03 * (0.3 + b)) % this.W + this.W) % this.W;
      const py = ((v * this.H * 1.5 + this.cam.y * this.cam.s * 0.03 * (0.3 + b)) % this.H + this.H) % this.H;
      c.fillStyle = `rgba(200,215,255,${0.15 + 0.6 * b})`;
      c.fillRect(px, py, b > 0.6 ? 1.6 : 1, b > 0.6 ? 1.6 : 1);
    }
  }

  draw(scene) {
    const { frame, sim, overlays, replayTick, prediction, time } = scene;
    const ep = this.ep;
    if (!ep) return;
    this.clear();
    const c = this.ctx;
    const s = this.cam.s;

    // play-area boundary
    const R = ep.desc.level_radius * ep.K.OUT_OF_BOUNDS_FACTOR;
    c.strokeStyle = "rgba(255,93,108,0.18)"; c.setLineDash([6, 8]); c.lineWidth = 1;
    c.beginPath(); c.arc(this.sx(0), this.sy(0), R * s, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);

    // zones
    ep.zones.forEach((z, zi) => this.drawZone(z, zi, frame, time));

    // orbits
    c.lineWidth = 1;
    ep.bodies.forEach((b, i) => {
      const pts = this.orbits[i];
      if (!pts) return;
      let ox = 0, oy = 0;
      if (b.parent >= 0) { const q = ep.body(frame, b.parent); ox = q[0]; oy = q[1]; }
      c.strokeStyle = b.kind === KIND.STATION ? "rgba(93,214,255,0.12)" : "rgba(150,170,210,0.13)";
      c.beginPath();
      pts.forEach(([x, y], j) => { const X = this.sx(x + ox), Y = this.sy(y + oy); j ? c.lineTo(X, Y) : c.moveTo(X, Y); });
      c.stroke();
    });

    // radiation danger rings around luminous bodies (1/s and 5/s)
    ep.bodies.forEach((b, i) => {
      if (!(b.lum > 0)) return;
      const [x, y] = ep.body(frame, i);
      for (const [rate, a] of [[1, 0.22], [5, 0.35]]) {
        const r = Math.sqrt(ep.K.RADIATION_K * b.lum / rate);
        if (r <= b.radius) continue;
        c.strokeStyle = `rgba(255,110,70,${a})`; c.setLineDash([2, 6]);
        c.beginPath(); c.arc(this.sx(x), this.sy(y), r * s, 0, Math.PI * 2); c.stroke();
      }
      c.setLineDash([]);
    });

    // bodies
    ep.bodies.forEach((b, i) => this.drawBody(b, i, frame, time));

    // start and target
    this.drawStart();
    this.drawTarget(frame, time, sim);

    // overlays (recorded runs)
    for (const ov of overlays) this.drawOverlay(ov, replayTick);

    // prediction
    if (prediction && this.showPrediction) this.drawPrediction(prediction);

    // live ship
    if (sim && scene.showLive) {
      this.drawTrail(sim.trajectory, "rgba(255,255,255,0.35)");
      this.drawShip(sim.x, sim.y, sim.angle, sim.running ? sim.throttle : 0, "#ffffff", time);
    }
    this.drawTargetArrow(frame, sim);
    this.drawScale();
  }

  drawZone(z, zi, frame, time) {
    const ep = this.ep, c = this.ctx, s = this.cam.s;
    let cx = 0, cy = 0, gm = 0;
    if (z.anchor >= 0) { const b = ep.body(frame, z.anchor); cx = b[0]; cy = b[1]; gm = ep.bodies[z.anchor].mass; }
    const X = this.sx(cx), Y = this.sy(cy);
    const expoOn = z.exposure && ep.task.weights[3] > 0;
    const ring = (fill) => {
      c.beginPath();
      c.arc(X, Y, z.r_out * s, 0, Math.PI * 2);
      if (z.r_in > 0) c.arc(X, Y, z.r_in * s, 0, Math.PI * 2, true);
      c.fillStyle = fill; c.fill("evenodd");
    };
    if (z.kind === ZONE.DEBRIS) {
      ring(`rgba(170,140,105,${0.05 + 0.03 * Math.min(z.strength, 2)})`);
      const t = frame * ep.K.PHYS_DT;
      c.fillStyle = "rgba(215,190,150,0.8)";
      for (const [r, th0, size, alpha] of this.zoneParticles[zi]) {
        const om = z.spin !== 0 && gm > 0 ? z.spin * Math.sqrt(gm / r) / r : 0;
        const th = th0 + om * t;
        const px = X + Math.cos(th) * r * s, py = Y - Math.sin(th) * r * s;
        if (px < -5 || py < -5 || px > this.W + 5 || py > this.H + 5) continue;
        c.globalAlpha = alpha * Math.min(1, 0.4 + 0.3 * z.strength);
        const sz = Math.max(0.8, size * Math.min(s, 1.5));
        c.fillRect(px, py, sz, sz);
      }
      c.globalAlpha = 1;
    } else if (z.kind === ZONE.RADIATION) {
      const g = c.createRadialGradient(X, Y, Math.max(z.r_in * s, 0), X, Y, z.r_out * s);
      const a = Math.min(0.28, 0.05 + 0.03 * z.strength);
      g.addColorStop(0, "rgba(190,70,255,0)"); g.addColorStop(0.5, `rgba(190,70,255,${a})`); g.addColorStop(1, "rgba(190,70,255,0)");
      ring(g);
      c.strokeStyle = "rgba(210,120,255,0.3)"; c.setLineDash([3, 5]);
      for (const r of [z.r_in, z.r_out]) if (r > 0) { c.beginPath(); c.arc(X, Y, r * s, 0, Math.PI * 2); c.stroke(); }
      c.setLineDash([]);
    } else if (z.kind === ZONE.SENSOR) {
      ring(expoOn ? "rgba(255,210,70,0.09)" : "rgba(255,90,90,0.035)");
      c.strokeStyle = expoOn ? "rgba(255,210,70,0.75)" : "rgba(255,90,90,0.35)";
      c.lineWidth = expoOn ? 1.5 : 1; c.setLineDash([8, 6]);
      c.beginPath(); c.arc(X, Y, z.r_out * s, 0, Math.PI * 2); c.stroke(); c.setLineDash([]); c.lineWidth = 1;
    }
    if (expoOn) {
      c.fillStyle = "rgba(255,220,90,0.9)"; c.font = "10px var(--font), monospace";
      c.fillText("AVOID (exposure)", X + z.r_out * s * 0.72, Y - z.r_out * s * 0.72);
    }
  }

  drawBody(b, i, frame, time) {
    const ep = this.ep, c = this.ctx, s = this.cam.s;
    if (b.kind === KIND.TRACER) return;
    const [x, y] = ep.body(frame, i);
    const X = this.sx(x), Y = this.sy(y);
    const R = Math.max(b.radius * s, b.kind === KIND.STATION ? 0 : 2);
    if (X < -R * 6 - 50 || Y < -R * 6 - 50 || X > this.W + R * 6 + 50 || Y > this.H + R * 6 + 50) return;
    const st = this.bodyStyle[i];

    if (b.atmo_h > 0) {
      const g = c.createRadialGradient(X, Y, R, X, Y, (b.radius + b.atmo_h) * s + 1);
      g.addColorStop(0, "rgba(130,190,255,0.45)"); g.addColorStop(1, "rgba(130,190,255,0)");
      c.fillStyle = g; c.beginPath(); c.arc(X, Y, (b.radius + b.atmo_h) * s + 1, 0, Math.PI * 2); c.fill();
    }
    switch (b.kind) {
      case KIND.STAR: {
        const g = c.createRadialGradient(X, Y, 0, X, Y, R * 4.5);
        g.addColorStop(0, st.core); g.addColorStop(0.2, st.mid); g.addColorStop(1, st.edge);
        c.fillStyle = g; c.beginPath(); c.arc(X, Y, R * 4.5, 0, Math.PI * 2); c.fill();
        c.fillStyle = st.core; c.beginPath(); c.arc(X, Y, R, 0, Math.PI * 2); c.fill();
        break;
      }
      case KIND.BH: {
        const g = c.createRadialGradient(X, Y, R, X, Y, R * 5);
        g.addColorStop(0, "rgba(255,200,140,0.55)"); g.addColorStop(0.3, "rgba(255,140,60,0.15)"); g.addColorStop(1, "rgba(255,140,60,0)");
        c.fillStyle = g; c.beginPath(); c.arc(X, Y, R * 5, 0, Math.PI * 2); c.fill();
        c.fillStyle = "#000"; c.beginPath(); c.arc(X, Y, R, 0, Math.PI * 2); c.fill();
        c.strokeStyle = "rgba(255,220,170,0.9)"; c.lineWidth = Math.max(1, R * 0.12);
        c.beginPath(); c.arc(X, Y, R * 1.15, 0, Math.PI * 2); c.stroke(); c.lineWidth = 1;
        break;
      }
      case KIND.NEUTRON: {
        const g = c.createRadialGradient(X, Y, 0, X, Y, R * 8);
        g.addColorStop(0, "rgba(220,240,255,1)"); g.addColorStop(0.15, "rgba(140,190,255,0.7)"); g.addColorStop(1, "rgba(100,150,255,0)");
        c.fillStyle = g; c.beginPath(); c.arc(X, Y, R * 8, 0, Math.PI * 2); c.fill();
        const a = time * 2.2;
        c.fillStyle = "rgba(170,210,255,0.18)";
        for (const off of [0, Math.PI]) {
          c.beginPath(); c.moveTo(X, Y);
          c.arc(X, Y, R * 60, a + off - 0.05, a + off + 0.05); c.closePath(); c.fill();
        }
        break;
      }
      case KIND.GIANT: {
        c.save(); c.beginPath(); c.arc(X, Y, R, 0, Math.PI * 2); c.clip();
        c.fillStyle = st.colors[0]; c.fillRect(X - R, Y - R, 2 * R, 2 * R);
        c.translate(X, Y); c.rotate(st.tilt);
        c.fillStyle = st.colors[1];
        for (let k = 0; k < st.bands; k++) {
          const yy = -R + (2 * R) * (k + 0.5) / st.bands;
          c.fillRect(-R, yy - R * 0.08, 2 * R, R * 0.16 * (1 + (k % 2)));
        }
        c.restore();
        this.shade(X, Y, R);
        break;
      }
      case KIND.ROCKY: case KIND.MOON: {
        c.fillStyle = st.color; c.beginPath(); c.arc(X, Y, R, 0, Math.PI * 2); c.fill();
        this.shade(X, Y, R);
        break;
      }
      case KIND.ASTEROID: {
        c.fillStyle = "#8d857b"; c.beginPath();
        st.poly.forEach((f, k) => { const a = (k / st.poly.length) * Math.PI * 2; const px = X + Math.cos(a) * R * f, py = Y + Math.sin(a) * R * f; k ? c.lineTo(px, py) : c.moveTo(px, py); });
        c.closePath(); c.fill();
        break;
      }
      case KIND.STATION: {
        const sz = Math.max(4, 3 * s);
        c.save(); c.translate(X, Y); c.rotate(time * 0.8);
        c.strokeStyle = "#5dd6ff"; c.lineWidth = 1.5;
        c.strokeRect(-sz / 2, -sz / 2, sz, sz);
        c.beginPath(); c.moveTo(-sz, 0); c.lineTo(sz, 0); c.stroke();
        c.restore(); c.lineWidth = 1;
        if (Math.floor(time * 2) % 2 === 0) { c.fillStyle = "#ff5d6c"; c.fillRect(X - 1, Y - 1, 2, 2); }
        break;
      }
    }
  }

  shade(X, Y, R) {
    const c = this.ctx;
    const g = c.createRadialGradient(X - R * 0.4, Y - R * 0.4, R * 0.1, X, Y, R);
    g.addColorStop(0, "rgba(255,255,255,0.12)"); g.addColorStop(1, "rgba(0,0,0,0.45)");
    c.fillStyle = g; c.beginPath(); c.arc(X, Y, R, 0, Math.PI * 2); c.fill();
  }

  drawStart() {
    const t = this.ep.task, c = this.ctx;
    const X = this.sx(t.start_pos[0]), Y = this.sy(t.start_pos[1]);
    c.strokeStyle = "rgba(220,230,255,0.6)"; c.beginPath(); c.arc(X, Y, 6, 0, Math.PI * 2); c.stroke();
    c.fillStyle = "rgba(220,230,255,0.8)"; c.font = "11px monospace"; c.fillText("A", X + 8, Y - 8);
  }

  drawTarget(frame, time, sim) {
    const ep = this.ep, c = this.ctx, s = this.cam.s;
    const wps = ep.task.waypoints;
    const leg = sim ? sim.leg : 0;
    const pulse = 0.55 + 0.45 * Math.sin(time * 4);
    wps.forEach((wp, w) => {
      const done = sim ? sim.visited[w] : false;
      const cur = w === leg && !done;
      const [tx, ty] = ep.waypoint(frame, w);
      const X = this.sx(tx), Y = this.sy(ty);
      const r = Math.max(wp.radius * s, 7);
      c.strokeStyle = done ? "rgba(93,255,157,0.28)"
        : cur ? `rgba(93,255,157,${0.5 + 0.5 * pulse})` : "rgba(93,255,157,0.45)";
      c.lineWidth = cur ? 1.8 : 1.1;
      c.beginPath(); c.arc(X, Y, r, 0, Math.PI * 2); c.stroke();
      if (cur) {
        c.beginPath(); c.arc(X, Y, r + 4 + 3 * pulse, 0, Math.PI * 2);
        c.globalAlpha = 0.35; c.stroke(); c.globalAlpha = 1;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          c.beginPath(); c.moveTo(X + dx * (r + 2), Y + dy * (r + 2));
          c.lineTo(X + dx * (r + 9), Y + dy * (r + 9)); c.stroke();
        }
      }
      c.lineWidth = 1;
      c.fillStyle = done ? "rgba(93,255,157,0.45)" : "#5dff9d";
      c.font = "11px monospace";
      const label = wps.length > 1 ? `${w + 1}` : "B";
      c.fillText(done ? `${label} ✓` : (wp.rendezvous ? `${label} dock ≤${ep.K.RENDEZVOUS_V_TOL}` : label),
                 X + r + 8, Y - r - 4);
    });
    // thin line along the planned order, so a tour reads as a route
    if (wps.length > 1 && !ep.task.order_free) {
      c.strokeStyle = "rgba(93,255,157,0.22)"; c.setLineDash([3, 5]); c.beginPath();
      c.moveTo(this.sx(ep.task.start_pos[0]), this.sy(ep.task.start_pos[1]));
      wps.forEach((wp, w) => { const [x, y] = ep.waypoint(frame, w); c.lineTo(this.sx(x), this.sy(y)); });
      c.stroke(); c.setLineDash([]);
    }
  }

  drawShip(x, y, angle, throttle, color, time, scale = 1) {
    const c = this.ctx;
    const X = this.sx(x), Y = this.sy(y);
    c.save(); c.translate(X, Y); c.rotate(-angle);
    const L = 11 * scale;
    if (throttle > 0.01) {
      const fl = L * (0.6 + 0.9 * throttle) * (0.85 + 0.3 * Math.random());
      c.fillStyle = "rgba(255,170,60,0.9)";
      c.beginPath(); c.moveTo(-L * 0.55, -L * 0.28); c.lineTo(-L * 0.55 - fl, 0); c.lineTo(-L * 0.55, L * 0.28); c.closePath(); c.fill();
    }
    c.fillStyle = color; c.strokeStyle = "rgba(0,0,0,0.6)";
    c.beginPath(); c.moveTo(L, 0); c.lineTo(-L * 0.6, -L * 0.5); c.lineTo(-L * 0.35, 0); c.lineTo(-L * 0.6, L * 0.5); c.closePath();
    c.fill(); c.stroke();
    c.restore();
  }

  drawTrail(traj, color, upto = Infinity) {
    const c = this.ctx;
    const n = Math.min(traj.length, upto + 1);
    if (n < 2) return;
    c.strokeStyle = color; c.lineWidth = 1.5; c.beginPath();
    for (let i = 0; i < n; i++) { const X = this.sx(traj[i][0]), Y = this.sy(traj[i][1]); i ? c.lineTo(X, Y) : c.moveTo(X, Y); }
    c.stroke(); c.lineWidth = 1;
  }

  drawOverlay(ov, tick) {
    if (!ov.visible) return;
    const traj = ov.trajectory;
    const c = this.ctx;
    c.globalAlpha = 0.35; this.drawTrail(traj, ov.color); c.globalAlpha = 1;
    const i = Math.min(Math.floor(tick), traj.length - 1);
    this.drawTrail(traj, ov.color, i);
    const p = traj[i];
    this.drawShip(p[0], p[1], p[2], p[3], ov.color, 0, 0.9);
    const end = traj[traj.length - 1];
    if (ov.status !== "arrived" && i === traj.length - 1) {
      const X = this.sx(end[0]), Y = this.sy(end[1]);
      c.strokeStyle = ov.color; c.lineWidth = 2;
      c.beginPath(); c.moveTo(X - 5, Y - 5); c.lineTo(X + 5, Y + 5); c.moveTo(X + 5, Y - 5); c.lineTo(X - 5, Y + 5); c.stroke();
      c.lineWidth = 1;
    }
  }

  // The predictability cone: the ensemble's spread, drawn behind the predicted
  // line.  Up to the horizon the forecast means something, so the cone is cool
  // and tight; past it the ensemble has swollen wider than the arrival radius,
  // and it is drawn amber, open and dashed to say the forecast has expired.
  //
  // Degrades to nothing if the caller handed over a single-line `predict()`
  // result (no arms / no width), or if the level has not earned the cone.
  drawFan(pred) {
    if (!this.showFan || !pred || !pred.arms || !pred.width || pred.width.length < 2) return;
    const c = this.ctx, pts = pred.points, s = this.cam.s;
    const n = Math.min(pred.width.length, pts.length);
    const h = Math.max(1, Math.min(pred.horizon == null ? n - 1 : pred.horizon, n - 1));

    // envelope: the base line offset either side by the ensemble's half-width
    const up = [], dn = [];
    for (let j = 0; j < n; j++) {
      const a = pts[Math.max(j - 1, 0)], b = pts[Math.min(j + 1, n - 1)];
      let tx = b[0] - a[0], ty = b[1] - a[1];
      const L = Math.hypot(tx, ty);
      if (L < 1e-9) { tx = 1; ty = 0; } else { tx /= L; ty /= L; }
      const w = pred.width[j];
      up.push([this.sx(pts[j][0] - ty * w), this.sy(pts[j][1] + tx * w)]);
      dn.push([this.sx(pts[j][0] + ty * w), this.sy(pts[j][1] - tx * w)]);
    }
    const band = (i0, i1, fill) => {
      if (i1 <= i0) return;
      c.beginPath();
      c.moveTo(up[i0][0], up[i0][1]);
      for (let j = i0 + 1; j <= i1; j++) c.lineTo(up[j][0], up[j][1]);
      for (let j = i1; j >= i0; j--) c.lineTo(dn[j][0], dn[j][1]);
      c.closePath(); c.fillStyle = fill; c.fill();
    };

    c.save();
    band(0, h, "rgba(93,170,255,0.16)");          // still worth something
    band(h, n - 1, "rgba(255,179,71,0.20)");      // expired: only "somewhere in here"

    // the individual futures, so the fill reads as an ensemble and not a smear
    c.beginPath();
    for (const arm of pred.arms) {
      const m = Math.min(arm.length, n);
      if (m < 2) continue;
      c.moveTo(this.sx(arm[0][0]), this.sy(arm[0][1]));
      for (let j = 1; j < m; j++) c.lineTo(this.sx(arm[j][0]), this.sy(arm[j][1]));
    }
    c.strokeStyle = "rgba(170,205,255,0.34)";
    c.lineWidth = 1;
    c.stroke();

    if (h < n - 1) {
      // the opening edges, dashed, so the flare is legible over a starfield
      c.beginPath();
      for (const edge of [up, dn]) {
        c.moveTo(edge[h][0], edge[h][1]);
        for (let j = h + 1; j < n; j++) c.lineTo(edge[j][0], edge[j][1]);
      }
      c.strokeStyle = "rgba(255,190,100,0.55)"; c.setLineDash([3, 4]); c.stroke(); c.setLineDash([]);

      // and the marker where prediction stops being worth anything
      const X = this.sx(pts[h][0]), Y = this.sy(pts[h][1]);
      const r = Math.max(5, pred.width[h] * s);
      c.strokeStyle = "rgba(255,200,120,0.9)"; c.lineWidth = 1.5; c.setLineDash([2, 3]);
      c.beginPath(); c.arc(X, Y, r, 0, Math.PI * 2); c.stroke();
      c.setLineDash([]); c.lineWidth = 1;
      const label = `${(pred.horizonSeconds || 0).toFixed(1)}s · forecast ends`;
      c.font = "10px ui-monospace, monospace";
      const tw = c.measureText(label).width;
      c.fillStyle = "rgba(4,6,12,0.7)";
      c.fillRect(X + r + 4, Y - r - 17, tw + 7, 14);
      c.fillStyle = "rgba(255,214,160,0.95)";
      c.fillText(label, X + r + 7, Y - r - 7);
    }
    c.restore();
  }

  drawPrediction(pred) {
    const c = this.ctx, pts = pred.points;
    this.drawFan(pred);
    // past the horizon the single line is a lie, so it is drawn faint and
    // loosely dashed rather than pretending to a precision it does not have
    const hasFan = this.showFan && pred.width && pred.width.length > 1;
    const h = hasFan && pred.horizon != null ? Math.min(pred.horizon, pts.length - 1) : pts.length - 1;
    let dash = null;
    for (let i = 1; i < pts.length; i++) {
      const d = pts[i][2];
      const k = Math.min(1, d / 5);
      const live = i <= h;
      const a = (0.55 - 0.35 * i / pts.length) * (live ? 1 : 0.4);
      c.strokeStyle = `rgba(${Math.round(93 + 162 * k)},${Math.round(214 - 150 * k)},${Math.round(255 - 170 * k)},${a})`;
      const want = live ? "live" : "dead";
      if (want !== dash) { c.setLineDash(live ? [4, 4] : [2, 7]); dash = want; }
      c.beginPath();
      c.moveTo(this.sx(pts[i - 1][0]), this.sy(pts[i - 1][1])); c.lineTo(this.sx(pts[i][0]), this.sy(pts[i][1])); c.stroke();
    }
    c.setLineDash([]);
    if (pred.end) {
      const p = pts[pts.length - 1]; const X = this.sx(p[0]), Y = this.sy(p[1]);
      c.strokeStyle = "#ff5d6c"; c.lineWidth = 2;
      c.beginPath(); c.moveTo(X - 6, Y - 6); c.lineTo(X + 6, Y + 6); c.moveTo(X + 6, Y - 6); c.lineTo(X - 6, Y + 6); c.stroke();
      c.lineWidth = 1;
    }
  }

  drawTargetArrow(frame, sim) {
    const [tx, ty] = this.ep.waypoint(frame, sim ? sim.leg : 0);
    const X = this.sx(tx), Y = this.sy(ty);
    const ins = this.insets, m = 24;
    const l = ins.left + m, r = this.W - ins.right - m, t = ins.top + m, b = this.H - ins.bottom - m;
    if (X > l && X < r && Y > t && Y < b) return;
    const cx = (l + r) / 2, cy = (t + b) / 2;
    const a = Math.atan2(Y - cy, X - cx);
    const k = Math.min((r - l) / 2 / Math.abs(Math.cos(a) || 1e-6), (b - t) / 2 / Math.abs(Math.sin(a) || 1e-6));
    const ax = cx + Math.cos(a) * k, ay = cy + Math.sin(a) * k;
    const c = this.ctx;
    c.save(); c.translate(ax, ay); c.rotate(a);
    c.fillStyle = "#5dff9d"; c.beginPath(); c.moveTo(10, 0); c.lineTo(-6, -7); c.lineTo(-6, 7); c.closePath(); c.fill();
    c.restore();
    if (sim) {
      c.fillStyle = "#5dff9d"; c.font = "10px monospace";
      c.fillText(`${Math.round(Math.hypot(tx - sim.x, ty - sim.y))}u`, ax - Math.cos(a) * 30 - 10, ay - Math.sin(a) * 18 + 4);
    }
  }

  drawScale() {
    const c = this.ctx, s = this.cam.s;
    const target = 120 / s;
    const p = Math.pow(10, Math.floor(Math.log10(target)));
    const len = [1, 2, 5, 10].map(k => k * p).filter(v => v <= target).pop() || p;
    const x = this.scaleAtTop ? this.insets.left + 8 : Math.max(8, this.W - this.insets.right - 20 - len * s);
    const y = this.scaleAtTop ? this.insets.top + 22 : this.H - this.insets.bottom - 6;
    c.strokeStyle = "rgba(200,215,255,0.5)"; c.beginPath(); c.moveTo(x, y); c.lineTo(x + len * s, y); c.stroke();
    c.fillStyle = "rgba(200,215,255,0.6)"; c.font = "10px monospace"; c.fillText(`${len} u`, x, y - 5);
  }
}
