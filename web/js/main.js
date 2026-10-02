import { Episode, ShipSim, predict, predictFan, instruments, STATUS_NAMES } from "./sim.js";
import { Renderer, OVERLAY_COLORS } from "./render.js";

const $ = (id) => document.getElementById(id);
const renderer = new Renderer($("view"));

const app = {
  pools: [],
  ep: null,
  sim: null,
  mode: "loading",      // ready | playing | paused | done | replay
  overlays: [],
  replayTick: 0,
  replayPlaying: false,
  speed: 1,
  acc: 0,
  time: 0,
  prediction: null,
  saved: false,
  rewinding: false,
  rewindHeld: 0,
  panel: null,          // last instruments() reading
  panelTimer: 0,
  gauge: null,          // which readouts this level has earned
};
const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const PRED_SECONDS = 20;     // how far the ballistic forecast is integrated
function setSpeed(v) {
  app.speed = v;
  $("speedv").textContent = v < 1 ? `${v}×`.replace("0.", ".") : `${v}×`;
}
function stepSpeed(dir) {
  const i = SPEEDS.indexOf(app.speed);
  setSpeed(SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, (i < 0 ? 2 : i) + dir))]);
}

window.__spacenav = app;   // debug handle: lets UI tests read sim state from the page

// ---------------------------------------------------------------- input
const keys = new Set();
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  keys.add(e.code);
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) e.preventDefault();
  switch (e.code) {
    case "Space": togglePlay(); break;
    case "KeyR": restart(); break;
    case "KeyF": setFollow(!renderer.follow); banner(renderer.follow ? "follow on" : "follow off", 800); break;
    case "KeyP": togglePrediction(); break;
    case "KeyH": document.body.classList.toggle("nopanels"); break;
    case "BracketLeft": stepSpeed(-1); break;
    case "BracketRight": stepSpeed(1); break;
    case "KeyZ": if (!e.repeat) beginRewind(); break;
    case "KeyN": nextTask(); break;
  }
  if (["ArrowUp", "KeyW", "ArrowLeft", "ArrowRight", "KeyA", "KeyD"].includes(e.code)) {
    if (app.mode === "ready") start();
    else if (app.mode === "paused" && !app.rewinding) resume();
  }
});
window.addEventListener("keyup", (e) => { keys.delete(e.code); if (e.code === "KeyZ") endRewind(); });
window.addEventListener("blur", () => keys.clear());

const touch = { left: false, right: false, thrust: false, gentle: false };

// Turn input is shaped rather than binary: at 3.5 rad/s full stick, a 200 ms tap
// would swing the nose ~40 degrees.  A tap starts gentle and ramps to full over
// RAMP_S of holding, so nudges are possible and long turns still go at full rate.
// The ship's turn rate itself is simulator physics and is left alone.
const TURN_TAP = 0.28;         // stick fraction at the instant of a tap
const RAMP_S = 0.55;           // seconds of holding to reach full stick
let turnSince = 0;             // when the current turn input started

function turnSensitivity() {
  const el = $("sens");
  return el ? +el.value : 0.7;
}

function control() {
  const left = keys.has("ArrowLeft") || keys.has("KeyA") || touch.left;
  const right = keys.has("ArrowRight") || keys.has("KeyD") || touch.right;
  const up = keys.has("ArrowUp") || keys.has("KeyW") || touch.thrust;
  const gentle = keys.has("ShiftLeft") || keys.has("ShiftRight") || touch.gentle;
  const dir = (left ? 1 : 0) - (right ? 1 : 0);
  if (!dir) { turnSince = 0; return [0, up ? (gentle ? 0.3 : 1) : 0]; }
  if (!turnSince) turnSince = performance.now();
  const held = (performance.now() - turnSince) / 1000;
  const ramp = Math.min(1, TURN_TAP + (1 - TURN_TAP) * held / RAMP_S);
  const fine = gentle ? 0.4 : 1;          // the gentle button also fine-tunes steering
  const perRealSecond = 1 / Math.max(app.speed, 0.25);   // keep the feel speed-independent
  return [dir * ramp * fine * turnSensitivity() * perRealSecond, up ? (gentle ? 0.3 : 1) : 0];
}

// on-screen pads: hold to turn / thrust; ½ latches gentle thrust
function bindPad(id, key) {
  const el = $(id);
  const set = (on) => { touch[key] = on; el.classList.toggle("active", on); };
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch {}
    set(true);
    if (app.mode === "ready") start();
    else if (app.mode === "paused" && !app.rewinding) resume();
  });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) el.addEventListener(ev, () => set(false));
}
bindPad("tLeft", "left");
bindPad("tRight", "right");
bindPad("tThrust", "thrust");
$("tGentle").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  touch.gentle = !touch.gentle;
  $("tGentle").classList.toggle("latched", touch.gentle);
});
document.addEventListener("contextmenu", (e) => { if (document.body.classList.contains("mobile")) e.preventDefault(); });

// wheel zoom; one-pointer drag pans, two-pointer pinch zooms (mouse and touch alike)
const cv = $("view");
cv.addEventListener("wheel", (e) => { e.preventDefault(); renderer.zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
const pointers = new Map();
let pinch = null;
function pinchState() {
  const [a, b] = [...pointers.values()];
  return { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
}
cv.addEventListener("pointerdown", (e) => {
  closeDrawer();
  try { cv.setPointerCapture(e.pointerId); } catch {}
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 2) pinch = pinchState();
});
cv.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (pointers.size === 1) {
    renderer.cam.x -= dx / renderer.cam.s;
    renderer.cam.y += dy / renderer.cam.s;
    if (Math.abs(dx) + Math.abs(dy) > 2) setFollow(false);
  } else if (pointers.size === 2 && pinch) {
    const now = pinchState();
    renderer.cam.x -= (now.mx - pinch.mx) / renderer.cam.s;
    renderer.cam.y += (now.my - pinch.my) / renderer.cam.s;
    if (pinch.d > 0) renderer.zoomAt(now.mx, now.my, now.d / pinch.d);
    pinch = now;
  }
});
for (const ev of ["pointerup", "pointercancel"]) cv.addEventListener(ev, (e) => {
  pointers.delete(e.pointerId);
  pinch = pointers.size === 2 ? pinchState() : null;
});

// ---------------------------------------------------------------- layout
const isMobile = () => window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 820;

function applyLayout() {
  const mobile = isMobile();
  document.body.classList.toggle("mobile", mobile);
  renderer.scaleAtTop = mobile;
  if (!mobile) { renderer.insets = { left: 300, right: 300, top: 20, bottom: 20 }; return; }
  const hud = $("hud").getBoundingClientRect();
  const pads = $("touch").getBoundingClientRect();
  // in landscape the pads only cover the bottom corners, so keep the bottom edge for the map
  const landscape = window.innerWidth > window.innerHeight;
  const padsInWay = !document.body.classList.contains("replaying") && !landscape;
  renderer.insets = { left: 8, right: 8, top: hud.bottom + 6,
                      bottom: padsInWay ? window.innerHeight - pads.top + 6 : 20 };
}
window.addEventListener("resize", () => { applyLayout(); });

function openDrawer() {
  if (app.mode === "playing") { app.mode = "paused"; banner("paused", 0, "▶ to resume"); }
  $("side").classList.add("open");
}
function closeDrawer() { $("side").classList.remove("open"); }
function setFollow(on) { renderer.follow = on; $("bFollow").classList.toggle("on", on); }

// ---------------------------------------------------------------- episodes
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r;
}

function missionSpec() {
  const stops = $("stops"), moving = $("moving"), dock = $("docking"), con = $("contested");
  if (!stops || !moving) return "3w50m";        // defensive: an older cached page
  return `${stops.value}w${moving.value}m${dock && dock.checked ? "d" : ""}` +
         `${con && con.checked ? "c" : ""}`;
}

function currentId() {
  return `${$("pool").value}-${$("level").value}-${$("seed").value}-${missionSpec()}`;
}

async function loadEpisode(eid) {
  app.mode = "loading";
  banner("loading…");
  const desc = await api(`/api/episode/${eid}`).then(r => r.json());
  app.ep = new Episode(desc);      // bodies are integrated locally from desc.init
  app.sim = new ShipSim(app.ep);
  app.overlays = [];
  app.saved = false;
  renderer.setEpisode(app.ep);
  app.panel = null;
  applyInstruments();          // before measuring: the panel sets the HUD height
  document.body.classList.remove("replaying");
  closeDrawer();
  renderMission();          // before measuring: the chips set the HUD height
  applyLayout();
  frameEpisode();
  if (location.hash.slice(1) !== eid) history.replaceState(null, "", `#${eid}`);
  syncPicker(desc);
  renderMission();
  hideResult();
  app.mode = "ready";
  banner(...launchHint());
  refreshRuns();
}

// Frame A, B, the big bodies near the route, and any zone that counts for exposure.
function frameEpisode() {
  const ep = app.ep, t = ep.task;
  const a = t.start_pos;
  const wps = t.waypoints.map((wp, w) => ep.waypoint(0, w).slice(0, 2));
  const xs = [a[0], ...wps.map(p => p[0])], ys = [a[1], ...wps.map(p => p[1])];
  const mid = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
  const reach = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.8 + 80;
  const pts = [a, ...wps];
  ep.bodies.forEach((body, i) => {
    if (body.mass < 50 || body.radius <= 0) return;
    const [x, y] = ep.body(0, i);
    if (Math.hypot(x - mid[0], y - mid[1]) < reach) pts.push([x - body.radius, y - body.radius], [x + body.radius, y + body.radius]);
  });
  if (t.weights[3] > 0) ep.zones.forEach((z) => {
    if (!z.exposure) return;
    const [x, y] = z.anchor >= 0 ? ep.body(0, z.anchor) : [0, 0];
    pts.push([x - z.r_out, y - z.r_out], [x + z.r_out, y + z.r_out]);
  });
  renderer.fit(pts, 1.25);
}

function syncPicker(desc) {
  $("pool").value = desc.pool;
  $("level").value = desc.index;
  $("seed").value = desc.seed;
  const m = desc.mission || { stops: desc.task.waypoints.length, moving: 50, docking: false };
  if (!$("stops")) return;
  $("stops").value = String(m.stops);
  $("moving").value = [0, 33, 50, 75, 100].reduce((a, b) => Math.abs(b - m.moving) < Math.abs(a - m.moving) ? b : a, 50);
  $("docking").checked = !!m.docking;
  if ($("contested")) $("contested").checked = !!m.chaos;
  const p = app.pools.find(p => p.name === desc.pool);
  if (p) {
    $("level").max = p.size - 1;
    fillFamilies(p);
    $("famsel").value = desc.family;
  }
}

// ------------------------------------------------------------- curriculum
// Bands measured offline (scripts/build_curriculum.py): how coarse a model of
// the forces still flies the level, and how much timing slack the plan leaves.
const BAND_NOTE = {
  trivial: "Point and burn. One gravity well, wide margins.",
  human: "One well, but the tank forces a real transfer — plan it.",
  calculated: "Patched conics cannot fly these: no single body dominates where you " +
              "are going. Intuition will not get you there; reading the numbers will.",
  precision: "The plan exists and no human hand can hold its timing. Agent territory.",
  opaque: "No planner found a way through. Kept only as evidence.",
};

async function loadCurriculum() {
  const sel = $("cband");
  if (!sel) return;
  let data;
  try {
    data = await api("/api/curriculum?name=v1").then(r => r.json());
  } catch (e) {
    $("cnote").textContent = "no curriculum built yet";
    return;
  }
  app.curriculum = data;
  const bands = [...new Set(data.episodes.map(e => e.band))];
  sel.innerHTML = bands.map(b =>
    `<option value="${b}">${b} (${data.episodes.filter(e => e.band === b).length})</option>`).join("");
  sel.onchange = renderCurriculum;
  renderCurriculum();
}

function renderCurriculum() {
  const band = $("cband").value;
  const list = app.curriculum.episodes.filter(e => e.band === band);
  $("cnote").textContent = BAND_NOTE[band] || "";
  $("clist").innerHTML = list.slice(0, 40).map(e =>
    `<button class="lvlbtn" data-eid="${e.id}" title="model scale ${e.model_scale || "full field"} · ` +
    `window ${e.window_s}s · eta ${e.eta_mean}">${e.stage || band} · ${e.family} ` +
    `<span class="dim">${e.scales} scale${e.scales === 1 ? "" : "s"}</span></button>`).join("");
  for (const b of $("clist").querySelectorAll("button")) {
    b.onclick = () => loadEpisode(b.dataset.eid);
  }
}

function fillFamilies(p) {
  const sel = $("famsel");
  sel.innerHTML = "";
  for (const [fam, info] of Object.entries(p.families)) {
    const o = document.createElement("option");
    o.value = fam; o.textContent = `${fam} (${info.count})`;
    sel.appendChild(o);
  }
}

function renderMission() {
  const d = app.ep.desc, t = d.task, K = d.constants;
  $("family").textContent = d.family.replace("_", " ");
  $("eid").textContent = d.id;
  const chips = t.weights.map((w, i) => w > 0.005 ? `<span class="chip">${K.OBJECTIVES[i]} <b>${Math.round(w * 100)}%</b></span>` : "").join("");
  $("objective").innerHTML = `<span class="dim small">minimise:</span>${chips}`;
  const n = t.waypoints.length;
  const dv = (t.fuel * t.accel).toFixed(0);
  const legs = n > 1 ? `${n} waypoints ${t.order_free ? "in any order" : "in order"}` : "one target";
  const dock = t.waypoints.filter(w => w.rendezvous).length;
  $("targetinfo").textContent = `${legs}${dock ? `, ${dock} needing a dock at ≤ ${K.RENDEZVOUS_V_TOL} u/s` : ""}. `
    + `Ship: ${dv} u/s of Δv, thrust ${t.accel.toFixed(1)} u/s².`;
  const w = $("warn");
  if (!d.valid) { w.textContent = "task sampler could not find a clean A/B for this seed"; w.classList.remove("hidden"); }
  else w.classList.add("hidden");
}

// ---------------------------------------------------------------- game flow
function start() {
  if (!app.sim || app.mode !== "ready") return;
  closeDrawer();
  app.mode = "playing"; app.acc = 0; banner("");
}
function resume() {
  if (app.mode !== "paused") return;
  app.mode = "playing"; app.acc = 0; banner("");
}

// Hold to rewind: the run pauses, steps back through tick history (faster the
// longer it is held), and flying resumes from there on the next input.
function beginRewind() {
  const sim = app.sim;
  if (!sim || !["playing", "paused", "done"].includes(app.mode) || sim.history.length <= 1) return;
  hideResult();
  if (app.mode === "done") sim.rewindTo(sim.history.length - 1);
  app.mode = "paused";
  app.rewinding = true; app.rewindHeld = 0; app.rewindAcc = 0;
  sim.rewinds += 1;
  $("bRewind").classList.add("active");
  banner("rewinding", 0, "release to stop");
}
function endRewind() {
  if (!app.rewinding) return;
  app.rewinding = false;
  $("bRewind").classList.remove("active");
  banner("paused", 0, isMobile() ? "▶ or any control to fly on from here" : "Space or any control to fly on from here");
}
function rewindFrame(dt) {
  const sim = app.sim, K = app.ep.K;
  app.rewindHeld += dt;
  const rate = Math.min(1 + app.rewindHeld * 1.5, 8) / K.CTRL_DT;   // ticks per second
  app.rewindAcc += dt * rate;
  const n = Math.floor(app.rewindAcc);
  if (n > 0) { app.rewindAcc -= n; sim.rewindTo(sim.tick - n); }
}

function togglePlay() {
  if (app.mode === "ready") return start();
  if (app.mode === "playing") { app.mode = "paused"; banner("paused", 0, isMobile() ? "▶ to resume" : "Space to resume"); }
  else if (app.mode === "paused") resume();
  else if (app.mode === "replay") app.replayPlaying = !app.replayPlaying;
  else if (app.mode === "done") restart();
}
function restart() {
  if (!app.ep) return;
  app.sim = new ShipSim(app.ep);
  app.mode = "ready"; app.saved = false; app.replayTick = 0; app.rewinding = false;
  hideResult();
  document.body.classList.remove("replaying");
  applyLayout();
  banner(...launchHint());
}
function launchHint() {
  return isMobile() ? ["hold ▲ to launch", 0, "⟲ ⟳ turn · ▲ thrust · reach B"]
                    : ["press ↑ or Space to launch", 0, "hold ↑ to thrust · ←/→ to turn · reach B"];
}
function togglePrediction() {
  renderer.showPrediction = !renderer.showPrediction;
  $("bPred").classList.toggle("on", renderer.showPrediction);
  // drop the stale fan so the horizon readout falls back to its estimate
  // instead of reporting a measurement from a forecast no longer being run
  if (!renderer.showPrediction) { app.prediction = null; app.panelTimer = 0; }
}
async function nextTask() {
  $("seed").value = Number($("seed").value) + 1;
  await loadEpisode(currentId());
}

function finish() {
  app.mode = "done";
  const sim = app.sim, K = app.ep.K, t = app.ep.task;
  const st = STATUS_NAMES[sim.status];
  $("rtitle").textContent = st === "arrived" ? "Arrived" : st;
  $("rtitle").className = st === "arrived" ? "arrived" : "fail";
  const c = sim.costs;
  const rows = K.OBJECTIVES.map((name, i) => {
    const raw = [c.time, c.path, c.fuel, c.exposure][i];
    const norm = raw / t.ref_scale[i];
    return `<tr><td>${name}</td><td>${raw.toFixed(1)}</td><td>${norm.toFixed(2)}</td><td>${(t.weights[i] * 100).toFixed(0)}%</td><td>${(t.weights[i] * norm).toFixed(3)}</td></tr>`;
  }).join("");
  $("rtable").innerHTML = `<tr><th>cost</th><th>raw</th><th>/ ref</th><th>weight</th><th>score</th></tr>${rows}
    <tr><td>damage</td><td>${c.damage.toFixed(1)}</td><td colspan=3 class="dim">hull ${sim.health.toFixed(0)}/${K.SHIP_HEALTH}</td></tr>
    <tr class="total"><td>objective</td><td colspan=3></td><td>${st === "arrived" ? sim.objective().toFixed(3) : "—"}</td></tr>`;
  $("rnote").textContent = st === "arrived" ? "Lower objective is better. Refs are straight-line placeholders until the optimiser baseline lands."
    : "Mission failed.";
  $("save").disabled = false;
  $("result").classList.remove("hidden");
}
function hideResult() { $("result").classList.add("hidden"); }

async function saveRun() {
  const sim = app.sim;
  $("save").disabled = true;
  const body = {
    episode: app.ep.desc.id, name: sim.rewinds ? `human (${sim.rewinds} rewinds)` : "human", actions: sim.actions,
    client: { status: STATUS_NAMES[sim.status], costs: sim.costs, objective: sim.objective(),
              trajectory: sim.trajectory.map(p => p.map(v => +v.toFixed(3))) },
  };
  const r = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json());
  const ok = r.jax_status === STATUS_NAMES[sim.status];
  $("rnote").textContent = `Saved. JAX re-sim: ${r.jax_status}${ok ? " ✓" : " ✗ (mismatch!)"}; max position error ${r.parity_max_err?.toFixed(3)} u.`;
  app.saved = true;
  refreshRuns();
}

// ---------------------------------------------------------------- runs & replay
async function refreshRuns() {
  if (!app.ep) return;
  const runs = await api(`/api/runs/${app.ep.desc.id}`).then(r => r.json());
  const keep = app.overlays.filter(o => o.source !== "human");
  const known = new Set(keep.map(o => o.id));
  for (const r of runs) if (!known.has(r.id)) keep.push(makeOverlay(r, r.name));
  app.overlays = keep;
  renderRuns();
}

function makeOverlay(r, label) {
  const color = OVERLAY_COLORS[app.overlays.length % OVERLAY_COLORS.length];
  return { id: r.id || label, label, source: r.source || "baseline", status: r.status, costs: r.costs,
           objective: r.objective, trajectory: r.trajectory, color, visible: true };
}

function renderRuns() {
  const box = $("runs");
  box.innerHTML = "";
  if (!app.overlays.length) { box.innerHTML = `<div class="dim small">no runs yet</div>`; return; }
  app.overlays.forEach((o) => {
    const row = document.createElement("label");
    row.className = "run";
    const score = o.status === "arrived" ? o.objective.toFixed(3) : "";
    row.innerHTML = `<input type="checkbox" ${o.visible ? "checked" : ""}><span class="sw" style="background:${o.color}"></span>
      <span>${o.label}</span><span class="st ${o.status}">${o.status} ${score}</span>`;
    row.querySelector("input").addEventListener("change", (e) => { o.visible = e.target.checked; });
    box.appendChild(row);
  });
}

async function runBaseline(name) {
  const r = await api(`/api/episode/${app.ep.desc.id}/baseline/${name}`).then(r => r.json());
  app.overlays = app.overlays.filter(o => o.id !== `baseline:${name}`);
  const ov = makeOverlay({ ...r, id: `baseline:${name}` }, name);
  app.overlays.push(ov);
  renderRuns();
  enterReplay();
}

function enterReplay() {
  if (!app.ep) return;
  hideResult();
  // include the current (possibly unsaved) human attempt
  app.overlays = app.overlays.filter(o => o.id !== "live");
  if (app.sim && app.sim.trajectory.length > 1) {
    app.overlays.unshift({ id: "live", label: "you (latest)", source: "live", status: STATUS_NAMES[app.sim.status],
      objective: app.sim.objective(), trajectory: app.sim.trajectory, color: "#ffffff", visible: true });
  }
  renderRuns();
  app.mode = "replay"; app.replayTick = 0; app.replayPlaying = true;
  closeDrawer();
  document.body.classList.add("replaying");
  applyLayout();
  const maxT = Math.max(1, ...app.overlays.map(o => o.trajectory.length - 1));
  $("scrub").max = maxT;
  banner("replay", 0, isMobile() ? "▶ play/pause · ↻ back to flying · scrub in ☰" : "Space play/pause · drag the slider to scrub");
}

// ---------------------------------------------------------------- HUD
function banner(text, ms = 0, sub = "") {
  $("banner").innerHTML = text ? `${text}${sub ? `<span class="sub">${sub}</span>` : ""}` : "";
  if (ms) setTimeout(() => { if ($("banner").textContent.startsWith(text)) $("banner").innerHTML = ""; }, ms);
}

function updateHud() {
  const sim = app.sim, ep = app.ep;
  if (!sim) return;
  const K = ep.K;
  $("hull").style.width = `${(100 * sim.health / K.SHIP_HEALTH).toFixed(1)}%`;
  $("hullv").textContent = sim.health.toFixed(0);
  $("fuel").style.width = `${(100 * sim.fuel / ep.task.fuel).toFixed(1)}%`;
  $("fuelv").textContent = sim.fuel.toFixed(1);
  $("time").textContent = sim.costs.time.toFixed(1);
  $("speed").textContent = Math.hypot(sim.vx, sim.vy).toFixed(1);
  const [tx, ty, tvx, tvy] = ep.waypoint(sim.frame, sim.leg);
  $("dist").textContent = Math.max(0, Math.hypot(tx - sim.x, ty - sim.y) - ep.task.waypoints[sim.leg].radius).toFixed(0);
  const relv = Math.hypot(tvx - sim.vx, tvy - sim.vy);
  $("relv").textContent = relv.toFixed(1);
  $("relv").style.color = ep.task.waypoints[sim.leg].rendezvous
    ? (relv <= K.RENDEZVOUS_V_TOL ? "var(--good)" : "var(--warn)") : "";
  $("score").textContent = sim.objective().toFixed(2);
  const hz = sim.lastHazard, parts = hz.parts;
  const tags = [];
  for (const [k, label] of [["radiation", "RADIATION"], ["heat", "ATMO HEAT"], ["debris", "DEBRIS"], ["belts", "RAD BELT"]]) {
    if (parts[k] > 0.05) tags.push(`<span class="hz ${k}">${label} ${parts[k].toFixed(1)}/s</span>`);
  }
  if (hz.exposure > 0.01 && ep.task.weights[3] > 0) tags.push(`<span class="hz exposure">EXPOSED</span>`);
  $("hazards").innerHTML = tags.join("");
}

// ---------------------------------------------------------- instruments
// Each lesson earns one instrument (spacenav/lessons.py INSTRUMENTS).  A level
// declares what it grants in `desc.instruments`, as either lesson keys or
// instrument names; an episode without the field grants everything, so pages
// served by an older API keep the full panel.
const INSTRUMENT_BY_LESSON = {
  "free-ride": "none",
  "dead-heading": "target marker",
  "leading": "lead indicator",
  "conics": "tidal discriminator",
  "gravity-assist": "budget gauge",
  "rolling-with-it": "predictability cone",
  "one-exact-boost": "flight computer",
};
const ALL_INSTRUMENTS = ["target marker", "lead indicator", "tidal discriminator",
                         "budget gauge", "predictability cone", "flight computer"];

function instrumentSet(desc) {
  const raw = desc && Array.isArray(desc.instruments) ? desc.instruments : null;
  if (!raw) return new Set(ALL_INSTRUMENTS);
  const out = new Set();
  for (const it of raw) {
    const name = INSTRUMENT_BY_LESSON[it] || it;
    if (name && name !== "none") out.add(name);
  }
  return out;
}

function panelWanted() {
  const box = $("showinstr");
  return box ? box.checked : true;
}

// Reads the level's grant, shows/hides the panel and its cells, and decides
// whether the prediction gets the full ensemble or just the single line.
function applyInstruments() {
  const set = instrumentSet(app.ep && app.ep.desc);
  const computer = set.has("flight computer");
  app.gauge = {
    eta: computer || set.has("tidal discriminator"),
    budget: computer || set.has("budget gauge"),
    cone: computer || set.has("predictability cone"),
    stretch: computer,
  };
  const g = app.gauge;
  const any = g.eta || g.budget || g.cone || g.stretch;
  $("instr").classList.toggle("hidden", !(any && panelWanted()));
  setIns("insEta", g.eta, "—", "");
  setIns("insA", g.budget, "—", "");
  setIns("insD", g.eta && g.budget, "—", "");
  setIns("insHz", g.cone, "—", "");
  setIns("insS", g.stretch, "—", "");
  // the cone is the one instrument that lives on the map rather than in the HUD
  renderer.showFan = g.cone;
  const note = $("instrnote");
  if (note) {
    const declared = app.ep && Array.isArray(app.ep.desc.instruments);
    const names = [...set].filter(Boolean);
    note.textContent = declared
      ? (names.length ? `this level grants: ${names.join(", ")}` : "this level grants no instruments")
      : "level declares none — all instruments on";
  }
  app.panelTimer = 0;
}

function setIns(id, show, text, cls) {
  const el = $(id);
  if (!el) return;
  el.classList.toggle("hidden", !show);
  el.classList.remove("lv-ok", "lv-mid", "lv-bad");
  if (!show) return;
  el.querySelector("span").textContent = text;
  if (cls) el.classList.add(cls);
}

const etaLevel = (v) => (v < 0.05 ? "lv-ok" : v <= 0.3 ? "lv-mid" : "lv-bad");
const budgetLevel = (v) => (v < 1 ? "lv-bad" : v <= 3 ? "lv-mid" : "lv-ok");
const demandLevel = (v) => (v < 0.05 ? "lv-ok" : v <= 0.3 ? "lv-mid" : "lv-bad");
const horizonLevel = (v) => (v >= 8 ? "lv-ok" : v >= 3 ? "lv-mid" : "lv-bad");

// small numbers read as "0.00" otherwise, which looks like a measurement of zero
function fmtFrac(v) {
  if (!isFinite(v)) return "∞";
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  if (v < 0.01) return v > 0 ? "<.01" : "0";
  return v.toFixed(2);
}

function updateInstruments() {
  const g = app.gauge, sim = app.sim;
  if (!g || !sim || $("instr").classList.contains("hidden")) return;
  const p = app.panel;
  if (!p) return;
  if (g.eta) setIns("insEta", true, fmtFrac(p.eta), etaLevel(p.eta));
  if (g.budget) setIns("insA", true, fmtFrac(p.A), budgetLevel(p.A));
  if (g.eta && g.budget) setIns("insD", true, fmtFrac(p.D), demandLevel(p.D));
  if (g.cone) {
    // the forecast is only integrated PRED_SECONDS ahead, so a cone still tight
    // at the end of the window reads as a floor, not as a measurement
    const h = Math.min(p.horizonSeconds, PRED_SECONDS);
    const open = p.measured && !p.saturated && p.horizonSeconds < PRED_SECONDS;
    setIns("insHz", true, open ? `${h.toFixed(1)}s` : `≥${PRED_SECONDS}s`, horizonLevel(h));
  }
  if (g.stretch) setIns("insS", true, `${p.stretch.toFixed(3)}/s`, "");
}

// ---------------------------------------------------------------- main loop
let last = performance.now();
let predTimer = 0;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  app.time += dt;
  const sim = app.sim;

  if (sim && app.mode === "playing") {
    const K = app.ep.K;
    app.acc += dt * app.speed;
    let n = 0;
    while (app.acc >= K.PHYS_DT && sim.running && n < 2000) {
      if (sim.frame % K.SUBSTEPS === 0) sim.setAction(...control());
      sim.substep();
      app.acc -= K.PHYS_DT; n++;
    }
    if (!sim.running) finish();
  }

  if (sim && app.rewinding) rewindFrame(dt);

  let bodyFrame = sim ? sim.frame : 0;
  if (app.mode === "replay") {
    const K = app.ep.K;
    if (app.replayPlaying) {
      app.replayTick += dt * app.speed / K.CTRL_DT;
      if (app.replayTick >= +$("scrub").max) { app.replayTick = +$("scrub").max; app.replayPlaying = false; }
      $("scrub").value = app.replayTick;
    }
    bodyFrame = Math.round(app.replayTick) * K.SUBSTEPS;
  }

  if (sim && (app.mode === "playing" || app.mode === "ready" || app.mode === "paused")) {
    predTimer -= dt;
    if (predTimer <= 0 && renderer.showPrediction) {
      // the ensemble costs 8 extra ballistic integrations, so run it a little
      // less often than the single line used to refresh.  Without the
      // predictability cone earned, only the single line is computed, and
      // drawFan degrades to drawing nothing.
      app.prediction = renderer.showFan ? predictFan(sim, PRED_SECONDS)
                                        : predict(sim, PRED_SECONDS, 6);
      predTimer = 0.18;
    }
  } else app.prediction = null;

  // the readouts are a handful of passes over the bodies; 5 Hz is plenty
  if (sim) {
    app.panelTimer -= dt;
    if (app.panelTimer <= 0) {
      const pred = app.prediction;
      const measured = !!(pred && pred.width && pred.horizonSeconds != null);
      app.panel = instruments(sim, { horizonSeconds: measured ? pred.horizonSeconds : null });
      app.panel.measured = measured;
      app.panel.saturated = measured && pred.horizon >= pred.width.length - 1;
      app.panelTimer = 0.2;
    }
  }

  if (renderer.follow && sim && app.mode !== "replay") {
    renderer.cam.x += (sim.x - renderer.cam.x) * Math.min(1, dt * 6);
    renderer.cam.y += (sim.y - renderer.cam.y) * Math.min(1, dt * 6);
  }

  renderer.draw({ frame: bodyFrame, sim, overlays: app.mode === "replay" || app.mode === "done" ? app.overlays : [],
                  replayTick: app.mode === "replay" ? app.replayTick : 1e9, prediction: app.prediction,
                  time: app.time, showLive: app.mode !== "replay" });
  updateHud();
  updateInstruments();
  const playing = app.mode === "playing" || (app.mode === "replay" && app.replayPlaying);
  $("bPlay").textContent = playing ? "❚❚" : "▶";
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- wiring
$("go").onclick = () => loadEpisode(currentId());
$("nextseed").onclick = nextTask;
$("rand").onclick = () => {
  const p = app.pools.find(p => p.name === $("pool").value);
  const fam = p.families[$("famsel").value];
  $("level").value = fam.first + Math.floor(Math.random() * fam.count);
  $("seed").value = Math.floor(Math.random() * 1000);
  loadEpisode(currentId());
};
$("pool").onchange = () => { const p = app.pools.find(p => p.name === $("pool").value); fillFamilies(p); $("level").value = 0; };
$("famsel").onchange = () => { const p = app.pools.find(p => p.name === $("pool").value); $("level").value = p.families[$("famsel").value].first; };
$("pilot").onclick = () => runBaseline("pilot");
$("coast").onclick = () => runBaseline("coast");
$("refresh").onclick = refreshRuns;
$("replay").onclick = enterReplay;
$("toplay").onclick = restart;
$("scrub").oninput = (e) => { if (app.mode !== "replay") enterReplay(); app.replayPlaying = false; app.replayTick = +e.target.value; };
$("save").onclick = saveRun;
$("bPlay").onclick = togglePlay;
$("bSlower").onclick = () => stepSpeed(-1);
$("bFaster").onclick = () => stepSpeed(1);
{
  const b = $("bRewind");
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); try { b.setPointerCapture(e.pointerId); } catch {} beginRewind(); });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, endRewind);
}
$("bRestart").onclick = restart;
$("bFollow").onclick = () => setFollow(!renderer.follow);
$("bPred").onclick = togglePrediction;
$("bMenu").onclick = () => ($("side").classList.contains("open") ? closeDrawer() : openDrawer());
$("bClose").onclick = closeDrawer;
{
  const el = $("sens");
  try { const v = localStorage.getItem("spacenav.sens"); if (v) el.value = v; } catch {}
  const show = () => { $("sensv").textContent = `${(+el.value).toFixed(2)} · hold to turn faster · ½ for fine steering`; };
  el.addEventListener("input", () => { show(); try { localStorage.setItem("spacenav.sens", el.value); } catch {} });
  show();
}
$("mission").onclick = () => $("mission").classList.toggle("expanded");
{
  const box = $("showinstr");
  if (box) {
    try { const v = localStorage.getItem("spacenav.instruments"); if (v !== null) box.checked = v === "1"; } catch {}
    box.addEventListener("change", () => {
      try { localStorage.setItem("spacenav.instruments", box.checked ? "1" : "0"); } catch {}
      applyInstruments();
      applyLayout();
      banner(box.checked ? "instruments on" : "instruments off", 900);
    });
  }
  // the panel is one line of sky on a phone, so the explanation is a tap away
  $("instr").addEventListener("click", (e) => {
    e.stopPropagation();
    $("gauges").classList.toggle("expanded");
  });
}
$("retry").onclick = restart;
$("back3").onclick = () => {
  const sim = app.sim;
  if (!sim) return;
  hideResult();
  sim.rewindTo(sim.history.length - 1 - Math.round(3 / app.ep.K.CTRL_DT));
  sim.rewinds += 1;
  app.mode = "paused";
  banner("paused 3 s earlier", 0, isMobile() ? "▶ or any control to fly on" : "Space or any control to fly on");
};
$("rreplay").onclick = enterReplay;
$("rnext").onclick = nextTask;

async function boot() {
  setSpeed(1);
  applyLayout();
  app.pools = await api("/api/pools").then(r => r.json());
  const sel = $("pool");
  for (const p of app.pools) { const o = document.createElement("option"); o.value = p.name; o.textContent = `${p.name} (${p.size})`; sel.appendChild(o); }
  const hash = location.hash.slice(1);
  const eid = hash || `${app.pools.find(p => p.name === "val_seen") ? "val_seen" : app.pools[0].name}-0-0-f`;
  await loadEpisode(eid);
  loadCurriculum().catch(() => {});      // optional: only there once probed
  requestAnimationFrame(frame);
}
window.addEventListener("hashchange", () => {
  const eid = location.hash.slice(1);
  if (eid && (!app.ep || eid !== app.ep.desc.id)) loadEpisode(eid).catch((e) => banner("error", 0, e.message));
});
boot().catch((e) => banner("error", 0, e.message));
