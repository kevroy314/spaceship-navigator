/**
 * renderer.js — Three.js rendering module.
 *
 * Renders the gravitational field heatmap, celestial bodies, the player ship,
 * its trail, target approach radius, and optional grid lines.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Module-level references (set during initScene)
// ---------------------------------------------------------------------------

let scene, camera, renderer;

// Groups
let fieldGroup, bodiesGroup, shipGroup, trailGroup, gridGroup;

// Objects
let fieldMesh = null;
let fieldTexture = null;
let fieldTextureSize = 0;
let shipMesh = null;
let targetRingMesh = null;

// Trail
const TRAIL_MAX = 600;
let trailPositions = [];
let trailLine = null;

// Grid
let gridVisible = false;
let gridHelper = null;

// Pulsing state
let pulseTime = 0;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * One-time scene initialisation.
 */
export function initScene(_scene, _camera, _renderer) {
  scene = _scene;
  camera = _camera;
  renderer = _renderer;

  // Create groups
  fieldGroup  = new THREE.Group(); fieldGroup.name  = 'field';
  bodiesGroup = new THREE.Group(); bodiesGroup.name = 'bodies';
  shipGroup   = new THREE.Group(); shipGroup.name   = 'ship';
  trailGroup  = new THREE.Group(); trailGroup.name  = 'trail';
  gridGroup   = new THREE.Group(); gridGroup.name   = 'grid';

  scene.add(fieldGroup);
  scene.add(trailGroup);
  scene.add(bodiesGroup);
  scene.add(shipGroup);

  // Ambient light (for MeshBasicMaterial we don't strictly need it, but
  // keeps the door open for fancier materials later).
  scene.add(new THREE.AmbientLight(0xffffff, 1));

  // Grid (optional – toggled by pressing G)
  createGrid();
  window.addEventListener('keydown', (e) => {
    if (e.key === 'g' || e.key === 'G') {
      toggleGrid();
    }
  });
}

export function getCamera() {
  return camera;
}

export function resizeRenderer(_w, _h) {
  // Currently nothing extra; camera update is handled in main.js
}

// ---------------------------------------------------------------------------
// Gravitational field heatmap
// ---------------------------------------------------------------------------

/**
 * fieldData: { width: number, height: number, values: number[],
 *              min: number, max: number, extent: [x0, y0, x1, y1] }
 *
 * `values` is a flat row-major array of potential magnitudes.
 */
export function updateField(fieldData) {
  if (!fieldData || !fieldData.values) return;

  const { width, height, values, min, max, extent } = fieldData;

  // (Re-)create texture if size changed
  if (!fieldTexture || fieldTextureSize !== width * height) {
    if (fieldTexture) fieldTexture.dispose();

    const data = new Uint8Array(width * height * 4);
    fieldTexture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
    fieldTexture.minFilter = THREE.LinearFilter;
    fieldTexture.magFilter = THREE.LinearFilter;
    fieldTextureSize = width * height;

    // Remove old mesh
    if (fieldMesh) {
      fieldGroup.remove(fieldMesh);
      fieldMesh.geometry.dispose();
      fieldMesh.material.dispose();
    }

    const [x0, y0, x1, y1] = extent;
    const geo = new THREE.PlaneGeometry(x1 - x0, y1 - y0);
    const mat = new THREE.MeshBasicMaterial({
      map: fieldTexture,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    });
    fieldMesh = new THREE.Mesh(geo, mat);
    fieldMesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, -1);
    fieldGroup.add(fieldMesh);
  }

  // Write colour data — blue (low potential) to red (high potential)
  const img = fieldTexture.image.data;
  const range = max - min || 1;

  for (let i = 0; i < values.length; i++) {
    const t = (values[i] - min) / range; // 0..1
    const idx = i * 4;
    // Blue → Cyan → Yellow → Red
    const r = Math.min(1, Math.max(0, 2 * t - 0.5));
    const g = t < 0.5 ? t * 2 : 2 * (1 - t);
    const b = Math.min(1, Math.max(0, 1 - 2 * t));
    img[idx]     = (r * 255) | 0;
    img[idx + 1] = (g * 255) | 0;
    img[idx + 2] = (b * 255) | 0;
    img[idx + 3] = 180;
  }

  fieldTexture.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

const bodyTypeColors = {
  star:     0xffdd44,
  planet:   0x4488ee,
  asteroid: 0x888888,
  target:   0x44ff88,
};

/**
 * bodiesData: Array<{ id, type, x, y, mass, radius? }>
 */
export function updateBodies(bodiesData) {
  if (!bodiesData) return;

  pulseTime += 0.016; // ~60 fps approximation

  // Dispose existing
  while (bodiesGroup.children.length) {
    const c = bodiesGroup.children[0];
    bodiesGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }

  // Remove old target ring
  if (targetRingMesh) {
    bodiesGroup.remove(targetRingMesh);
    if (targetRingMesh.geometry) targetRingMesh.geometry.dispose();
    if (targetRingMesh.material) targetRingMesh.material.dispose();
    targetRingMesh = null;
  }

  for (const body of bodiesData) {
    const visualRadius = body.radius ?? Math.max(1, Math.log(body.mass + 1) * 1.5);
    let color = bodyTypeColors[body.type] ?? 0xcccccc;

    // Pulsing for target
    if (body.type === 'target') {
      const pulse = 0.5 + 0.5 * Math.sin(pulseTime * 4);
      // Interpolate toward white during pulse
      const base = new THREE.Color(0x44ff88);
      const bright = new THREE.Color(0xccffee);
      base.lerp(bright, pulse);
      color = base;
    }

    const geo = new THREE.CircleGeometry(visualRadius, 32);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: body.type === 'star' ? 1.0 : 0.9,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(body.x, body.y, 0);
    bodiesGroup.add(mesh);

    // Glow ring for stars
    if (body.type === 'star') {
      const glowGeo = new THREE.RingGeometry(visualRadius, visualRadius * 1.4, 48);
      const glowMat = new THREE.MeshBasicMaterial({
        color: 0xffee88,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
      });
      const glowMesh = new THREE.Mesh(glowGeo, glowMat);
      glowMesh.position.set(body.x, body.y, 0);
      bodiesGroup.add(glowMesh);
    }

    // Target approach radius
    if (body.type === 'target' && body.approach_radius) {
      const ringGeo = new THREE.RingGeometry(
        body.approach_radius - 0.3,
        body.approach_radius + 0.3,
        64,
      );
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x44ff88,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      targetRingMesh = new THREE.Mesh(ringGeo, ringMat);
      targetRingMesh.position.set(body.x, body.y, 0);
      bodiesGroup.add(targetRingMesh);

      // Dashed circle via line segments
      const dashSegs = 48;
      const dashPoints = [];
      for (let i = 0; i <= dashSegs; i++) {
        if (i % 2 === 1) { // skip every other segment for dashing
          const a0 = ((i - 1) / dashSegs) * Math.PI * 2;
          const a1 = (i / dashSegs) * Math.PI * 2;
          dashPoints.push(
            new THREE.Vector3(
              body.x + Math.cos(a0) * body.approach_radius,
              body.y + Math.sin(a0) * body.approach_radius,
              0.1,
            ),
            new THREE.Vector3(
              body.x + Math.cos(a1) * body.approach_radius,
              body.y + Math.sin(a1) * body.approach_radius,
              0.1,
            ),
          );
        }
      }
      if (dashPoints.length) {
        const dashGeo = new THREE.BufferGeometry().setFromPoints(dashPoints);
        const dashMat = new THREE.LineBasicMaterial({
          color: 0x44ff88,
          transparent: true,
          opacity: 0.5,
        });
        const dashLine = new THREE.LineSegments(dashGeo, dashMat);
        bodiesGroup.add(dashLine);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Ship
// ---------------------------------------------------------------------------

/**
 * shipData: { x, y, heading, fuel, max_fuel, mass }
 */
export function updateShip(shipData) {
  if (!shipData) return;

  // Dispose old
  while (shipGroup.children.length) {
    const c = shipGroup.children[0];
    shipGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }

  // Fuel fraction → colour: cyan (full) → red (empty)
  const fuelFrac = shipData.max_fuel
    ? Math.max(0, Math.min(1, shipData.fuel / shipData.max_fuel))
    : 1;
  const fullColor = new THREE.Color(0x00e8ff); // bright cyan
  const emptyColor = new THREE.Color(0xcc2222); // dark red
  const shipColor = fullColor.clone().lerp(emptyColor, 1 - fuelFrac);

  // Triangle / arrow shape
  const shape = new THREE.Shape();
  const s = 2.2; // half-size
  shape.moveTo(0, s * 1.6);       // nose
  shape.lineTo(-s, -s);            // left wing
  shape.lineTo(0, -s * 0.4);      // indent
  shape.lineTo(s, -s);             // right wing
  shape.closePath();

  const geo = new THREE.ShapeGeometry(shape);
  const mat = new THREE.MeshBasicMaterial({
    color: shipColor,
    transparent: true,
    opacity: 0.95,
  });
  shipMesh = new THREE.Mesh(geo, mat);
  shipMesh.position.set(shipData.x, shipData.y, 1);
  shipMesh.rotation.z = (shipData.heading ?? 0);
  shipGroup.add(shipMesh);

  // Small glow ring around ship
  const glowGeo = new THREE.RingGeometry(2.8, 3.4, 32);
  const glowMat = new THREE.MeshBasicMaterial({
    color: shipColor,
    transparent: true,
    opacity: 0.15,
    side: THREE.DoubleSide,
  });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.set(shipData.x, shipData.y, 0.9);
  shipGroup.add(glow);
}

// ---------------------------------------------------------------------------
// Trail
// ---------------------------------------------------------------------------

/**
 * Call with shipData each frame to append to the trail.
 */
export function updateTrail(shipData) {
  if (!shipData) return;

  trailPositions.push(new THREE.Vector3(shipData.x, shipData.y, 0.5));
  if (trailPositions.length > TRAIL_MAX) {
    trailPositions.shift();
  }

  // Remove old line
  while (trailGroup.children.length) {
    const c = trailGroup.children[0];
    trailGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }

  if (trailPositions.length < 2) return;

  const geo = new THREE.BufferGeometry().setFromPoints(trailPositions);

  // Fading colours: older = more transparent (we use vertex colors)
  const colors = new Float32Array(trailPositions.length * 3);
  for (let i = 0; i < trailPositions.length; i++) {
    const t = i / (trailPositions.length - 1); // 0 = oldest, 1 = newest
    // Cyan fading to transparent blue
    colors[i * 3]     = 0.1;            // R
    colors[i * 3 + 1] = 0.6 + 0.3 * t; // G
    colors[i * 3 + 2] = 1.0;            // B
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.5,
    linewidth: 1,
  });
  trailLine = new THREE.Line(geo, mat);
  trailGroup.add(trailLine);
}

/**
 * Clear the trail (e.g., on episode reset).
 */
export function clearTrail() {
  trailPositions = [];
  while (trailGroup.children.length) {
    const c = trailGroup.children[0];
    trailGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

function createGrid() {
  gridHelper = new THREE.GridHelper(400, 40, 0x1a2a44, 0x0f1828);
  gridHelper.rotation.x = Math.PI / 2; // align to XY
  gridHelper.position.z = -2;
  gridHelper.visible = false;
  scene.add(gridHelper);
}

function toggleGrid() {
  if (!gridHelper) return;
  gridVisible = !gridVisible;
  gridHelper.visible = gridVisible;
}
