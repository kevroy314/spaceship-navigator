/**
 * main.js — Entry point for Spaceship Navigator frontend.
 *
 * Sets up the Three.js renderer with a 2D orthographic camera looking down
 * at the XY plane, initialises all modules, and runs the animation loop.
 */

import * as THREE from 'three';
import { initScene, updateField, updateBodies, updateShip, updateTrail, getCamera, resizeRenderer } from './renderer.js';
import { initPlayer, destroyPlayer, getPlayerState } from './player.js';
import { initReplay, destroyReplay, tickReplay, getReplayFrame } from './replay.js';
import { initUI, setMode, updateStats, updateFuelBar } from './ui.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {'play' | 'replay'} */
let currentMode = 'play';
let clock = new THREE.Clock();

// Three.js core objects – created once
let renderer, scene, camera;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

function init() {
  // --- Three.js setup ---
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x0a0a12, 1);
  document.body.prepend(renderer.domElement);

  // Orthographic camera: view centred on origin, world units visible
  const aspect = window.innerWidth / window.innerHeight;
  const viewSize = 100; // half-extent in world units
  camera = new THREE.OrthographicCamera(
    -viewSize * aspect,
     viewSize * aspect,
     viewSize,
    -viewSize,
    0.1,
    1000,
  );
  camera.position.set(0, 0, 100);
  camera.lookAt(0, 0, 0);

  scene = new THREE.Scene();

  // --- Module init ---
  initScene(scene, camera, renderer);
  initUI(onModeChange, onConfigChange);

  // Start in play mode
  switchToPlay();

  // --- Resize handler ---
  window.addEventListener('resize', onResize);

  // --- Animation loop ---
  animate();
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

function onModeChange(mode) {
  if (mode === currentMode) return;
  if (mode === 'play') {
    switchToPlay();
  } else {
    switchToReplay();
  }
}

function switchToPlay() {
  currentMode = 'play';
  destroyReplay();
  initPlayer(onPlayStateUpdate);
  setMode('play');
}

function switchToReplay() {
  currentMode = 'replay';
  destroyPlayer();
  initReplay();
  setMode('replay');
}

function onConfigChange(_configId) {
  // When the config changes we may want to re-initialise the current mode.
  if (currentMode === 'replay') {
    destroyReplay();
    initReplay();
  }
}

// ---------------------------------------------------------------------------
// State update callback from player WebSocket
// ---------------------------------------------------------------------------

function onPlayStateUpdate(state) {
  if (!state) return;

  if (state.field) {
    updateField(state.field);
  }
  if (state.bodies) {
    updateBodies(state.bodies);
  }
  if (state.ship) {
    updateShip(state.ship);
    updateTrail(state.ship);

    // Update HUD
    updateStats({
      fuel: state.ship.fuel,
      velocity: Math.hypot(state.ship.vx ?? 0, state.ship.vy ?? 0),
      distance: state.ship.distance_to_target,
      time: state.ship.time_elapsed,
      reward: state.ship.reward,
      mass: state.ship.mass,
    });

    // Fuel bar: fraction of remaining fuel
    const fuelFrac = state.ship.max_fuel
      ? state.ship.fuel / state.ship.max_fuel
      : 1;
    updateFuelBar(fuelFrac);
  }
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);

  const dt = clock.getDelta();

  if (currentMode === 'replay') {
    tickReplay(dt);
    const frame = getReplayFrame();
    if (frame) {
      onPlayStateUpdate(frame);
    }
  }

  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);

  const aspect = w / h;
  const viewSize = 100;
  camera.left   = -viewSize * aspect;
  camera.right  =  viewSize * aspect;
  camera.top    =  viewSize;
  camera.bottom = -viewSize;
  camera.updateProjectionMatrix();

  resizeRenderer(w, h);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init();
