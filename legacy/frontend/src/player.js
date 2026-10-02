/**
 * player.js — Player input handling and WebSocket communication for "play" mode.
 *
 * Keyboard controls:
 *   W / ArrowUp    — main thrust (forward)
 *   A / ArrowLeft  — rotate left
 *   D / ArrowRight — rotate right
 *
 * Sends action frames to the backend via WebSocket as JSON:
 *   { main_thrust: 0-1, rotation_thrust: -1 to 1 }
 *
 * Receives simulation state updates from the backend.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let stateCallback = null;
let connected = false;

// Input state
const keys = {
  thrust: false,
  rotateLeft: false,
  rotateRight: false,
};

let inputLoopId = null;
const INPUT_RATE_MS = 50; // 20 Hz input tick

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Start the player mode — opens a WebSocket and begins sending input.
 * @param {function} onStateUpdate — called with each state frame from the server.
 */
export function initPlayer(onStateUpdate) {
  stateCallback = onStateUpdate;
  connectWebSocket();
  addKeyListeners();
  startInputLoop();
}

/**
 * Tear down player mode — close WS, remove listeners.
 */
export function destroyPlayer() {
  stopInputLoop();
  removeKeyListeners();
  closeWebSocket();
  stateCallback = null;
  resetKeys();
}

/**
 * Returns the current player input state (useful for UI display).
 */
export function getPlayerState() {
  return {
    connected,
    thrust: keys.thrust,
    rotateLeft: keys.rotateLeft,
    rotateRight: keys.rotateRight,
  };
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function getWSUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/play`;
}

function connectWebSocket() {
  if (ws) closeWebSocket();

  try {
    ws = new WebSocket(getWSUrl());
  } catch (err) {
    console.warn('[player] WebSocket creation failed:', err);
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    connected = true;
    updateConnectionUI(true);
    console.log('[player] WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === 'status') {
        console.log('[player] Server:', msg.message);
        return;
      }

      if (msg.type === 'state' && msg.data && stateCallback) {
        // Transform server state into the format renderer expects
        const data = msg.data;
        const ds = data.domain_size || 1;

        // Scale positions from meters to world units
        // Use domain_size to normalise: map [0, domain_size] → [-100, 100]
        const toWorld = (pos) => [(pos[0] / ds) * 200 - 100, (pos[1] / ds) * 200 - 100];

        const shipWorld = toWorld(data.ship.pos);
        const frame = {
          field: data.potential_field ? {
            width: data.potential_field[0]?.length ?? 0,
            height: data.potential_field.length,
            values: data.potential_field.flat(),
            min: Math.min(...data.potential_field.flat()),
            max: Math.max(...data.potential_field.flat()),
            extent: [-100, -100, 100, 100],
          } : null,
          bodies: data.bodies ? data.bodies.positions.map((pos, i) => {
            const w = toWorld(pos);
            const mass = data.bodies.masses[i];
            const isTarget = data.bodies.is_target[i];
            let type = 'planet';
            if (i === 0) type = 'star';
            if (isTarget) type = 'target';
            if (mass < 1e24) type = 'asteroid';
            return {
              id: i, type, x: w[0], y: w[1], mass,
              radius: Math.max(0.5, Math.log10(mass + 1) * 0.8 - 18),
              approach_radius: isTarget ? (data.metrics?.approach_radius ?? 5) : undefined,
            };
          }) : [],
          ship: {
            x: shipWorld[0],
            y: shipWorld[1],
            vx: data.ship.vel[0],
            vy: data.ship.vel[1],
            heading: data.ship.heading - Math.PI / 2, // adjust for renderer
            fuel: data.ship.fuel_mass,
            max_fuel: data.ship.total_mass - data.ship.dry_mass,
            mass: data.ship.total_mass,
            distance_to_target: data.metrics?.distance_to_target ?? 0,
            relative_velocity: data.metrics?.relative_velocity ?? 0,
            reward: data.metrics?.cumulative_reward ?? 0,
            time_elapsed: data.step ?? 0,
          },
        };
        stateCallback(frame);
      }
    } catch (err) {
      console.warn('[player] Failed to parse message:', err);
    }
  };

  ws.onclose = () => {
    connected = false;
    updateConnectionUI(false);
    console.log('[player] WebSocket closed');
    scheduleReconnect();
  };

  ws.onerror = (err) => {
    console.warn('[player] WebSocket error:', err);
    connected = false;
    updateConnectionUI(false);
  };
}

function closeWebSocket() {
  if (ws) {
    ws.onclose = null; // prevent reconnect
    ws.close();
    ws = null;
  }
  connected = false;
  updateConnectionUI(false);
}

let reconnectTimeout = null;

function scheduleReconnect() {
  if (reconnectTimeout) return;
  reconnectTimeout = setTimeout(() => {
    reconnectTimeout = null;
    if (!connected && stateCallback) {
      connectWebSocket();
    }
  }, 2000);
}

function updateConnectionUI(isConnected) {
  const dot = document.getElementById('connection-dot');
  const text = document.getElementById('connection-text');
  if (dot) {
    dot.classList.toggle('connected', isConnected);
  }
  if (text) {
    text.textContent = isConnected ? 'Connected' : 'Disconnected';
  }
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

function onKeyDown(e) {
  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      keys.thrust = true;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      keys.rotateLeft = true;
      break;
    case 'KeyD':
    case 'ArrowRight':
      keys.rotateRight = true;
      break;
  }
}

function onKeyUp(e) {
  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      keys.thrust = false;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      keys.rotateLeft = false;
      break;
    case 'KeyD':
    case 'ArrowRight':
      keys.rotateRight = false;
      break;
  }
}

function addKeyListeners() {
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
}

function removeKeyListeners() {
  window.removeEventListener('keydown', onKeyDown);
  window.removeEventListener('keyup', onKeyUp);
}

function resetKeys() {
  keys.thrust = false;
  keys.rotateLeft = false;
  keys.rotateRight = false;
}

// ---------------------------------------------------------------------------
// Input sending loop
// ---------------------------------------------------------------------------

function startInputLoop() {
  if (inputLoopId) return;
  inputLoopId = setInterval(sendInput, INPUT_RATE_MS);
}

function stopInputLoop() {
  if (inputLoopId) {
    clearInterval(inputLoopId);
    inputLoopId = null;
  }
}

function sendInput() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const mainThrust = keys.thrust ? 1.0 : 0.0;
  let rotationThrust = 0;
  if (keys.rotateLeft) rotationThrust -= 1;
  if (keys.rotateRight) rotationThrust += 1;

  const message = {
    type: 'action',
    action: {
      main_thrust: mainThrust,
      rotation_thrust: rotationThrust,
    },
  };

  ws.send(JSON.stringify(message));
}

// ---------------------------------------------------------------------------
// Gamepad support (basic — stretch goal)
// ---------------------------------------------------------------------------

function pollGamepad() {
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  const gp = gamepads[0];
  if (!gp) return null;

  // Standard mapping: left stick X for rotation, right trigger for thrust
  const deadzone = 0.15;
  let rotation = gp.axes[0] ?? 0;
  if (Math.abs(rotation) < deadzone) rotation = 0;

  let thrust = gp.buttons[7]?.value ?? 0; // right trigger

  return { main_thrust: thrust, rotation_thrust: rotation };
}

// Gamepad can be integrated into sendInput in the future by merging with key state.
