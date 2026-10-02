/**
 * replay.js — Replay viewer for training episodes.
 *
 * Fetches episode data from the REST API and provides playback controls:
 *   - Play / pause
 *   - Speed control (0.5x to 10x)
 *   - Step forward / back
 *   - Timeline scrubbing
 *   - Training iteration browsing
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SPEED_LEVELS = [0.5, 1, 2, 4, 6, 8, 10];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let active = false;

/** Currently loaded episode frames */
let frames = [];

/** Playback state */
let playing = false;
let speedIndex = 1; // index into SPEED_LEVELS
let currentFrame = 0;
let accumulator = 0; // fractional frame accumulator
let frameDuration = 1 / 20; // seconds per frame at 1x speed (assume 20 fps sim)

/** Episode metadata */
let episodeMeta = null;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function initReplay() {
  active = true;
  frames = [];
  currentFrame = 0;
  playing = false;
  accumulator = 0;
  speedIndex = 1;

  attachReplayListeners();
  updateSpeedDisplay();
}

export function destroyReplay() {
  active = false;
  frames = [];
  playing = false;
  detachReplayListeners();
}

/**
 * Called from the animation loop with delta time in seconds.
 */
export function tickReplay(dt) {
  if (!active || !playing || frames.length === 0) return;

  const speed = SPEED_LEVELS[speedIndex] ?? 1;
  accumulator += dt * speed;

  while (accumulator >= frameDuration && currentFrame < frames.length - 1) {
    accumulator -= frameDuration;
    currentFrame++;
  }

  // Pause at end
  if (currentFrame >= frames.length - 1) {
    playing = false;
    updatePlayPauseButton();
  }

  updateScrubSlider();
}

/**
 * Returns the current replay frame for rendering, or null if nothing loaded.
 */
export function getReplayFrame() {
  if (!active || frames.length === 0) return null;
  return frames[currentFrame] ?? null;
}

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------

/**
 * Fetch list of available configs from the API.
 * @returns {Promise<Array<{id: string, name: string}>>}
 */
export async function fetchConfigs() {
  try {
    const res = await fetch('/api/configs');
    if (!res.ok) return [];
    const data = await res.json();
    // Normalise: API returns { config_hash, ... }, UI expects { id, name }
    return data.map((c) => ({
      id: c.config_hash,
      name: c.metadata?.name ?? c.config_hash,
      ...c,
    }));
  } catch (err) {
    console.warn('[replay] Failed to fetch configs:', err);
    return [];
  }
}

/**
 * Fetch runs for a config.
 */
export async function fetchRuns(configId) {
  try {
    const res = await fetch(`/api/configs/${configId}/iterations`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.map((r) => ({
      id: r.iteration,
      ...r,
    }));
  } catch (err) {
    console.warn('[replay] Failed to fetch runs:', err);
    return [];
  }
}

/**
 * Fetch and load a specific episode.
 */
export async function loadEpisode(configId, runId, episodeId) {
  try {
    const res = await fetch(
      `/api/configs/${configId}/iterations/${runId}/episodes/${episodeId}`
    );
    if (!res.ok) {
      console.warn('[replay] Episode fetch failed:', res.status);
      return false;
    }

    const data = await res.json();
    frames = data.frames ?? [];
    episodeMeta = data.meta ?? null;
    currentFrame = 0;
    accumulator = 0;
    playing = false;

    // Update UI
    updatePlayPauseButton();
    updateScrubSlider();
    updateIterationDisplay();

    console.log(`[replay] Loaded episode: ${frames.length} frames`);
    return true;
  } catch (err) {
    console.warn('[replay] Failed to load episode:', err);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Playback controls
// ---------------------------------------------------------------------------

function play() {
  if (frames.length === 0) return;
  if (currentFrame >= frames.length - 1) {
    currentFrame = 0; // restart if at end
  }
  playing = true;
  updatePlayPauseButton();
}

function pause() {
  playing = false;
  updatePlayPauseButton();
}

function togglePlayPause() {
  playing ? pause() : play();
}

function stepForward() {
  pause();
  if (currentFrame < frames.length - 1) {
    currentFrame++;
    updateScrubSlider();
  }
}

function stepBack() {
  pause();
  if (currentFrame > 0) {
    currentFrame--;
    updateScrubSlider();
  }
}

function setSpeed(index) {
  speedIndex = Math.max(0, Math.min(SPEED_LEVELS.length - 1, index));
  updateSpeedDisplay();
}

function scrubTo(fraction) {
  if (frames.length === 0) return;
  currentFrame = Math.round(fraction * (frames.length - 1));
  accumulator = 0;
}

// ---------------------------------------------------------------------------
// UI bindings
// ---------------------------------------------------------------------------

function onPlayPauseClick() {
  togglePlayPause();
}

function onStepBackClick() {
  stepBack();
}

function onStepFwdClick() {
  stepForward();
}

function onSpeedSliderInput(e) {
  setSpeed(parseInt(e.target.value, 10));
}

function onScrubSliderInput(e) {
  const val = parseInt(e.target.value, 10);
  const max = parseInt(e.target.max, 10) || 1;
  scrubTo(val / max);
}

// Listeners
let listenersAttached = false;

function attachReplayListeners() {
  if (listenersAttached) return;
  listenersAttached = true;

  const btnPlayPause = document.getElementById('btn-play-pause');
  const btnStepBack  = document.getElementById('btn-step-back');
  const btnStepFwd   = document.getElementById('btn-step-fwd');
  const speedSlider  = document.getElementById('speed-slider');
  const scrubSlider  = document.getElementById('scrub-slider');

  btnPlayPause?.addEventListener('click', onPlayPauseClick);
  btnStepBack?.addEventListener('click', onStepBackClick);
  btnStepFwd?.addEventListener('click', onStepFwdClick);
  speedSlider?.addEventListener('input', onSpeedSliderInput);
  scrubSlider?.addEventListener('input', onScrubSliderInput);
}

function detachReplayListeners() {
  if (!listenersAttached) return;
  listenersAttached = false;

  const btnPlayPause = document.getElementById('btn-play-pause');
  const btnStepBack  = document.getElementById('btn-step-back');
  const btnStepFwd   = document.getElementById('btn-step-fwd');
  const speedSlider  = document.getElementById('speed-slider');
  const scrubSlider  = document.getElementById('scrub-slider');

  btnPlayPause?.removeEventListener('click', onPlayPauseClick);
  btnStepBack?.removeEventListener('click', onStepBackClick);
  btnStepFwd?.removeEventListener('click', onStepFwdClick);
  speedSlider?.removeEventListener('input', onSpeedSliderInput);
  scrubSlider?.removeEventListener('input', onScrubSliderInput);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function updatePlayPauseButton() {
  const btn = document.getElementById('btn-play-pause');
  if (btn) btn.textContent = playing ? 'Pause' : 'Play';
}

function updateSpeedDisplay() {
  const el = document.getElementById('speed-display');
  const slider = document.getElementById('speed-slider');
  const speed = SPEED_LEVELS[speedIndex] ?? 1;
  if (el) el.textContent = `${speed}x`;
  if (slider) slider.value = speedIndex;
}

function updateScrubSlider() {
  const slider = document.getElementById('scrub-slider');
  if (!slider) return;
  const max = parseInt(slider.max, 10) || 1;
  const frac = frames.length > 1 ? currentFrame / (frames.length - 1) : 0;
  slider.value = Math.round(frac * max);
}

function updateIterationDisplay() {
  const el = document.getElementById('iteration-value');
  if (el && episodeMeta) {
    el.textContent = episodeMeta.iteration ?? '--';
  }
}
