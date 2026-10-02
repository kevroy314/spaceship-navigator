/**
 * ui.js — UI controller.
 *
 * Manages the overlay panel: mode toggle, config dropdown, stats display,
 * fuel bar, and replay control visibility.
 */

import { fetchConfigs, fetchRuns, loadEpisode } from './replay.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let modeCallback = null;
let configCallback = null;
let configs = [];
let currentConfigId = null;
let currentRuns = [];
let currentRunId = null;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialise the UI: wire up buttons, populate the config dropdown.
 *
 * @param {function} onModeChange  — called with 'play' | 'replay'
 * @param {function} onConfigChange — called with configId string
 */
export function initUI(onModeChange, onConfigChange) {
  modeCallback = onModeChange;
  configCallback = onConfigChange;

  // Mode toggle buttons
  const btnPlay   = document.getElementById('btn-play');
  const btnReplay = document.getElementById('btn-replay');

  btnPlay?.addEventListener('click', () => {
    if (modeCallback) modeCallback('play');
  });

  btnReplay?.addEventListener('click', () => {
    if (modeCallback) modeCallback('replay');
  });

  // Config selector
  const configSelect = document.getElementById('config-select');
  configSelect?.addEventListener('change', (e) => {
    const id = e.target.value;
    if (id) {
      currentConfigId = id;
      if (configCallback) configCallback(id);
      onConfigSelected(id);
    }
  });

  // Iteration slider
  const iterSlider = document.getElementById('iteration-slider');
  iterSlider?.addEventListener('change', onIterationChange);

  // Episode selector
  const epSelect = document.getElementById('episode-select');
  epSelect?.addEventListener('change', onEpisodeChange);

  // Populate configs from API
  populateConfigs();
}

/**
 * Switch visible UI state between play and replay.
 */
export function setMode(mode) {
  const btnPlay   = document.getElementById('btn-play');
  const btnReplay = document.getElementById('btn-replay');
  const replayCtl = document.getElementById('replay-controls');

  if (mode === 'play') {
    btnPlay?.classList.add('active');
    btnReplay?.classList.remove('active');
    replayCtl?.classList.remove('visible');
  } else {
    btnPlay?.classList.remove('active');
    btnReplay?.classList.add('active');
    replayCtl?.classList.add('visible');
  }
}

/**
 * Update the stats display panel.
 *
 * @param {{ fuel, velocity, distance, time, reward, mass }} stats
 */
export function updateStats(stats) {
  setStatText('stat-fuel', formatNum(stats.fuel));
  setStatText('stat-velocity', formatNum(stats.velocity));
  setStatText('stat-distance', formatNum(stats.distance));
  setStatText('stat-time', formatNum(stats.time, 1));
  setStatText('stat-reward', formatNum(stats.reward, 2));
  setStatText('stat-mass', formatNum(stats.mass));
}

/**
 * Update the fuel bar width and color.
 * @param {number} fraction — 0 to 1
 */
export function updateFuelBar(fraction) {
  const bar = document.getElementById('fuel-bar-inner');
  if (!bar) return;
  const pct = Math.max(0, Math.min(100, fraction * 100));
  bar.style.width = `${pct}%`;
}

// ---------------------------------------------------------------------------
// Config / run / episode population
// ---------------------------------------------------------------------------

async function populateConfigs() {
  const select = document.getElementById('config-select');
  if (!select) return;

  configs = await fetchConfigs();

  select.innerHTML = '';

  if (configs.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No configs available';
    select.appendChild(opt);
    return;
  }

  for (const cfg of configs) {
    const opt = document.createElement('option');
    opt.value = cfg.id;
    opt.textContent = cfg.name ?? cfg.id;
    select.appendChild(opt);
  }

  // Auto-select first
  currentConfigId = configs[0].id;
  select.value = currentConfigId;
  if (configCallback) configCallback(currentConfigId);
  onConfigSelected(currentConfigId);
}

async function onConfigSelected(configId) {
  currentRuns = await fetchRuns(configId);

  // Update iteration slider range
  const slider = document.getElementById('iteration-slider');
  const iterVal = document.getElementById('iteration-value');
  if (slider && currentRuns.length > 0) {
    slider.min = 0;
    slider.max = currentRuns.length - 1;
    slider.value = currentRuns.length - 1; // default to latest
    if (iterVal) iterVal.textContent = currentRuns.length - 1;
  }

  // Load episodes for latest run
  if (currentRuns.length > 0) {
    currentRunId = currentRuns[currentRuns.length - 1].id ?? currentRuns.length - 1;
    populateEpisodes(currentRunId);
  }
}

async function onIterationChange(e) {
  const idx = parseInt(e.target.value, 10);
  const iterVal = document.getElementById('iteration-value');
  if (iterVal) iterVal.textContent = idx;

  if (currentRuns[idx]) {
    currentRunId = currentRuns[idx].id ?? idx;
    populateEpisodes(currentRunId);
  }
}

function populateEpisodes(runId) {
  const select = document.getElementById('episode-select');
  if (!select) return;

  // We'll find episodes from the run data, or default to a placeholder
  const run = currentRuns.find((r) => (r.id ?? r) === runId);
  const episodes = run?.episodes ?? [];

  select.innerHTML = '';

  if (episodes.length === 0) {
    // Add a single default option — server will fill in
    const opt = document.createElement('option');
    opt.value = '0';
    opt.textContent = 'Episode 0';
    select.appendChild(opt);
  } else {
    for (let i = 0; i < episodes.length; i++) {
      const ep = episodes[i];
      const opt = document.createElement('option');
      opt.value = ep.id ?? i;
      opt.textContent = `Episode ${ep.id ?? i}`;
      select.appendChild(opt);
    }
  }

  // Auto-load first episode
  const firstEpId = select.value;
  if (currentConfigId && currentRunId != null && firstEpId) {
    loadEpisode(currentConfigId, currentRunId, firstEpId);
  }
}

function onEpisodeChange(e) {
  const epId = e.target.value;
  if (currentConfigId && currentRunId != null && epId) {
    loadEpisode(currentConfigId, currentRunId, epId);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setStatText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function formatNum(val, decimals = 1) {
  if (val == null || val === undefined) return '--';
  if (typeof val !== 'number') return String(val);
  return val.toFixed(decimals);
}
