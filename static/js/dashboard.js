/* ═══════════════════════════════════════════════════════════════
   AUTONOMOUS DRONE RL — MISSION CONTROL
   dashboard.js — Complete frontend logic
   Palette: Aerospace Slate + Amber Signal
════════════════════════════════════════════════════════════════ */

'use strict';

// ── DESIGN TOKENS (mirrored for canvas rendering) ────────────────
const C = {
  void:     '#080c14',
  panel:    '#0d1420',
  border:   '#1a2540',
  surface:  '#1c2a3e',
  surface2: '#243450',
  amber:    '#e8a020',
  amberDim: '#7a4c08',
  ice:      '#a8c8e8',
  signal:   '#3dc8a0',
  hazard:   '#e05040',
  text:     '#d8e4f0',
  subtext:  '#506080',
};

// ── CONFIG ────────────────────────────────────────────────────────
const CFG = {
  GRID:          10,
  ADV_WORLD:     20.0,
  TRAIL_CAP:     55,
  CURVE_CAP:     200,
  POLL_STATE_MS: 1000,
  POLL_CURR_MS:  2500,
  POLL_CURVE_MS: 5000,
  RECONNECT_MS:  3500,
  LIDAR_MAX_FAC: Math.SQRT2,
  GOAL_FLASH_MS: 1100,
};

// ── APP STATE ─────────────────────────────────────────────────────
const APP = {
  mode:       'coord',
  simEs:      null,
  lastFrame:  null,
  trail:      [],
  prevVals:   {},
  wiObs:      [],       // user-placed what-if obstacles (world coords)
  wiResult:   null,
  metrics:    {},
  simDims:    { w: 400, h: 400, cell: 40, scale: 1 },
  reconnTimer: null,
};

// ── DOM SHORTHAND ─────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── CANVAS CONTEXTS ───────────────────────────────────────────────
const simCv   = $('sim-canvas');    const simCx   = simCv.getContext('2d');
const lidarCv = $('lidar-canvas');  const lidarCx = lidarCv.getContext('2d');
const curveCv = $('curve-canvas');  const curveCx = curveCv.getContext('2d');
const windCv  = $('wind-canvas');   const windCx  = windCv.getContext('2d');
const wiCv    = $('wi-canvas');     const wiCx    = wiCv.getContext('2d');

// ═══════════════════════════════════════════════════════════════════
// CANVAS SIZING — DPI-aware, resize-responsive
// ═══════════════════════════════════════════════════════════════════

function scaleCanvas(cv, w, h) {
  const dpr = window.devicePixelRatio || 1;
  cv.width  = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width  = w + 'px';
  cv.style.height = h + 'px';
  const cx = cv.getContext('2d');
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Compute logical sim dimensions from container width
function computeSimDims(containerW) {
  if (APP.mode === 'advanced') {
    const w = Math.min(containerW, 600);
    return { w, h: w, cell: 0, scale: w / CFG.ADV_WORLD };
  } else {
    const cell = Math.max(30, Math.floor(containerW / CFG.GRID));
    const size = cell * CFG.GRID;
    return { w: size, h: size, cell, scale: cell };
  }
}

function resizeSim() {
  const wrap = $('sim-wrap');
  if (!wrap) return;
  const cw = wrap.clientWidth;
  const dims = computeSimDims(cw);
  APP.simDims = dims;
  scaleCanvas(simCv, dims.w, dims.h);
  scaleCanvas(wiCv,  dims.w, dims.h);
  simCv.style.height = dims.h + 'px';
  wiCv.style.height  = dims.h + 'px';
  if (APP.lastFrame) drawSim(APP.lastFrame);
  else               drawSimIdle();
  redrawWhatIf();
}

new ResizeObserver(() => resizeSim()).observe($('sim-wrap'));

function resizeCurve() {
  const wrap = $('curve-wrap');
  if (!wrap) return;
  scaleCanvas(curveCv, wrap.clientWidth, 200);
  drawCurve();
}
new ResizeObserver(() => resizeCurve()).observe($('curve-wrap'));

// ═══════════════════════════════════════════════════════════════════
// TELEMETRY PULSE — amber phosphor flicker on value change
// ═══════════════════════════════════════════════════════════════════

function setTelem(id, val, raw) {
  const el = $(id);
  if (!el) return;
  const str = String(val);
  if (APP.prevVals[id] !== str) {
    el.textContent = str;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) {
      el.classList.remove('pulsing');
      void el.offsetWidth;            // reflow to restart animation
      el.classList.add('pulsing');
      el.addEventListener('animationend', () => el.classList.remove('pulsing'), { once: true });
    }
    APP.prevVals[id] = str;
  }
  // Mirror to mobile bar
  if (id === 't-episode') $('mb-ep').textContent = 'EP ' + str;
  if (id === 't-reward')  $('mb-rew').textContent = str;
}

// ═══════════════════════════════════════════════════════════════════
// DRAW HELPERS
// ═══════════════════════════════════════════════════════════════════

function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

function arrow(cx, x1, y1, x2, y2, color, lw = 1.5, headLen = 7) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  cx.save();
  cx.strokeStyle = color; cx.fillStyle = color; cx.lineWidth = lw;
  cx.beginPath(); cx.moveTo(x1, y1); cx.lineTo(x2, y2); cx.stroke();
  cx.beginPath();
  cx.moveTo(x2, y2);
  cx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
  cx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
  cx.closePath(); cx.fill();
  cx.restore();
}

// ═══════════════════════════════════════════════════════════════════
// TRAJECTORY TRAIL
// ═══════════════════════════════════════════════════════════════════

function pushTrail(x, y) {
  APP.trail.push({ x, y });
  if (APP.trail.length > CFG.TRAIL_CAP) APP.trail.shift();
}

function drawTrail(cx, trail, toCanvasX, toCanvasY) {
  if (trail.length < 2) return;
  cx.save();
  for (let i = 1; i < trail.length; i++) {
    const t = i / trail.length;
    const p0 = trail[i - 1], p1 = trail[i];
    cx.beginPath();
    cx.moveTo(toCanvasX(p0.x), toCanvasY(p0.y));
    cx.lineTo(toCanvasX(p1.x), toCanvasY(p1.y));
    cx.strokeStyle = hexAlpha(C.ice, t * 0.5);
    cx.lineWidth   = 0.8 + t * 0.8;
    cx.stroke();
  }
  // Head dot
  const last = trail[trail.length - 1];
  cx.beginPath();
  cx.arc(toCanvasX(last.x), toCanvasY(last.y), 2.5, 0, Math.PI * 2);
  cx.fillStyle = hexAlpha(C.ice, 0.55);
  cx.fill();
  cx.restore();
}

// ═══════════════════════════════════════════════════════════════════
// DRAW DRONE SHAPE (quad-rotor top-down)
// ═══════════════════════════════════════════════════════════════════

function drawDrone(cx, px, py, radius, headingAngle) {
  const armR = radius * 1.05;
  cx.save();
  cx.translate(px, py);
  cx.rotate(headingAngle + Math.PI / 4); // align arm to heading

  // 4 arms + rotor discs
  for (let i = 0; i < 4; i++) {
    const a = i * Math.PI / 2;
    const ex = Math.cos(a) * armR, ey = Math.sin(a) * armR;
    cx.strokeStyle = C.ice; cx.lineWidth = 1.4;
    cx.beginPath(); cx.moveTo(0, 0); cx.lineTo(ex, ey); cx.stroke();
    cx.beginPath(); cx.arc(ex, ey, 3.5, 0, Math.PI * 2);
    cx.fillStyle = hexAlpha(C.ice, 0.28); cx.fill();
    cx.strokeStyle = hexAlpha(C.ice, 0.55); cx.lineWidth = 0.8; cx.stroke();
  }

  // Body
  const bg = cx.createRadialGradient(0, 0, 0, 0, 0, radius * 0.5);
  bg.addColorStop(0, '#4ab4e8'); bg.addColorStop(1, '#1a3a56');
  cx.beginPath(); cx.arc(0, 0, radius * 0.5, 0, Math.PI * 2);
  cx.fillStyle = bg; cx.fill();
  cx.strokeStyle = hexAlpha(C.ice, 0.7); cx.lineWidth = 1; cx.stroke();

  cx.restore();

  // Heading indicator (amber, not rotated with body)
  cx.save(); cx.translate(px, py); cx.rotate(headingAngle);
  cx.strokeStyle = C.amber; cx.lineWidth = 2;
  cx.beginPath(); cx.moveTo(0, 0); cx.lineTo(0, -radius * 1.15); cx.stroke();
  cx.beginPath();
  cx.moveTo(0, -radius * 1.15);
  cx.lineTo(-2.5, -radius * 0.9); cx.lineTo(2.5, -radius * 0.9);
  cx.closePath(); cx.fillStyle = C.amber; cx.fill();
  cx.restore();
}

// ═══════════════════════════════════════════════════════════════════
// GOAL GLOW
// ═══════════════════════════════════════════════════════════════════

function drawGoal(cx, gx, gy, r, reached) {
  const glow = cx.createRadialGradient(gx, gy, 0, gx, gy, r * 3);
  glow.addColorStop(0, hexAlpha(C.signal, reached ? 0.5 : 0.25));
  glow.addColorStop(1, hexAlpha(C.signal, 0));
  cx.beginPath(); cx.arc(gx, gy, r * 3, 0, Math.PI * 2);
  cx.fillStyle = glow; cx.fill();

  cx.beginPath(); cx.arc(gx, gy, r, 0, Math.PI * 2);
  cx.strokeStyle = reached ? C.signal : hexAlpha(C.signal, 0.7);
  cx.lineWidth = reached ? 2.5 : 1.5; cx.stroke();

  cx.fillStyle = C.signal;
  cx.font = `bold ${Math.max(8, r * 0.7)}px 'Space Grotesk', sans-serif`;
  cx.textAlign = 'center'; cx.textBaseline = 'middle';
  cx.fillText('G', gx, gy);
}

// ═══════════════════════════════════════════════════════════════════
// IDLE STATE (no SSE running)
// ═══════════════════════════════════════════════════════════════════

function drawSimIdle() {
  const { w, h } = APP.simDims;
  simCx.clearRect(0, 0, w, h);
  simCx.fillStyle = '#060810';
  simCx.fillRect(0, 0, w, h);
  // Subtle grid
  simCx.strokeStyle = hexAlpha(C.border, 0.4);
  simCx.lineWidth = 0.5;
  const step = APP.mode === 'advanced' ? (w / CFG.ADV_WORLD) * 2 : APP.simDims.cell;
  for (let x = 0; x <= w; x += step) {
    simCx.beginPath(); simCx.moveTo(x, 0); simCx.lineTo(x, h); simCx.stroke();
  }
  for (let y = 0; y <= h; y += step) {
    simCx.beginPath(); simCx.moveTo(0, y); simCx.lineTo(w, y); simCx.stroke();
  }
  // Center label
  simCx.fillStyle = hexAlpha(C.subtext, 0.5);
  simCx.font = "500 0.7rem 'Space Grotesk', sans-serif";
  simCx.textAlign = 'center'; simCx.textBaseline = 'middle';
  simCx.fillText('SELECT MODE AND PRESS ▶ START', w / 2, h / 2);
}

// ═══════════════════════════════════════════════════════════════════
// GRID SIM RENDERER (coord / lidar modes)
// ═══════════════════════════════════════════════════════════════════

const ACTION_HEADINGS = [-Math.PI / 2, Math.PI / 2, Math.PI, 0]; // up down left right
const ACTION_NAMES    = ['↑ UP', '↓ DOWN', '← LEFT', '→ RIGHT'];

let _flashTimer = null;
function showGoalFlash() {
  const el = $('goal-flash');
  el.style.display = 'block';
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(() => { el.style.display = 'none'; }, CFG.GOAL_FLASH_MS);
}

function drawGridSim(data) {
  const { cell, w, h } = APP.simDims;
  simCx.clearRect(0, 0, w, h);

  // Background
  simCx.fillStyle = '#060810'; simCx.fillRect(0, 0, w, h);

  // Grid lines
  simCx.strokeStyle = hexAlpha(C.border, 0.7); simCx.lineWidth = 0.5;
  for (let i = 0; i <= CFG.GRID; i++) {
    simCx.beginPath(); simCx.moveTo(i * cell, 0); simCx.lineTo(i * cell, h); simCx.stroke();
    simCx.beginPath(); simCx.moveTo(0, i * cell); simCx.lineTo(w, i * cell); simCx.stroke();
  }

  // Obstacles
  for (const [ox, oy] of (data.obstacles || [])) {
    const px = ox * cell, py = oy * cell;
    simCx.fillStyle = hexAlpha(C.hazard, 0.18);
    simCx.fillRect(px, py, cell, cell);
    simCx.strokeStyle = hexAlpha(C.hazard, 0.55);
    simCx.lineWidth = 1;
    simCx.strokeRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
    // Hazard cross
    simCx.strokeStyle = hexAlpha(C.hazard, 0.35); simCx.lineWidth = 1;
    simCx.beginPath();
    simCx.moveTo(px + 6, py + 6); simCx.lineTo(px + cell - 6, py + cell - 6); simCx.stroke();
    simCx.beginPath();
    simCx.moveTo(px + cell - 6, py + 6); simCx.lineTo(px + 6, py + cell - 6); simCx.stroke();
  }

  // LIDAR rays (lidar mode)
  if (data.rays) {
    const [dx, dy] = data.drone;
    const dcx = dx * cell + cell / 2, dcy = dy * cell + cell / 2;
    simCx.save(); simCx.globalAlpha = 0.45; simCx.strokeStyle = '#d4b040'; simCx.lineWidth = 1;
    for (const [rx, ry] of data.rays) {
      simCx.beginPath();
      simCx.moveTo(dcx, dcy);
      simCx.lineTo(rx * cell + cell / 2, ry * cell + cell / 2);
      simCx.stroke();
      simCx.globalAlpha = 0.7;
      simCx.fillStyle = C.amber;
      simCx.beginPath();
      simCx.arc(rx * cell + cell / 2, ry * cell + cell / 2, 3, 0, Math.PI * 2);
      simCx.fill();
      simCx.globalAlpha = 0.45;
    }
    simCx.restore();
  }

  // Trail
  const toGX = x => x * cell + cell / 2;
  drawTrail(simCx, APP.trail, toGX, toGX);

  // Goal
  const [gx, gy] = data.goal;
  drawGoal(simCx, gx * cell + cell / 2, gy * cell + cell / 2, cell * 0.36, data.goal_reached);

  // Drone
  const [drx, dry] = data.drone;
  const heading = ACTION_HEADINGS[data.action] ?? 0;
  drawDrone(simCx, drx * cell + cell / 2, dry * cell + cell / 2, cell * 0.3, heading);

  if (data.goal_reached) showGoalFlash();

  // Coord readout
  $('coord-readout').textContent = `${drx}, ${dry}`;
}

// ═══════════════════════════════════════════════════════════════════
// ADVANCED SIM RENDERER (continuous 20×20 world)
// ═══════════════════════════════════════════════════════════════════

function drawAdvancedSim(data) {
  const sc   = APP.simDims.scale;
  const { w, h } = APP.simDims;
  const gs   = data.grid_size || CFG.ADV_WORLD;

  simCx.clearRect(0, 0, w, h);
  simCx.fillStyle = '#060810'; simCx.fillRect(0, 0, w, h);

  // Grid every 2 units
  simCx.strokeStyle = hexAlpha(C.border, 0.55); simCx.lineWidth = 0.5;
  for (let i = 0; i <= gs; i += 2) {
    simCx.beginPath(); simCx.moveTo(i*sc,0); simCx.lineTo(i*sc,h); simCx.stroke();
    simCx.beginPath(); simCx.moveTo(0,i*sc); simCx.lineTo(w,i*sc); simCx.stroke();
  }

  // Goal
  const [gx, gy] = data.goal;
  drawGoal(simCx, gx * sc, gy * sc, 0.9 * sc, data.goal_reached);

  // Obstacles
  for (const o of (data.obstacles || [])) {
    const ox = o.x * sc, oy = o.y * sc, r = o.radius * sc;
    const moving = Math.abs(o.vx) > 0.001 || Math.abs(o.vy) > 0.001;

    // Glow halo
    const halo = simCx.createRadialGradient(ox, oy, 0, ox, oy, r * 2.2);
    halo.addColorStop(0, hexAlpha(C.hazard, moving ? 0.35 : 0.18));
    halo.addColorStop(1, hexAlpha(C.hazard, 0));
    simCx.beginPath(); simCx.arc(ox, oy, r * 2.2, 0, Math.PI * 2);
    simCx.fillStyle = halo; simCx.fill();

    // Body
    simCx.beginPath(); simCx.arc(ox, oy, r, 0, Math.PI * 2);
    simCx.fillStyle = moving ? hexAlpha(C.hazard, 0.7) : hexAlpha(C.hazard, 0.45);
    simCx.fill();
    simCx.strokeStyle = moving ? hexAlpha(C.hazard, 0.9) : hexAlpha(C.hazard, 0.55);
    simCx.lineWidth = 1; simCx.stroke();

    // Velocity arrow for moving obstacles
    if (moving) {
      const sp = Math.sqrt(o.vx ** 2 + o.vy ** 2);
      const al = Math.min(r * 2.5, 22);
      arrow(simCx, ox, oy, ox + (o.vx / sp) * al, oy + (o.vy / sp) * al, '#f09030', 1.5, 5);
    }
  }

  // 64-beam LIDAR
  const [dx, dy] = data.drone;
  const dcx = dx * sc, dcy = dy * sc;
  const maxDist = CFG.LIDAR_MAX_FAC * gs;

  simCx.save();
  for (const [rx, ry] of (data.lidar || [])) {
    const dist = Math.sqrt((rx - dx) ** 2 + (ry - dy) ** 2);
    const nd   = Math.min(1, dist / maxDist);
    const rv   = Math.round(255 * (1 - nd));
    const gv   = Math.round(180 * nd);
    const alpha = 0.25 + (1 - nd) * 0.45;
    simCx.strokeStyle = `rgba(${rv},${gv},30,${alpha})`;
    simCx.lineWidth   = nd < 0.35 ? 1.2 : 0.7;
    simCx.beginPath(); simCx.moveTo(dcx, dcy); simCx.lineTo(rx * sc, ry * sc); simCx.stroke();
    simCx.fillStyle = `rgba(${rv},${gv},30,${nd < 0.4 ? 0.9 : 0.4})`;
    simCx.beginPath(); simCx.arc(rx * sc, ry * sc, nd < 0.4 ? 2.5 : 1.4, 0, Math.PI * 2); simCx.fill();
  }
  simCx.restore();

  // Trail
  const toAX = v => v * sc;
  drawTrail(simCx, APP.trail, toAX, toAX);

  // Drone
  const [ax, ay] = data.action || [0, 0];
  const heading = Math.atan2(ay, ax);
  drawDrone(simCx, dcx, dcy, 9, heading);

  // Energy bar (bottom strip)
  const energy = data.energy ?? 1.0;
  simCx.fillStyle = '#040609';
  simCx.fillRect(0, h - 7, w, 7);
  const ec = energy > 0.6 ? C.signal : energy > 0.3 ? C.amber : C.hazard;
  simCx.fillStyle = ec;
  simCx.fillRect(0, h - 7, w * energy, 7);

  // Wind vane (top-right corner overlay)
  drawWindOnCanvas(simCx, w - 40, 40, data.wind || [0, 0]);

  if (data.goal_reached) showGoalFlash();

  const [ddx, ddy] = data.drone;
  $('coord-readout').textContent = `${ddx.toFixed(2)}, ${ddy.toFixed(2)}`;
}

function drawWindOnCanvas(cx, wx, wy, wind) {
  const [fx, fy] = wind;
  const mag = Math.sqrt(fx * fx + fy * fy);
  cx.save();
  cx.strokeStyle = hexAlpha(C.surface2, 0.8); cx.lineWidth = 0.8;
  cx.beginPath(); cx.arc(wx, wy, 20, 0, Math.PI * 2); cx.stroke();
  if (mag > 0.0005) {
    const len = Math.min(mag * 200, 16);
    arrow(cx, wx, wy, wx + fx / mag * len, wy + fy / mag * len, hexAlpha(C.ice, 0.65), 1.2, 4);
  }
  cx.fillStyle = hexAlpha(C.subtext, 0.5);
  cx.font = "500 6px 'Space Grotesk',sans-serif";
  cx.textAlign = 'center'; cx.textBaseline = 'top';
  cx.fillText('WIND', wx, wy + 22);
  cx.restore();
}

// ── Unified draw dispatcher ──────────────────────────────────────
function drawSim(data) {
  if (data.grid_size !== undefined) {
    pushTrail(data.drone[0], data.drone[1]);
    drawAdvancedSim(data);
    updateInstrumentsAdvanced(data);
  } else {
    pushTrail(data.drone[0], data.drone[1]);
    drawGridSim(data);
    updateInstrumentsGrid(data);
  }
}

// ═══════════════════════════════════════════════════════════════════
// INSTRUMENTS PANEL UPDATERS
// ═══════════════════════════════════════════════════════════════════

function updateInstrumentsAdvanced(data) {
  // Energy
  const energy = data.energy ?? 1.0;
  const ec = energy > 0.6 ? C.signal : energy > 0.3 ? C.amber : C.hazard;
  $('energy-fill').style.width      = (energy * 100).toFixed(1) + '%';
  $('energy-fill').style.background = ec;
  $('energy-val').textContent       = (energy * 100).toFixed(1) + '%';

  // Wind
  const [wx, wy] = data.wind || [0, 0];
  $('wind-val').textContent = `(${wx.toFixed(3)}, ${wy.toFixed(3)})`;
  drawWindMini(data.wind);

  // Motor
  const mok = data.motor_ok || [1, 1];
  ['motor-0', 'motor-1'].forEach((id, i) => {
    const ok = mok[i] > 0.5;
    $(id).textContent = `M${i+1} ${ok ? '●' : '✕'}`;
    $(id).classList.toggle('fail', !ok);
  });

  // Hide policy bars in advanced mode
  $('policy-block').style.opacity = '0.3';

  // Polar LIDAR
  drawLidar(data.lidar, data.drone, data.grid_size || CFG.ADV_WORLD);

  // Telemetry
  setTelem('t-episode', data.episode ?? '—');
  setTelem('t-step',    data.step ?? '—');
  setTelem('t-reward',  typeof data.ep_reward === 'number' ? data.ep_reward.toFixed(1) : '—');
  setTelem('t-step-rew', typeof data.reward === 'number' ? data.reward.toFixed(3) : '—');
  setTelem('t-algo', data.algo || 'PPO');
}

function updateInstrumentsGrid(data) {
  $('energy-fill').style.width = '100%';
  $('energy-fill').style.background = C.signal;
  $('energy-val').textContent = '—';
  $('wind-val').textContent   = '(—, —)';
  $('motor-0').classList.remove('fail'); $('motor-0').textContent = 'M1 ●';
  $('motor-1').classList.remove('fail'); $('motor-1').textContent = 'M2 ●';
  $('policy-block').style.opacity = '1';
  drawWindMini([0, 0]);

  // LIDAR
  if (data.rays) drawLidar(data.rays.map(([x,y]) => [x, y]), data.drone, 10, true);
  else           drawLidar(null, data.drone, 10, false);

  setTelem('t-episode', data.episode ?? '—');
  setTelem('t-step',    data.step ?? '—');
  setTelem('t-reward',  '—');
  setTelem('t-step-rew', typeof data.reward === 'number' ? data.reward.toFixed(3) : '—');
}

// ── Wind mini-compass ─────────────────────────────────────────────
function drawWindMini(wind) {
  const W = windCv.width  / (window.devicePixelRatio || 1);
  const H = windCv.height / (window.devicePixelRatio || 1);
  const cx2 = W / 2, cy2 = H / 2, R = Math.min(W, H) * 0.38;
  const [fx, fy] = wind || [0, 0];
  const mag = Math.sqrt(fx * fx + fy * fy);

  windCx.clearRect(0, 0, W, H);
  // Ring
  windCx.strokeStyle = hexAlpha(C.border, 0.9); windCx.lineWidth = 1;
  windCx.beginPath(); windCx.arc(cx2, cy2, R, 0, Math.PI * 2); windCx.stroke();
  // Cardinal marks
  windCx.fillStyle = hexAlpha(C.subtext, 0.5);
  windCx.font = `500 7px 'Space Grotesk',sans-serif`;
  windCx.textAlign = 'center'; windCx.textBaseline = 'middle';
  const cardinals = [['N', 0, -1], ['E', 1, 0], ['S', 0, 1], ['W', -1, 0]];
  for (const [lbl, ex, ey] of cardinals) {
    windCx.fillText(lbl, cx2 + ex * (R + 8), cy2 + ey * (R + 8));
  }
  // Wind arrow
  if (mag > 0.001) {
    const len = Math.min(mag * 180, R * 0.85);
    arrow(windCx, cx2, cy2, cx2 + fx / mag * len, cy2 + fy / mag * len, hexAlpha(C.ice, 0.8), 1.5, 5);
  } else {
    windCx.fillStyle = hexAlpha(C.subtext, 0.3);
    windCx.beginPath(); windCx.arc(cx2, cy2, 2, 0, Math.PI * 2); windCx.fill();
  }
}

// ═══════════════════════════════════════════════════════════════════
// POLAR LIDAR RING — signature element
// ═══════════════════════════════════════════════════════════════════

function drawLidar(beams, dronePos, worldSize, isGrid) {
  const W   = lidarCv.width  / (window.devicePixelRatio || 1);
  const H   = lidarCv.height / (window.devicePixelRatio || 1);
  const cx2 = W / 2, cy2 = H / 2;
  const R   = Math.min(W, H) * 0.42;
  const maxRange = CFG.LIDAR_MAX_FAC * (worldSize || CFG.ADV_WORLD);

  lidarCx.clearRect(0, 0, W, H);
  lidarCx.fillStyle = '#06080e'; lidarCx.fillRect(0, 0, W, H);

  // Range rings (3 levels)
  for (let i = 1; i <= 3; i++) {
    lidarCx.beginPath(); lidarCx.arc(cx2, cy2, R * i / 3, 0, Math.PI * 2);
    lidarCx.strokeStyle = i === 3
      ? hexAlpha(C.border, 0.75)
      : hexAlpha(C.border, 0.35);
    lidarCx.lineWidth = i === 3 ? 1 : 0.5;
    if (i < 3) { lidarCx.setLineDash([2, 6]); } else { lidarCx.setLineDash([]); }
    lidarCx.stroke();
    lidarCx.setLineDash([]);
  }

  // Compass labels
  lidarCx.fillStyle = hexAlpha(C.subtext, 0.55);
  lidarCx.font = `500 7px 'Space Grotesk',sans-serif`;
  lidarCx.textAlign = 'center'; lidarCx.textBaseline = 'middle';
  const dirs = [['N', 0,-1],['E',1,0],['S',0,1],['W',-1,0]];
  for (const [lbl, ex, ey] of dirs) {
    lidarCx.fillText(lbl, cx2 + ex * (R + 11), cy2 + ey * (R + 11));
  }

  // No beams state
  if (!beams || !beams.length || !dronePos) {
    lidarCx.fillStyle = hexAlpha(C.subtext, 0.35);
    lidarCx.font = `500 8px 'JetBrains Mono',monospace`;
    lidarCx.textAlign = 'center'; lidarCx.textBaseline = 'middle';
    lidarCx.fillText('NO SIGNAL', cx2, cy2);
    return;
  }

  const [ddx, ddy] = dronePos;

  // Draw beams
  for (const beam of beams) {
    let bx, by;
    if (Array.isArray(beam)) { bx = beam[0]; by = beam[1]; }
    else { bx = beam[0]; by = beam[1]; }

    const relX = bx - ddx, relY = by - ddy;
    const dist  = Math.sqrt(relX * relX + relY * relY);
    const nd    = Math.min(1, dist / maxRange);

    // Map to polar canvas: keep axis orientation (y down in world = y down on display)
    const canvX = cx2 + (relX / maxRange) * R;
    const canvY = cy2 + (relY / maxRange) * R;

    // Color by distance: amber (near) → ice (mid) → dim (far)
    let color, alpha, lw;
    if (nd < 0.33) {
      color = C.amber;   alpha = 0.85; lw = 1.8;
    } else if (nd < 0.66) {
      color = C.ice;     alpha = 0.55; lw = 1.2;
    } else {
      color = C.subtext; alpha = 0.28; lw = 0.8;
    }

    lidarCx.globalAlpha = alpha;
    lidarCx.strokeStyle = color; lidarCx.lineWidth = lw;
    lidarCx.beginPath(); lidarCx.moveTo(cx2, cy2); lidarCx.lineTo(canvX, canvY); lidarCx.stroke();

    // Endpoint dot
    lidarCx.globalAlpha = nd < 0.4 ? 0.95 : 0.45;
    lidarCx.fillStyle = color;
    lidarCx.beginPath(); lidarCx.arc(canvX, canvY, nd < 0.4 ? 2.2 : 1.2, 0, Math.PI * 2); lidarCx.fill();
  }
  lidarCx.globalAlpha = 1;

  // Center drone dot
  const centerGrad = lidarCx.createRadialGradient(cx2, cy2, 0, cx2, cy2, 5);
  centerGrad.addColorStop(0, C.amber); centerGrad.addColorStop(1, hexAlpha(C.amber, 0));
  lidarCx.beginPath(); lidarCx.arc(cx2, cy2, 5, 0, Math.PI * 2);
  lidarCx.fillStyle = centerGrad; lidarCx.fill();

  // Beam count label
  lidarCx.fillStyle = hexAlpha(C.subtext, 0.45);
  lidarCx.font = `400 6.5px 'JetBrains Mono',monospace`;
  lidarCx.textAlign = 'center'; lidarCx.textBaseline = 'bottom';
  lidarCx.fillText(`${beams.length} BEAMS`, cx2, H - 4);
}

// ═══════════════════════════════════════════════════════════════════
// SSE MANAGEMENT — auto-reconnect on drop
// ═══════════════════════════════════════════════════════════════════

function _simUrl() {
  if (APP.mode === 'advanced') return '/simulate/advanced/start';
  if (APP.mode === 'lidar')    return '/simulate/start?lidar=1';
  return '/simulate/start?lidar=0';
}

function startSim() {
  stopSim();
  APP.trail = [];
  $('sim-status').textContent = '● CONNECTING';
  $('sim-status').className   = 'sim-status';
  _connectSim();
}

function _connectSim() {
  if (APP.simEs) { try { APP.simEs.close(); } catch (_) {} }
  const es = new EventSource(_simUrl());
  APP.simEs = es;

  es.onmessage = e => {
    clearTimeout(APP.reconnTimer);
    try {
      const data = JSON.parse(e.data);
      APP.lastFrame = data;
      drawSim(data);
      updateSimTelem(data);
      $('sim-status').textContent = `● ${APP.mode.toUpperCase()}`;
      $('sim-status').className   = 'sim-status live';
      // sync wi-canvas if it has obs
      if (APP.mode === 'advanced') redrawWhatIf();
    } catch (_) {}
  };

  es.onerror = () => {
    $('sim-status').textContent = '⚠ RECONNECTING';
    $('sim-status').className   = 'sim-status';
    es.close();
    APP.simEs = null;
    // Only reconnect if user hasn't explicitly stopped
    APP.reconnTimer = setTimeout(() => {
      if (APP.simEs === null && !APP._userStopped) _connectSim();
    }, CFG.RECONNECT_MS);
  };
}

function stopSim() {
  APP._userStopped = true;
  clearTimeout(APP.reconnTimer);
  if (APP.simEs) { APP.simEs.close(); APP.simEs = null; }
  fetch(APP.mode === 'advanced' ? '/simulate/advanced/stop' : '/simulate/stop').catch(() => {});
  $('sim-status').textContent = 'IDLE';
  $('sim-status').className   = 'sim-status';
}

function updateSimTelem(data) {
  if (data.grid_size !== undefined) {
    setTelem('t-model', 'advanced_' + (data.algo || 'ppo').toLowerCase());
  } else {
    setTelem('t-model', data.rays ? 'lidar' : 'coord');
  }
}

// ── Policy confidence bars (grid mode) ──────────────────────────
function updateConfBars(probs, action) {
  const bars = [
    { id: 'cb-up', fill: 'cf-up', val: 'cv-up', key: 'up' },
    { id: 'cb-dn', fill: 'cf-dn', val: 'cv-dn', key: 'down' },
    { id: 'cb-lt', fill: 'cf-lt', val: 'cv-lt', key: 'left' },
    { id: 'cb-rt', fill: 'cf-rt', val: 'cv-rt', key: 'right' },
  ];
  const dirMap = { 0: 'up', 1: 'down', 2: 'left', 3: 'right' };
  const chosenDir = dirMap[action] ?? null;
  for (const b of bars) {
    const pct = ((probs?.[b.key] ?? 0.25) * 100).toFixed(1);
    $(b.fill).style.width = pct + '%';
    $(b.val).textContent  = pct + '%';
    const row = $(b.id);
    row.classList.toggle('best', b.key === chosenDir);
  }
}

// ═══════════════════════════════════════════════════════════════════
// REWARD CURVE — custom canvas renderer (no Chart.js)
// ═══════════════════════════════════════════════════════════════════

const ALGO_COLORS = { ppo: C.amber, sac: C.signal, coord: C.ice };

function drawCurve() {
  const W   = curveCv.width  / (window.devicePixelRatio || 1);
  const H   = curveCv.height / (window.devicePixelRatio || 1);
  const PAD = { t: 14, r: 48, b: 24, l: 48 };

  curveCx.clearRect(0, 0, W, H);
  curveCx.fillStyle = C.void; curveCx.fillRect(0, 0, W, H);

  const series = [];
  for (const [algo, d] of Object.entries(APP.metrics)) {
    const rewards = (d.rewards || []).slice(-CFG.CURVE_CAP);
    if (rewards.length < 2) continue;
    series.push({ algo, color: ALGO_COLORS[algo] || C.subtext, rewards });
  }

  if (!series.length) {
    curveCx.fillStyle = hexAlpha(C.subtext, 0.4);
    curveCx.font      = `500 0.65rem 'Space Grotesk',sans-serif`;
    curveCx.textAlign = 'center'; curveCx.textBaseline = 'middle';
    curveCx.fillText('NO TRAINING DATA — LAUNCH TRAINING TO SEE REWARD CURVE', W / 2, H / 2);
    return;
  }

  // Y range
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (const v of s.rewards) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  const yPad = Math.max(20, (yMax - yMin) * 0.1);
  yMin -= yPad; yMax += yPad;
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;

  const toX = (i, n) => PAD.l + (i / Math.max(n - 1, 1)) * plotW;
  const toY = v       => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  // Y grid lines + labels
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const y   = PAD.t + (i / yTicks) * plotH;
    const val = yMax - (i / yTicks) * (yMax - yMin);
    curveCx.strokeStyle = hexAlpha(C.border, 0.6); curveCx.lineWidth = 0.5;
    curveCx.beginPath(); curveCx.moveTo(PAD.l, y); curveCx.lineTo(W - PAD.r, y); curveCx.stroke();
    curveCx.fillStyle   = hexAlpha(C.subtext, 0.7);
    curveCx.font        = `400 8px 'JetBrains Mono',monospace`;
    curveCx.textAlign   = 'right'; curveCx.textBaseline = 'middle';
    curveCx.fillText(val.toFixed(0), PAD.l - 5, y);
  }

  // Series
  for (const s of series) {
    const data = s.rewards, n = data.length;

    // Area fill
    curveCx.beginPath();
    curveCx.moveTo(toX(0, n), toY(data[0]));
    for (let i = 1; i < n; i++) curveCx.lineTo(toX(i, n), toY(data[i]));
    curveCx.lineTo(toX(n - 1, n), H - PAD.b);
    curveCx.lineTo(toX(0, n),     H - PAD.b);
    curveCx.closePath();
    curveCx.fillStyle = hexAlpha(s.color, 0.06); curveCx.fill();

    // Raw line (faint)
    curveCx.beginPath();
    curveCx.moveTo(toX(0, n), toY(data[0]));
    for (let i = 1; i < n; i++) curveCx.lineTo(toX(i, n), toY(data[i]));
    curveCx.strokeStyle = hexAlpha(s.color, 0.3); curveCx.lineWidth = 0.8; curveCx.stroke();

    // Moving average (window adaptive)
    const win = Math.max(2, Math.min(15, Math.floor(n / 8)));
    curveCx.beginPath();
    let first = true;
    for (let i = 0; i < n; i++) {
      const lo  = Math.max(0, i - win);
      let sum   = 0;
      for (let j = lo; j <= i; j++) sum += data[j];
      const avg = sum / (i - lo + 1);
      if (first) { curveCx.moveTo(toX(i, n), toY(avg)); first = false; }
      else        curveCx.lineTo(toX(i, n), toY(avg));
    }
    curveCx.strokeStyle = s.color; curveCx.lineWidth = 2; curveCx.stroke();

    // Right-edge label
    const lastY = toY(data[n - 1]);
    curveCx.fillStyle   = s.color;
    curveCx.font        = `600 8px 'JetBrains Mono',monospace`;
    curveCx.textAlign   = 'left'; curveCx.textBaseline = 'middle';
    curveCx.fillText(s.algo.toUpperCase(), W - PAD.r + 4, lastY);
  }

  // Axes
  curveCx.strokeStyle = hexAlpha(C.border, 0.9); curveCx.lineWidth = 1;
  curveCx.beginPath();
  curveCx.moveTo(PAD.l, PAD.t); curveCx.lineTo(PAD.l, H - PAD.b);
  curveCx.lineTo(W - PAD.r, H - PAD.b); curveCx.stroke();
}

async function refreshCurve() {
  try {
    const res  = await fetch('/metrics');
    if (!res.ok) return;
    APP.metrics = await res.json();
    drawCurve();
    const total = Object.values(APP.metrics).reduce((s, d) => s + (d.rewards || []).length, 0);
    $('curve-status').textContent = total
      ? `${total} episode(s) plotted across ${Object.keys(APP.metrics).length} algo(s).`
      : 'No training data yet. Click ↑ PPO or ↑ SAC to begin.';
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════════
// STATE POLL (1 s)
// ═══════════════════════════════════════════════════════════════════

async function pollState() {
  try {
    const data = await (await fetch('/state')).json();
    setTelem('t-status', data.training ? 'TRAINING' : (APP.simEs ? 'LIVE' : 'IDLE'));
    if (data.training) {
      $('ind-dot').className = 'ind-dot training';
      $('sys-status-text').textContent = 'TRAINING ACTIVE';
    } else {
      $('ind-dot').className = 'ind-dot';
      $('sys-status-text').textContent = 'SYSTEM NOMINAL';
    }
    if (!APP.simEs) {
      setTelem('t-episode', data.episode ?? '—');
      if (data.reward !== undefined) setTelem('t-reward', data.reward.toFixed(1));
    }
    if (data.probs) updateConfBars(data.probs, data.last_action);
    if (data.model_type) setTelem('t-model', data.model_type);
    if (data.algo) setTelem('t-algo', data.algo.toUpperCase());
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════════
// CURRICULUM POLL (2.5 s)
// ═══════════════════════════════════════════════════════════════════

const STAGE_NAMES = [
  '', 'Clear Sky', 'Light Traffic', 'First Mover', 'Gusty', 'Moderate',
  'Obstacle Course', 'Storm', 'Motor Trouble', 'GPS Denied', 'Combat Zone'
];

function buildCurrPips(stage) {
  const wrap = $('curr-pips');
  wrap.innerHTML = '';
  for (let i = 1; i <= 10; i++) {
    const pip = document.createElement('div');
    pip.className = 'curr-pip' + (i < stage ? ' done' : i === stage ? ' active' : '');
    pip.title = STAGE_NAMES[i] || `Stage ${i}`;
    wrap.appendChild(pip);
  }
}

async function pollCurriculum() {
  try {
    const d = await (await fetch('/curriculum/state')).json();
    const stage = d.stage || 1;
    $('curr-badge').textContent = `STAGE ${stage}`;
    $('curr-name').textContent  = d.stage_name || STAGE_NAMES[stage] || '—';
    const sr = ((d.success_rate || 0) * 100).toFixed(1);
    $('curr-fill').style.width  = sr + '%';
    $('curr-pct').textContent   = sr + '%';
    buildCurrPips(stage);
    const hist = (d.history || []).slice(-32);
    $('curr-history').innerHTML = hist.map(v =>
      `<span class="ep-dot ${v ? 's' : 'f'}" title="${v ? 'Success' : 'Fail'}"></span>`
    ).join('');
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════════
// XAI FEATURE IMPORTANCE
// ═══════════════════════════════════════════════════════════════════

async function loadXAI() {
  $('xai-status').textContent = 'Computing — this may take 10–20 s…';
  $('xai-bars').innerHTML     = '';
  $('xai-entropy').style.display = 'none';
  try {
    const res = await fetch('/xai/importance?n=200');
    if (!res.ok) {
      const e = await res.json();
      $('xai-status').textContent = '⚠ ' + (e.error || 'Error');
      return;
    }
    const data = await res.json();
    const ent  = data.entropy || {};
    if (ent.mean !== undefined) {
      $('xai-entropy').textContent  = `Action entropy: ${ent.mean.toFixed(4)} nats (σ=${(ent.std||0).toFixed(4)})`;
      $('xai-entropy').style.display = 'block';
    }
    const imp  = data.importance || {};
    const rows = Object.entries(imp);
    if (!rows.length) { $('xai-status').textContent = 'No importance data returned.'; return; }
    $('xai-status').textContent = `Top ${rows.length} features by policy influence:`;
    const maxScore = Math.max(...rows.map(([,v]) => v), 0.001);
    $('xai-bars').innerHTML = rows.map(([name, score]) => {
      const pct = (score / maxScore * 100).toFixed(1);
      const raw = (score * 100).toFixed(2);
      return `<div class="xai-bar-row">
        <div class="xai-bar-label" title="${name}">${name}</div>
        <div class="xai-bar-track"><div class="xai-bar-fill" style="width:${pct}%"></div></div>
        <div class="xai-bar-val">${raw}%</div>
      </div>`;
    }).join('');
  } catch (e) {
    $('xai-status').textContent = '⚠ ' + e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════

async function loadBenchmark() {
  $('bench-status').textContent = 'Loading…';
  $('bench-body').innerHTML     = '';
  $('sig-result').classList.remove('visible');
  try {
    const res = await fetch('/benchmark');
    if (!res.ok) {
      const e = await res.json();
      $('bench-status').textContent = '⚠ ' + (e.error || 'Not found');
      return;
    }
    const data  = await res.json();
    const stats = data.stats || {};
    if (!Object.keys(stats).length) {
      $('bench-status').textContent = 'Benchmark file exists but contains no results.';
      return;
    }
    $('bench-status').textContent = `${Object.keys(stats).length} algo(s) compared.`;
    $('bench-body').innerHTML = Object.entries(stats).map(([algo, s]) => `
      <tr>
        <td>${algo.toUpperCase()}</td>
        <td>${(s.mean_reward || 0).toFixed(1)}</td>
        <td>±${(s.std_reward || 0).toFixed(1)}</td>
        <td>${((s.mean_success_rate || 0) * 100).toFixed(1)}%</td>
        <td>${s.n_seeds || '?'}</td>
      </tr>`).join('');
    const sig = data.significance_test;
    if (sig) {
      const p = typeof sig.p_value === 'number' ? sig.p_value.toFixed(4) : sig.p_value;
      $('sig-result').textContent =
        `Mann-Whitney U · p = ${p}${sig.significant_at_0_05 ? ' *' : ''} · Winner: ${sig.winner || '—'}`;
      $('sig-result').classList.add('visible');
    }
  } catch (e) {
    $('bench-status').textContent = '⚠ ' + e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// MISSION PLANNER (Groq AI)
// ═══════════════════════════════════════════════════════════════════

async function planMission() {
  const nl = $('mission-input').value.trim();
  if (!nl) return;
  const res_el = $('mission-result');
  res_el.innerHTML  = '<span class="empty-state-text">Planning…</span>';
  res_el.className  = 'mission-result';
  try {
    const res  = await fetch('/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mission: nl }),
    });
    const data = await res.json();
    if (data.error) { res_el.innerHTML = `<span style="color:var(--hazard)">⚠ ${data.error}</span>`; return; }
    const diff = data.difficulty || '—';
    const dc   = data.domain_config || {};
    res_el.innerHTML = `
      <strong style="color:var(--ice)">${data.mission_name || 'Custom Mission'}</strong>
      <span class="diff-badge diff-${diff}">${diff}</span><br>
      <span style="color:var(--subtext)">Static obs:</span> ${dc.n_static || 0} &nbsp;
      <span style="color:var(--subtext)">Moving:</span> ${dc.n_moving || 0} &nbsp;
      <span style="color:var(--subtext)">Wind:</span> ${(dc.wind_strength || 0).toFixed(3)} &nbsp;
      <span style="color:var(--subtext)">Motor fail:</span> ${((dc.motor_fail_prob || 0) * 100).toFixed(0)}%<br>
      <span style="color:var(--subtext)">Goal range:</span> ${dc.goal_dist_min || 0}–${dc.goal_dist_max || 0} m &nbsp;
      <span style="color:var(--subtext)">Algo:</span> <strong style="color:var(--amber)">${(data.algo || 'ppo').toUpperCase()}</strong><br>
      <em style="color:var(--subtext);font-size:0.68rem">${data.reasoning || ''}</em>
    `;
    res_el.classList.add('loaded');
  } catch (e) {
    res_el.innerHTML = `<span style="color:var(--hazard)">⚠ ${e.message}</span>`;
  }
}

async function loadMissionHistory() {
  const wrap = $('mission-history');
  try {
    const missions = await (await fetch('/missions')).json();
    if (!missions.length) { wrap.style.display = 'block'; wrap.innerHTML = '<div class="history-entry" style="color:var(--subtext)">No missions planned yet.</div>'; return; }
    wrap.style.display = 'block';
    wrap.innerHTML = missions.slice().reverse().map(m =>
      `<div class="history-entry" title="${m.raw_nl || ''}"
            onclick="document.getElementById('mission-input').value='${(m.raw_nl||'').replace(/'/g,"\\'")}'"
       >[${m.difficulty || '?'}] ${m.mission_name || 'Mission'} — ${(m.raw_nl || '').slice(0, 52)}…</div>`
    ).join('');
  } catch (_) {
    wrap.style.display = 'block';
    wrap.innerHTML = '<div class="history-entry" style="color:var(--subtext)">Backend not connected.</div>';
  }
}

// ═══════════════════════════════════════════════════════════════════
// TRAINING LAUNCHER
// ═══════════════════════════════════════════════════════════════════

function startTraining() { fetch('/train').catch(() => {}); }

async function launchTrain(algo) {
  const ts = algo === 'sac' ? 300000 : 500000;
  try {
    const res  = await fetch('/train/advanced', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algo, timesteps: ts, curriculum: true }),
    });
    const data = await res.json();
    $('curve-status').textContent =
      `Training ${data.algo.toUpperCase()} started (${data.timesteps.toLocaleString()} steps). Curve updates every 5 s.`;
    $('ind-dot').className = 'ind-dot training';
  } catch (e) {
    $('curve-status').textContent = '⚠ Could not launch training: ' + e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// WHAT-IF SIMULATOR
// ═══════════════════════════════════════════════════════════════════

function wiCanvasCoords(e) {
  const rect = wiCv.getBoundingClientRect();
  const scaleX = wiCv.width / (window.devicePixelRatio || 1) / rect.width;
  const scaleY = wiCv.height / (window.devicePixelRatio || 1) / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    cx: (clientX - rect.left) * scaleX,
    cy: (clientY - rect.top)  * scaleY,
  };
}

function canvasToWorld(cx, cy) {
  const { scale, w } = APP.simDims;
  if (APP.mode === 'advanced' && scale > 0) {
    return { wx: cx / scale, wy: cy / scale };
  } else {
    const cell = APP.simDims.cell || (w / CFG.GRID);
    return { wx: Math.floor(cx / cell), wy: Math.floor(cy / cell) };
  }
}

wiCv.addEventListener('click', e => {
  if (!APP.lastFrame) return;
  const { cx, cy } = wiCanvasCoords(e);
  const { wx, wy } = canvasToWorld(cx, cy);
  // Toggle: if click near existing obstacle, remove it
  const { scale, cell } = APP.simDims;
  const threshold = APP.mode === 'advanced' ? 1.2 : 0.8;
  const idx = APP.wiObs.findIndex(o => Math.hypot(o.wx - wx, o.wy - wy) < threshold);
  if (idx >= 0) APP.wiObs.splice(idx, 1);
  else          APP.wiObs.push({ wx, wy, r: APP.mode === 'advanced' ? 0.9 : 0.5 });
  APP.wiResult = null;
  redrawWhatIf();
});

wiCv.addEventListener('touchstart', e => {
  e.preventDefault();
  wiCv.dispatchEvent(new MouseEvent('click', { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY }));
}, { passive: false });

function redrawWhatIf() {
  const { w, h, scale, cell } = APP.simDims;
  wiCx.clearRect(0, 0, w, h);
  wiCx.fillStyle = '#060810'; wiCx.fillRect(0, 0, w, h);

  if (!APP.lastFrame) {
    wiCx.fillStyle = hexAlpha(C.subtext, 0.35);
    wiCx.font = `500 0.65rem 'Space Grotesk',sans-serif`;
    wiCx.textAlign = 'center'; wiCx.textBaseline = 'middle';
    wiCx.fillText('START ADVANCED SIM THEN TAP HERE', w / 2, h / 2);
    return;
  }

  const f = APP.lastFrame;
  const sc = APP.mode === 'advanced' ? scale : cell;

  // Subtle grid
  wiCx.strokeStyle = hexAlpha(C.border, 0.35); wiCx.lineWidth = 0.5;
  const gstep = APP.mode === 'advanced' ? sc * 2 : sc;
  for (let x = 0; x <= w; x += gstep) { wiCx.beginPath(); wiCx.moveTo(x,0); wiCx.lineTo(x,h); wiCx.stroke(); }
  for (let y = 0; y <= h; y += gstep) { wiCx.beginPath(); wiCx.moveTo(0,y); wiCx.lineTo(w,y); wiCx.stroke(); }

  // Original obstacles (dim)
  wiCx.globalAlpha = 0.4;
  if (APP.mode === 'advanced') {
    for (const o of (f.obstacles || [])) {
      wiCx.beginPath(); wiCx.arc(o.x * sc, o.y * sc, o.radius * sc, 0, Math.PI * 2);
      wiCx.fillStyle = hexAlpha(C.hazard, 0.3); wiCx.fill();
    }
  } else {
    for (const [ox, oy] of (f.obstacles || [])) {
      wiCx.fillStyle = hexAlpha(C.hazard, 0.25);
      wiCx.fillRect(ox * sc, oy * sc, sc, sc);
    }
  }
  wiCx.globalAlpha = 1;

  // Goal
  const [gx, gy] = f.goal;
  const gr = APP.mode === 'advanced' ? 0.9 * sc : sc * 0.4;
  drawGoal(wiCx, gx * sc, gy * sc, gr, false);

  // Drone (static)
  const [ddx, ddy] = f.drone;
  drawDrone(wiCx, ddx * sc, ddy * sc, APP.mode === 'advanced' ? 8 : sc * 0.28, 0);

  // User-placed obstacles (amber, interactive)
  for (const o of APP.wiObs) {
    const px = o.wx * sc, py = o.wy * sc, pr = (o.r || 0.9) * sc;
    const halo = wiCx.createRadialGradient(px, py, 0, px, py, pr * 2);
    halo.addColorStop(0, hexAlpha(C.amber, 0.35)); halo.addColorStop(1, hexAlpha(C.amber, 0));
    wiCx.beginPath(); wiCx.arc(px, py, pr * 2, 0, Math.PI * 2);
    wiCx.fillStyle = halo; wiCx.fill();
    wiCx.beginPath(); wiCx.arc(px, py, pr, 0, Math.PI * 2);
    wiCx.strokeStyle = C.amber; wiCx.lineWidth = 1.5; wiCx.stroke();
    wiCx.fillStyle = hexAlpha(C.amber, 0.2); wiCx.fill();
  }

  // Predicted trajectory overlay
  if (APP.wiResult) {
    const { baseline_action: ba, modified_action: ma } = APP.wiResult;
    const drx = ddx * sc, dry = ddy * sc, arrowLen = 60;
    if (ba) {
      const mag = Math.hypot(ba[0], ba[1]) || 1;
      arrow(wiCx, drx, dry,
        drx + ba[0] / mag * arrowLen, dry + ba[1] / mag * arrowLen,
        C.signal, 2, 8);
      wiCx.fillStyle = C.signal; wiCx.font = `600 7px 'Space Grotesk',sans-serif`;
      wiCx.textAlign = 'left'; wiCx.textBaseline = 'bottom';
      wiCx.fillText('BASELINE', drx + ba[0] / mag * arrowLen + 4, dry + ba[1] / mag * arrowLen);
    }
    if (ma) {
      const mag2 = Math.hypot(ma[0], ma[1]) || 1;
      wiCx.setLineDash([4, 4]);
      arrow(wiCx, drx, dry,
        drx + ma[0] / mag2 * arrowLen, dry + ma[1] / mag2 * arrowLen,
        C.amber, 2, 8);
      wiCx.setLineDash([]);
      wiCx.fillStyle = C.amber; wiCx.font = `600 7px 'Space Grotesk',sans-serif`;
      wiCx.textAlign = 'left'; wiCx.textBaseline = 'top';
      wiCx.fillText('MODIFIED', drx + ma[0] / mag2 * arrowLen + 4, dry + ma[1] / mag2 * arrowLen);
    }
  }

  // "WHAT-IF" watermark
  wiCx.fillStyle = hexAlpha(C.subtext, 0.12);
  wiCx.font = `700 0.55rem 'Space Grotesk',sans-serif`;
  wiCx.textAlign = 'right'; wiCx.textBaseline = 'top';
  wiCx.fillText('WHAT-IF MODE', w - 6, 5);
}

// Reconstruct approx obs vector from last advanced frame
function reconstructObs(frame) {
  const gs = frame.grid_size || CFG.ADV_WORLD;
  const maxRange = CFG.LIDAR_MAX_FAC * gs;
  const [ddx, ddy] = frame.drone;
  const [gx, gy]   = frame.goal;
  const obs = new Array(74).fill(0);

  // LIDAR [0:64]
  for (let i = 0; i < 64 && i < (frame.lidar || []).length; i++) {
    const [lx, ly] = frame.lidar[i];
    obs[i] = Math.min(1, Math.hypot(lx - ddx, ly - ddy) / maxRange);
  }
  // Modify for user-placed obstacles
  for (const o of APP.wiObs) {
    const dist = Math.hypot(o.wx - ddx, o.wy - ddy);
    const angle = Math.atan2(o.wy - ddy, o.wx - ddx);
    // Nearest beam index
    const beamIdx = Math.round(((angle + Math.PI) / (2 * Math.PI)) * 64) % 64;
    const nd = Math.min(1, (dist - o.r) / maxRange);
    for (let b = beamIdx - 1; b <= beamIdx + 1; b++) {
      const bi = ((b % 64) + 64) % 64;
      obs[bi] = Math.min(obs[bi], nd);
    }
  }
  // Goal
  const goalAngle = Math.atan2(gy - ddy, gx - ddx);
  obs[64] = Math.cos(goalAngle);
  obs[65] = Math.sin(goalAngle);
  obs[66] = Math.min(1, Math.hypot(gx - ddx, gy - ddy) / maxRange);
  obs[67] = 0; obs[68] = 0; // vel approx 0
  obs[69] = frame.energy ?? 1.0;
  const [wx, wy] = frame.wind || [0, 0];
  obs[70] = wx; obs[71] = wy;
  const mok = frame.motor_ok || [1, 1];
  obs[72] = mok[0]; obs[73] = mok[1];
  return obs;
}

async function runWhatIf() {
  if (!APP.lastFrame || APP.mode !== 'advanced') {
    $('wi-result').innerHTML = '<span style="color:var(--subtext)">Start the Advanced sim first.</span>';
    return;
  }
  if (!APP.wiObs.length) {
    $('wi-result').innerHTML = '<span style="color:var(--subtext)">Tap the canvas to place at least one obstacle.</span>';
    return;
  }
  $('wi-result').textContent = 'Predicting…';
  try {
    const obs   = reconstructObs(APP.lastFrame);
    const mods  = {};
    // Mark modified LIDAR channels
    for (const o of APP.wiObs) {
      const angle = Math.atan2(o.wy - APP.lastFrame.drone[1], o.wx - APP.lastFrame.drone[0]);
      const bi    = Math.round(((angle + Math.PI) / (2 * Math.PI)) * 64) % 64;
      mods[bi]    = obs[bi];
    }
    const res  = await fetch('/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ obs, modifications: mods }),
    });
    const data = await res.json();
    if (data.error) { $('wi-result').innerHTML = `<span style="color:var(--hazard)">⚠ ${data.error}</span>`; return; }
    APP.wiResult = data;
    redrawWhatIf();
    const [bx, by] = data.baseline_action || [0, 0];
    const [mx, my] = data.modified_action  || [0, 0];
    $('wi-result').innerHTML = `
      Baseline → <strong style="color:var(--signal)">(${bx.toFixed(3)}, ${by.toFixed(3)})</strong><br>
      Modified → <strong style="color:var(--amber)">(${mx.toFixed(3)}, ${my.toFixed(3)})</strong><br>
      <span style="color:var(--subtext)">Δ magnitude: ${data.delta_magnitude?.toFixed(4) || '—'} &nbsp;
      Entropy Δ: ${data.entropy_delta?.toFixed(4) || '—'}</span>
    `;
  } catch (e) {
    $('wi-result').innerHTML = `<span style="color:var(--hazard)">⚠ ${e.message}</span>`;
  }
}

function clearWhatIf() {
  APP.wiObs    = [];
  APP.wiResult = null;
  $('wi-result').textContent = '';
  redrawWhatIf();
}

// ═══════════════════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

function setMode(mode) {
  APP.mode = mode;
  APP.trail = [];
  APP.lastFrame = null;

  // Header pills
  document.querySelectorAll('.mode-btn, .mode-btn-lg').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
    btn.setAttribute('aria-pressed', btn.dataset.mode === mode);
  });

  // Resize canvas for new mode
  resizeSim();
  drawSimIdle();
  drawLidar(null, null, CFG.ADV_WORLD);
  drawWindMini([0, 0]);

  // Show/hide policy block
  $('policy-block').style.display = (mode === 'advanced') ? 'none' : '';
  $('motor-block').style.opacity  = (mode === 'advanced') ? '1' : '0.3';

  stopSim();
  APP._userStopped = false;
}

// ═══════════════════════════════════════════════════════════════════
// MOBILE NAV
// ═══════════════════════════════════════════════════════════════════

function openMobileNav() {
  $('mobile-nav').classList.add('open');
  $('mobile-nav').setAttribute('aria-hidden', 'false');
}
function closeMobileNav() {
  $('mobile-nav').classList.remove('open');
  $('mobile-nav').setAttribute('aria-hidden', 'true');
}
$('mobile-nav').addEventListener('click', e => {
  if (e.target === $('mobile-nav')) closeMobileNav();
});

// ═══════════════════════════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════════════════════════

function tickClock() {
  const now = new Date();
  $('sys-time').textContent =
    now.toTimeString().slice(0, 8) + ' UTC' + (now.getTimezoneOffset() <= 0 ? '+' : '') +
    String(-now.getTimezoneOffset() / 60).padStart(2, '0');
}
tickClock();
setInterval(tickClock, 1000);

// ═══════════════════════════════════════════════════════════════════
// POLLING INTERVALS
// ═══════════════════════════════════════════════════════════════════

setInterval(pollState,      CFG.POLL_STATE_MS);
setInterval(pollCurriculum, CFG.POLL_CURR_MS);
setInterval(refreshCurve,   CFG.POLL_CURVE_MS);

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

(function init() {
  // Build curriculum pips
  buildCurrPips(1);

  // Initial canvas states
  scaleCanvas(lidarCv, 180, 180);
  scaleCanvas(windCv, 56, 56);
  drawWindMini([0, 0]);
  drawLidar(null, null, CFG.ADV_WORLD);

  // Let the ResizeObserver fire to set initial sim/curve sizes
  // Also do an immediate pass in case it doesn't fire
  requestAnimationFrame(() => {
    resizeSim();
    resizeCurve();
  });

  // Initial data fetches
  pollState();
  pollCurriculum();
  refreshCurve();

  // Preload benchmark (non-blocking)
  loadBenchmark();
})();
