"""
Flask dashboard — single file, all routes.

Endpoints:
    GET  /                   Dashboard HTML
    GET  /train              Launch PPO training subprocess
    GET  /simulate           Launch Pygame simulator subprocess (local)
    GET  /simulate/start     SSE stream: browser canvas simulation
    GET  /simulate/stop      Stop SSE simulation thread
    GET  /state              Live training state JSON
    POST /predict            Model inference with action probs
    GET  /health             Health check
"""

import json
import os
import queue
import subprocess
import threading
import time

# Load Groq API key from file if not already in environment
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", ".groq_api_key")
if not os.environ.get("GROQ_API_KEY") and os.path.exists(_key_file):
    with open(_key_file) as _f:
        os.environ["GROQ_API_KEY"] = _f.read().strip()
from typing import Optional

from flask import Flask, Response, jsonify, render_template, render_template_string, request, stream_with_context

app = Flask(__name__)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATE_FILE = os.path.join(LOG_DIR, "live_state.json")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# STATE HELPERS
# ------------------------------------------------------------------
def init_state():
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "training": False, "simulation": False,
                "episode": 0, "reward": 0.0,
                "episode_length": 0, "last_action": None,
                "probs": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
            }, f)

def read_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def write_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

init_state()

# ------------------------------------------------------------------
# SSE SIMULATION STATE  (original grid env)
# ------------------------------------------------------------------
_sim_queue   = queue.Queue(maxsize=200)
_sim_stop    = threading.Event()
_sim_thread  = None

# ------------------------------------------------------------------
# SSE SIMULATION STATE  (advanced continuous env)
# ------------------------------------------------------------------
_adv_queue   = queue.Queue(maxsize=300)
_adv_stop    = threading.Event()
_adv_thread  = None


def _best_advanced_model_path() -> Optional[str]:
    """Return path to best available advanced model zip."""
    import glob
    patterns = [
        os.path.join(MODEL_DIR, "advanced", "ppo", "drone_ppo_advanced_final.zip"),
        os.path.join(MODEL_DIR, "advanced", "sac", "drone_sac_advanced_final.zip"),
        os.path.join(MODEL_DIR, "advanced", "**", "best_model.zip"),
    ]
    for p in patterns:
        for found in glob.glob(p, recursive=True):
            if os.path.exists(found):
                return found
    return None


def _run_advanced_sim(model_path: str) -> None:
    """
    Background thread: steps DroneEnvAdvanced and pushes SSE frames to _adv_queue.
    Runs continuous episodes; each frame contains full physics state for the canvas.
    """
    from stable_baselines3 import PPO, SAC
    from drone.envs.advanced import DroneEnvAdvanced, DomainConfig, ObstacleConfig

    config = DomainConfig(
        obstacles=ObstacleConfig(n_static=6, n_moving=3, speed_max=0.06),
        wind_strength=0.04,
        motor_fail_prob=0.0,
        goal_dist_min=8.0,
        goal_dist_max=14.0,
    )
    env = DroneEnvAdvanced(config=config)

    algo = "sac" if "sac" in os.path.basename(model_path).lower() else "ppo"
    AlgoClass = SAC if algo == "sac" else PPO
    try:
        model = AlgoClass.load(model_path)
    except Exception as exc:
        print(f"[AdvSim] Model load failed: {exc}")
        env.close()
        return

    obs, _ = env.reset()
    episode, step, ep_reward = 1, 0, 0.0

    while not _adv_stop.is_set():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        ep_reward += float(reward)

        lidar_eps = env.get_lidar_endpoints()
        frame = {
            "drone":       [float(env.drone_pos[0]), float(env.drone_pos[1])],
            "goal":        [float(env.goal_pos[0]),  float(env.goal_pos[1])],
            "obstacles":   env.get_obstacle_info(),
            "lidar":       [[float(x), float(y)] for x, y in lidar_eps],
            "action":      [float(a) for a in action.flatten()],
            "reward":      round(float(reward), 3),
            "ep_reward":   round(ep_reward, 1),
            "energy":      round(float(info.get("energy", 1.0)), 3),
            "wind":        [round(float(w), 4) for w in info.get("wind", [0.0, 0.0])],
            "motor_ok":    info.get("motor_ok", [1.0, 1.0]),
            "goal_reached":bool(info.get("goal_reached", False)),
            "step":        step,
            "episode":     episode,
            "grid_size":   float(env.config.grid_size),
            "algo":        algo.upper(),
        }
        try:
            _adv_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.07)   # ~14 fps

        if terminated or truncated:
            episode += 1
            ep_reward = 0.0
            obs, _ = env.reset()
            step = 0

    env.close()


def _run_browser_sim(model_path: str, use_lidar: bool):
    """
    Background thread: steps the env and pushes SSE frames to _sim_queue.

    Continuously runs episodes back-to-back with no pause.
    Stuck-loop guard: if the drone stays at the same position for >3 steps
    (happens when a deterministic policy keeps hitting the same obstacle),
    a random action is injected to break the loop.
    """
    import numpy as np
    from stable_baselines3 import PPO

    if use_lidar:
        from env.drone_env_lidar import DroneEnvLidar
        env = DroneEnvLidar()
    else:
        from env.drone_env import DroneEnv
        env = DroneEnv()

    model     = PPO.load(model_path)
    obs, _    = env.reset()
    step      = 0
    episode   = 1
    prev_pos  = None
    stuck     = 0

    while not _sim_stop.is_set():
        curr_pos = (int(env.drone_pos[0]), int(env.drone_pos[1]))

        # stuck-loop guard
        if curr_pos == prev_pos:
            stuck += 1
        else:
            stuck = 0
        prev_pos = curr_pos

        if stuck >= 3:
            action  = int(env.action_space.sample())
            stuck   = 0
        else:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

        obs, reward, done, _, _ = env.step(action)
        step += 1

        goal_reached = done and np.array_equal(env.drone_pos, env.goal_pos)

        frame = {
            "drone":        [int(env.drone_pos[0]), int(env.drone_pos[1])],
            "goal":         [int(env.goal_pos[0]),  int(env.goal_pos[1])],
            "obstacles":    [[int(x), int(y)] for x, y in env.obstacles],
            "action":       action,
            "reward":       round(float(reward), 3),
            "done":         bool(done),
            "goal_reached": bool(goal_reached),
            "step":         step,
            "episode":      episode,
            "rays":         (env.get_lidar_endpoints() if use_lidar else None),
        }
        try:
            _sim_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.12)  # ~8 fps — smooth but not too fast to follow

        if done:
            episode += 1
            obs, _ = env.reset()
            step     = 0
            prev_pos = None
            stuck    = 0

    env.close()


# ------------------------------------------------------------------
# DASHBOARD HTML
# ------------------------------------------------------------------
DASHBOARD = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Autonomous Drone RL</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0d0d1a; color: #e0e0e0;
    min-height: 100vh; padding: 16px;
  }
  h1 { text-align: center; font-size: 1.4rem; margin-bottom: 16px;
       color: #7eb8f7; letter-spacing: 1px; }
  .row  { display: flex; gap: 16px; flex-wrap: wrap; }
  .card {
    background: #141428; border: 1px solid #2a2a4a;
    border-radius: 10px; padding: 14px; flex: 1; min-width: 220px;
  }
  .card h2 { font-size: 0.85rem; color: #7eb8f7; text-transform: uppercase;
              letter-spacing: 1px; margin-bottom: 10px; }
  button {
    padding: 8px 18px; border: none; border-radius: 6px;
    cursor: pointer; font-size: 0.85rem; font-weight: 600; margin: 4px;
  }
  .btn-train { background: #28a745; color: #fff; }
  .btn-sim   { background: #007bff; color: #fff; }
  .btn-stop  { background: #dc3545; color: #fff; }
  .btn-lidar { background: #fd7e14; color: #fff; }
  .stat { font-size: 0.82rem; color: #aaa; margin: 3px 0; }
  .stat span { color: #fff; font-weight: 600; }

  /* canvas */
  #simCanvas {
    display: block; margin: 0 auto;
    border: 1px solid #2a2a4a; border-radius: 6px;
    background: #111122;
  }

  /* confidence bars */
  .bar-row { display: flex; align-items: center; margin: 5px 0; gap: 8px; }
  .bar-label { width: 40px; font-size: 0.78rem; color: #aaa; text-align: right; }
  .bar-track {
    flex: 1; height: 18px; background: #222240;
    border-radius: 4px; overflow: hidden;
    border: 1px solid #333355;
  }
  .bar-fill {
    height: 100%; width: 25%;
    border-radius: 4px; transition: width 0.3s ease, background 0.3s ease;
    background: #4466aa;
  }
  .bar-val { width: 38px; font-size: 0.75rem; color: #ccc; text-align: left; }
  .bar-row.active .bar-fill { background: #3ab0ff; box-shadow: 0 0 6px #3ab0ff88; }
  .bar-row.active .bar-label { color: #3ab0ff; font-weight: 700; }

  /* ── NEW: Curriculum card ── */
  .stage-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    background:#1a3a5c; color:#7eb8f7; font-weight:700; font-size:0.9rem;
  }
  .ep-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin:1px; vertical-align:middle;
  }
  .ep-dot.success { background:#27ae60; }
  .ep-dot.fail    { background:#c0392b; }

  /* ── NEW: Feature importance bars ── */
  .imp-row { display:flex; align-items:center; margin:3px 0; gap:6px; font-size:0.75rem; }
  .imp-label { width:90px; color:#aaa; text-align:right; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }
  .imp-track { flex:1; height:14px; background:#222240; border-radius:3px; overflow:hidden; }
  .imp-fill  { height:100%; border-radius:3px; background:#5588dd;
               transition:width 0.4s ease; }
  .imp-val   { width:35px; color:#ccc; text-align:left; }

  /* ── NEW: Benchmark table ── */
  .bench-table { width:100%; border-collapse:collapse; font-size:0.78rem; margin-top:8px; }
  .bench-table th { color:#7eb8f7; padding:4px 8px; border-bottom:1px solid #2a2a4a; text-align:left; }
  .bench-table td { padding:4px 8px; color:#ddd; }
  .bench-table tr:nth-child(even) td { background:#1a1a30; }

  /* ── NEW: Mission planner ── */
  .mission-input {
    width:100%; padding:8px; background:#111122; color:#e0e0e0;
    border:1px solid #2a2a4a; border-radius:6px; font-size:0.82rem;
    resize:vertical; margin-bottom:8px;
  }
  .mission-result {
    font-size:0.76rem; color:#aaa; margin-top:8px;
    border-left:3px solid #4499ff; padding-left:8px; line-height:1.5;
  }
  .difficulty-badge {
    display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.72rem;
    font-weight:700; margin-left:6px;
  }
  .diff-Easy     { background:#1a4a2a; color:#2ecc71; }
  .diff-Moderate { background:#4a3a0a; color:#f39c12; }
  .diff-Hard     { background:#4a1a0a; color:#e74c3c; }
  .diff-Extreme  { background:#3a0a3a; color:#e056ef; }

  /* ── Advanced canvas ── */
  #advCanvas {
    display:block; border-radius:8px;
    border:1px solid #2a2a4a; background:#070712;
    image-rendering:pixelated;
  }
  .adv-stat { font-size:0.76rem; color:#aaa; margin:2px 0; }
  .adv-stat span { color:#fff; font-weight:600; }
  .energy-bar-wrap { height:10px; background:#1a1a30; border-radius:5px;
                     overflow:hidden; margin:6px 0; }
  .energy-bar-fill { height:100%; border-radius:5px; transition:width 0.2s, background 0.3s; }
  .adv-badge { display:inline-block; padding:2px 8px; border-radius:10px;
               font-size:0.7rem; font-weight:700; background:#1a3a5c; color:#7eb8f7;
               margin-left:6px; }
  /* Training launch buttons */
  .btn-train-adv { background:#8e44ad; color:#fff; }
  .btn-train-sac { background:#16a085; color:#fff; }
</style>
</head>
<body>

<h1>&#x1F681; Autonomous Drone RL Control Center</h1>

<div class="row">

  <!-- CONTROLS -->
  <div class="card" style="max-width:260px">
    <h2>Controls</h2>
    <button class="btn-train" onclick="startTraining()">&#x25B6; Start Training</button>
    <button class="btn-sim"   onclick="startPygame()">&#x1F3AE; Pygame Sim</button>
    <hr style="border-color:#2a2a4a;margin:10px 0">
    <button class="btn-lidar" onclick="startBrowserSim(false)">&#x1F5A5; Coord Sim</button>
    <button class="btn-lidar" onclick="startBrowserSim(true)">&#x1F4E1; LIDAR Sim</button>
    <button class="btn-stop"  onclick="stopBrowserSim()">&#x23F9; Stop Sim</button>
  </div>

  <!-- LIVE STATS -->
  <div class="card" style="max-width:240px">
    <h2>Live Stats</h2>
    <div class="stat">Episode: <span id="ep">—</span></div>
    <div class="stat">Reward:  <span id="rew">—</span></div>
    <div class="stat">Length:  <span id="len">—</span></div>
    <div class="stat">Action:  <span id="act">—</span></div>
    <div class="stat">Training:<span id="trn">—</span></div>
    <div class="stat">Model:   <span id="mtype">—</span></div>
  </div>

  <!-- POLICY CONFIDENCE -->
  <div class="card" style="max-width:280px">
    <h2>Policy Confidence</h2>
    <div class="bar-row" id="bar-up">
      <div class="bar-label">Up</div>
      <div class="bar-track"><div class="bar-fill" id="fill-up"></div></div>
      <div class="bar-val"  id="val-up">25%</div>
    </div>
    <div class="bar-row" id="bar-down">
      <div class="bar-label">Down</div>
      <div class="bar-track"><div class="bar-fill" id="fill-down"></div></div>
      <div class="bar-val"  id="val-down">25%</div>
    </div>
    <div class="bar-row" id="bar-left">
      <div class="bar-label">Left</div>
      <div class="bar-track"><div class="bar-fill" id="fill-left"></div></div>
      <div class="bar-val"  id="val-left">25%</div>
    </div>
    <div class="bar-row" id="bar-right">
      <div class="bar-label">Right</div>
      <div class="bar-track"><div class="bar-fill" id="fill-right"></div></div>
      <div class="bar-val"  id="val-right">25%</div>
    </div>
  </div>

</div><!-- /row -->

<!-- CANVAS SIM -->
<div class="card" style="margin-top:16px">
  <h2>Browser Simulation <span id="simStatus" style="color:#aaa;font-size:0.75rem">— idle</span></h2>
  <div style="position:relative;display:inline-block;">
    <canvas id="simCanvas"></canvas>
    <div id="goalFlash" style="
      display:none; position:absolute; top:50%; left:50%;
      transform:translate(-50%,-50%);
      background:rgba(39,174,96,0.92); color:#fff;
      font-size:2rem; font-weight:700; padding:14px 32px;
      border-radius:12px; pointer-events:none;
    ">GOAL! &#x1F3C6;</div>
  </div>
  <div class="stat" style="text-align:center;margin-top:8px">
    Episode: <span id="simEp">—</span> &nbsp;|&nbsp;
    Step: <span id="simStep">—</span> &nbsp;|&nbsp;
    Reward: <span id="simReward">—</span> &nbsp;|&nbsp;
    Action: <span id="simAction">—</span>
  </div>
</div>

<script>
// ---------------------------------------------------------------
// CANVAS
// ---------------------------------------------------------------
const canvas  = document.getElementById("simCanvas");
const ctx     = canvas.getContext("2d");
const CELL    = 50;
const GRID    = 10;
canvas.width  = GRID * CELL;
canvas.height = GRID * CELL;

const ACTION_NAMES = ["Up","Down","Left","Right"];

let _flashTimer = null;

function showGoalFlash() {
  const el = document.getElementById("goalFlash");
  el.style.display = "block";
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(() => { el.style.display = "none"; }, 900);
}

function drawSim(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // background
  ctx.fillStyle = "#111122";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // grid lines
  ctx.strokeStyle = "#1e1e3a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= GRID; i++) {
    ctx.beginPath(); ctx.moveTo(i*CELL, 0); ctx.lineTo(i*CELL, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i*CELL); ctx.lineTo(canvas.width, i*CELL); ctx.stroke();
  }

  // obstacles
  ctx.fillStyle = "#c0392b";
  for (const [ox, oy] of data.obstacles) {
    ctx.fillRect(ox*CELL+3, oy*CELL+3, CELL-6, CELL-6);
  }

  // LIDAR rays
  if (data.rays) {
    const [dx, dy] = data.drone;
    const cx = dx*CELL + CELL/2, cy = dy*CELL + CELL/2;
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = "#ffe050";
    ctx.lineWidth   = 1.5;
    for (const [rx, ry] of data.rays) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(rx*CELL + CELL/2, ry*CELL + CELL/2);
      ctx.stroke();
      ctx.fillStyle = "#ffe050";
      ctx.beginPath();
      ctx.arc(rx*CELL + CELL/2, ry*CELL + CELL/2, 3, 0, Math.PI*2);
      ctx.fill();
    }
    ctx.restore();
  }

  // goal — pulse green when reached
  const [gx, gy] = data.goal;
  ctx.fillStyle = data.goal_reached ? "#2ecc71" : "#27ae60";
  ctx.beginPath();
  ctx.arc(gx*CELL+CELL/2, gy*CELL+CELL/2, CELL/2-4, 0, Math.PI*2);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 13px monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("G", gx*CELL+CELL/2, gy*CELL+CELL/2);

  // drone — bright blue circle with white centre dot
  const [drx, dry] = data.drone;
  ctx.fillStyle = "#2980b9";
  ctx.beginPath();
  ctx.arc(drx*CELL+CELL/2, dry*CELL+CELL/2, CELL/2-7, 0, Math.PI*2);
  ctx.fill();
  // white outline so it pops on dark cells
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 11px monospace";
  ctx.fillText("D", drx*CELL+CELL/2, dry*CELL+CELL/2);

  if (data.goal_reached) showGoalFlash();
}

// ---------------------------------------------------------------
// SSE SIMULATION
// ---------------------------------------------------------------
let _es = null;

function startBrowserSim(lidar) {
  stopBrowserSim();
  const url = "/simulate/start?lidar=" + (lidar ? "1" : "0");
  _es = new EventSource(url);
  document.getElementById("simStatus").textContent = "● running" + (lidar ? " (LIDAR)" : " (coord)");

  _es.onmessage = function(e) {
    const data = JSON.parse(e.data);
    drawSim(data);
    document.getElementById("simEp").textContent     = data.episode ?? "—";
    document.getElementById("simStep").textContent   = data.step;
    document.getElementById("simReward").textContent = data.reward.toFixed(2);
    document.getElementById("simAction").textContent = ACTION_NAMES[data.action] ?? data.action;
  };
  _es.onerror = function() {
    document.getElementById("simStatus").textContent = "— stopped";
  };
}

function stopBrowserSim() {
  if (_es) { _es.close(); _es = null; }
  fetch("/simulate/stop");
  document.getElementById("simStatus").textContent = "— idle";
}

// ---------------------------------------------------------------
// TRAINING / PYGAME
// ---------------------------------------------------------------
function startTraining() { fetch("/train"); }
function startPygame()   { fetch("/simulate"); }

// ---------------------------------------------------------------
// LIVE STATS POLL + CONFIDENCE BARS
// ---------------------------------------------------------------
const ACTIONS = ["up","down","left","right"];

function updateBars(probs, chosenAction) {
  const chosenName = ["up","down","left","right"][chosenAction] ?? null;
  for (const name of ACTIONS) {
    const pct = ((probs?.[name] ?? 0.25) * 100).toFixed(1);
    document.getElementById("fill-" + name).style.width = pct + "%";
    document.getElementById("val-"  + name).textContent  = pct + "%";
    const row = document.getElementById("bar-" + name);
    if (name === chosenName) row.classList.add("active");
    else                     row.classList.remove("active");
  }
}

async function pollState() {
  try {
    const res  = await fetch("/state");
    const data = await res.json();
    document.getElementById("ep").textContent    = data.episode   ?? "—";
    document.getElementById("rew").textContent   = (data.reward   ?? 0).toFixed(2);
    document.getElementById("len").textContent   = data.episode_length ?? "—";
    document.getElementById("act").textContent   = ACTION_NAMES[data.last_action] ?? "—";
    document.getElementById("trn").textContent   = data.training  ? " ✅ yes" : " no";
    document.getElementById("mtype").textContent = data.model_type ?? "coord";
    updateBars(data.probs, data.last_action);
  } catch (_) {}
}

setInterval(pollState, 1000);
pollState();

// ---------------------------------------------------------------
// CURRICULUM POLLING
// ---------------------------------------------------------------
async function pollCurriculum() {
  try {
    const res  = await fetch("/curriculum/state");
    const data = await res.json();
    document.getElementById("curr-stage").textContent =
      "Stage " + data.stage + ": " + (data.stage_name || "");
    const sr = ((data.success_rate || 0) * 100).toFixed(1);
    document.getElementById("curr-sr").textContent = sr + "%";
    document.getElementById("curr-sr-bar").style.width = sr + "%";
    document.getElementById("curr-eps").textContent = data.n_episodes || 0;
    const hist = data.history || [];
    const dotsEl = document.getElementById("curr-dots");
    dotsEl.innerHTML = hist.slice(-30).map(v =>
      `<span class="ep-dot ${v ? 'success' : 'fail'}"></span>`
    ).join("");
  } catch(_) {}
}
setInterval(pollCurriculum, 2000);
pollCurriculum();

// ---------------------------------------------------------------
// XAI — FEATURE IMPORTANCE
// ---------------------------------------------------------------
async function loadImportance() {
  document.getElementById("xai-status").textContent = "Computing...";
  document.getElementById("xai-bars").innerHTML = "";
  try {
    const res  = await fetch("/xai/importance?n=150");
    if (!res.ok) { const e = await res.json(); document.getElementById("xai-status").textContent = e.error || "Error"; return; }
    const data = await res.json();
    document.getElementById("xai-status").textContent =
      "Entropy: " + (data.entropy?.mean || 0).toFixed(3) + " nats";
    const imp = data.importance || {};
    const barsEl = document.getElementById("xai-bars");
    barsEl.innerHTML = "";
    for (const [name, score] of Object.entries(imp)) {
      const pct = (score * 100).toFixed(1);
      barsEl.innerHTML += `
        <div class="imp-row">
          <div class="imp-label" title="${name}">${name}</div>
          <div class="imp-track"><div class="imp-fill" style="width:${pct}%"></div></div>
          <div class="imp-val">${pct}%</div>
        </div>`;
    }
  } catch(e) {
    document.getElementById("xai-status").textContent = "Error: " + e.message;
  }
}

// ---------------------------------------------------------------
// BENCHMARK
// ---------------------------------------------------------------
async function loadBenchmark() {
  document.getElementById("bench-status").textContent = "Loading...";
  document.getElementById("bench-table-body").innerHTML = "";
  try {
    const res  = await fetch("/benchmark");
    if (!res.ok) { const e = await res.json(); document.getElementById("bench-status").textContent = e.error; return; }
    const data = await res.json();
    const stats = data.stats || {};
    let rows = "";
    for (const [algo, s] of Object.entries(stats)) {
      rows += `<tr>
        <td><b>${algo.toUpperCase()}</b></td>
        <td>${(s.mean_reward||0).toFixed(1)}</td>
        <td>± ${(s.std_reward||0).toFixed(1)}</td>
        <td>${((s.mean_success_rate||0)*100).toFixed(1)}%</td>
        <td>${s.n_seeds||"?"} seeds</td>
      </tr>`;
    }
    document.getElementById("bench-table-body").innerHTML = rows;
    const sig = data.significance_test;
    if (sig) {
      document.getElementById("bench-status").textContent =
        "Mann-Whitney p=" + (sig.p_value||"?").toFixed?.(sig.p_value||0, 4) +
        " | Winner: " + (sig.winner || "n/a");
    } else {
      document.getElementById("bench-status").textContent = "Loaded.";
    }
  } catch(e) {
    document.getElementById("bench-status").textContent = "Error: " + e.message;
  }
}

// ---------------------------------------------------------------
// MISSION PLANNER
// ---------------------------------------------------------------
// ---------------------------------------------------------------
// ADVANCED SIM CANVAS
// ---------------------------------------------------------------
const advCanvas = document.getElementById("advCanvas");
const actx = advCanvas.getContext("2d");
const ADV_SIZE = 500;
const GOAL_RADIUS_WORLD = 1.0;
let _advEs = null;
let _advFlashTimer = null;

function getScale(data) { return ADV_SIZE / (data.grid_size || 20.0); }

function showAdvGoalFlash() {
  const el = document.getElementById("goalFlash");
  el.style.display = "block";
  clearTimeout(_advFlashTimer);
  _advFlashTimer = setTimeout(() => { el.style.display = "none"; }, 1200);
}

function drawAdvancedSim(data) {
  const sc = getScale(data);
  actx.clearRect(0, 0, ADV_SIZE, ADV_SIZE);

  // Background
  actx.fillStyle = "#070712";
  actx.fillRect(0, 0, ADV_SIZE, ADV_SIZE);

  // Subtle grid every 2 world units
  actx.strokeStyle = "rgba(25,25,55,0.9)";
  actx.lineWidth = 0.5;
  const gs = data.grid_size || 20;
  for (let i = 0; i <= gs; i += 2) {
    actx.beginPath(); actx.moveTo(i*sc,0); actx.lineTo(i*sc,ADV_SIZE); actx.stroke();
    actx.beginPath(); actx.moveTo(0,i*sc); actx.lineTo(ADV_SIZE,i*sc); actx.stroke();
  }

  // Goal
  const [gx,gy] = data.goal;
  const gcx = gx*sc, gcy = gy*sc;
  const gg = actx.createRadialGradient(gcx,gcy,0,gcx,gcy,GOAL_RADIUS_WORLD*sc*2.5);
  gg.addColorStop(0, "rgba(46,204,113,0.55)");
  gg.addColorStop(1, "rgba(39,174,96,0)");
  actx.fillStyle = gg;
  actx.beginPath(); actx.arc(gcx,gcy,GOAL_RADIUS_WORLD*sc*2.5,0,Math.PI*2); actx.fill();
  actx.strokeStyle = data.goal_reached ? "#2ecc71" : "rgba(46,204,113,0.75)";
  actx.lineWidth = data.goal_reached ? 3 : 1.5;
  actx.beginPath(); actx.arc(gcx,gcy,GOAL_RADIUS_WORLD*sc,0,Math.PI*2); actx.stroke();
  actx.fillStyle = "#2ecc71";
  actx.font = "bold 10px monospace"; actx.textAlign="center"; actx.textBaseline="middle";
  actx.fillText("G", gcx, gcy);

  // Obstacles
  for (const o of (data.obstacles||[])) {
    const ox=o.x*sc, oy=o.y*sc, r=o.radius*sc;
    const moving = Math.abs(o.vx)>0.001||Math.abs(o.vy)>0.001;
    const grd = actx.createRadialGradient(ox,oy,0,ox,oy,r*1.8);
    grd.addColorStop(0, moving?"rgba(231,76,60,0.5)":"rgba(192,57,43,0.35)");
    grd.addColorStop(1, "rgba(0,0,0,0)");
    actx.fillStyle = grd;
    actx.beginPath(); actx.arc(ox,oy,r*1.8,0,Math.PI*2); actx.fill();
    actx.fillStyle = moving ? "#e74c3c" : "#922b21";
    actx.beginPath(); actx.arc(ox,oy,r,0,Math.PI*2); actx.fill();
    actx.strokeStyle = moving ? "#ff6b6b" : "#c0392b";
    actx.lineWidth = 1;
    actx.stroke();
    if (moving) {
      const sp = Math.sqrt(o.vx**2+o.vy**2);
      if (sp>0.001) {
        const al = Math.min(r*2,18);
        actx.strokeStyle="#f39c12"; actx.lineWidth=1.5;
        actx.beginPath(); actx.moveTo(ox,oy); actx.lineTo(ox+o.vx/sp*al,oy+o.vy/sp*al); actx.stroke();
      }
    }
  }

  // LIDAR rays
  const [dx,dy] = data.drone;
  const dcx=dx*sc, dcy=dy*sc;
  const maxDist = Math.sqrt(2)*gs;
  actx.save(); actx.globalAlpha=0.4; actx.lineWidth=0.7;
  for (const [rx,ry] of (data.lidar||[])) {
    const dist = Math.sqrt((rx-dx)**2+(ry-dy)**2)/maxDist;
    const rv=Math.round(255*(1-dist)), gv=Math.round(200*dist);
    actx.strokeStyle=`rgb(${rv},${gv},40)`;
    actx.beginPath(); actx.moveTo(dcx,dcy); actx.lineTo(rx*sc,ry*sc); actx.stroke();
    actx.globalAlpha=0.65;
    actx.fillStyle=`rgb(${rv},${gv},40)`;
    actx.beginPath(); actx.arc(rx*sc,ry*sc,1.8,0,Math.PI*2); actx.fill();
    actx.globalAlpha=0.4;
  }
  actx.restore();

  // Drone
  const drg = actx.createRadialGradient(dcx,dcy,0,dcx,dcy,12);
  drg.addColorStop(0,"#3498db"); drg.addColorStop(1,"#1a5276");
  actx.fillStyle=drg;
  actx.beginPath(); actx.arc(dcx,dcy,11,0,Math.PI*2); actx.fill();
  actx.strokeStyle="#7fb3d3"; actx.lineWidth=1.5; actx.stroke();
  // Heading arrow
  if (data.action&&data.action.length>=2) {
    const ax=data.action[0],ay=data.action[1],mag=Math.sqrt(ax**2+ay**2);
    if (mag>0.05) {
      actx.strokeStyle="#fff"; actx.lineWidth=2;
      actx.beginPath(); actx.moveTo(dcx,dcy);
      actx.lineTo(dcx+ax/mag*16,dcy+ay/mag*16); actx.stroke();
    }
  }
  actx.fillStyle="#fff"; actx.font="bold 8px monospace";
  actx.textAlign="center"; actx.textBaseline="middle";
  actx.fillText("D",dcx,dcy);

  if (data.goal_reached) showAdvGoalFlash();

  // Energy bar (bottom strip)
  const energy = data.energy??1.0;
  actx.fillStyle="#0a0a18";
  actx.fillRect(0,ADV_SIZE-9,ADV_SIZE,9);
  const eColor = energy>0.6?"#2ecc71":energy>0.3?"#f39c12":"#e74c3c";
  actx.fillStyle=eColor;
  actx.fillRect(0,ADV_SIZE-9,ADV_SIZE*energy,9);

  // Wind indicator circle (top-right corner)
  const [wx,wy]=data.wind||[0,0];
  const windMag=Math.sqrt(wx**2+wy**2);
  const wix=ADV_SIZE-36, wiy=36;
  actx.strokeStyle="rgba(80,120,220,0.4)"; actx.lineWidth=1;
  actx.beginPath(); actx.arc(wix,wiy,22,0,Math.PI*2); actx.stroke();
  if (windMag>0.0005) {
    const wl=Math.min(windMag*180,19);
    actx.strokeStyle="#5588ff"; actx.lineWidth=2;
    actx.beginPath(); actx.moveTo(wix,wiy);
    actx.lineTo(wix+wx/windMag*wl,wiy+wy/windMag*wl); actx.stroke();
  }
  actx.fillStyle="rgba(80,120,220,0.55)";
  actx.font="7px monospace"; actx.textAlign="center";
  actx.fillText("WIND",wix,wiy+30);
}

function startAdvSim() {
  stopAdvSim();
  _advEs = new EventSource("/simulate/advanced/start");
  document.getElementById("adv-sim-status").textContent = "● running";
  _advEs.onmessage = function(e) {
    const data = JSON.parse(e.data);
    drawAdvancedSim(data);
    document.getElementById("adv-ep").textContent    = data.episode;
    document.getElementById("adv-step").textContent  = data.step;
    document.getElementById("adv-rew").textContent   = data.reward.toFixed(3);
    document.getElementById("adv-eprew").textContent = data.ep_reward;
    document.getElementById("adv-algo-badge").textContent = data.algo||"PPO";
    const ep = data.energy??1.0;
    const eColor = ep>0.6?"#2ecc71":ep>0.3?"#f39c12":"#e74c3c";
    document.getElementById("adv-energy-bar").style.width  = (ep*100)+"%";
    document.getElementById("adv-energy-bar").style.background = eColor;
    document.getElementById("adv-energy-val").textContent  = (ep*100).toFixed(1)+"%";
    const w = data.wind||[0,0];
    document.getElementById("adv-wind").textContent  = `(${w[0].toFixed(3)}, ${w[1].toFixed(3)})`;
    const m = data.motor_ok||[1,1];
    document.getElementById("adv-motor").textContent = m.every(v=>v>0.5)?"OK":"FAIL";
    document.getElementById("adv-motor").style.color = m.every(v=>v>0.5)?"#2ecc71":"#e74c3c";
    document.getElementById("adv-goal-flag").textContent = data.goal_reached?"REACHED!":"—";
    document.getElementById("adv-goal-flag").style.color = data.goal_reached?"#2ecc71":"#aaa";
  };
  _advEs.onerror = function() {
    document.getElementById("adv-sim-status").textContent = "— error";
  };
}
function stopAdvSim() {
  if (_advEs) { _advEs.close(); _advEs=null; }
  fetch("/simulate/advanced/stop");
  document.getElementById("adv-sim-status").textContent = "idle";
}

// ---------------------------------------------------------------
// TRAINING LAUNCHER
// ---------------------------------------------------------------
async function launchTrain(algo) {
  const ts = algo==="sac" ? 300000 : 400000;
  const r = await fetch("/train/advanced", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({algo, timesteps:ts, curriculum:true}),
  });
  const d = await r.json();
  alert(`Training ${d.algo.toUpperCase()} started: ${d.timesteps} steps with curriculum.\nWatch the Reward Curve card update live.`);
}

// ---------------------------------------------------------------
// REWARD CURVE (Chart.js)
// ---------------------------------------------------------------
let _rewardChart = null;

function _algoColor(algo) {
  return {ppo:"#4499ff", sac:"#ff9944", coord:"#44ff99"}[algo] || "#aaaaaa";
}

async function loadRewardCurve() {
  const res = await fetch("/metrics");
  if (!res.ok) return;
  const data = await res.json();

  const datasets = [];
  for (const [algo, d] of Object.entries(data)) {
    const rewards = d.rewards||[];
    if (!rewards.length) continue;
    const col = _algoColor(algo);
    datasets.push({
      label: algo.toUpperCase(),
      data: rewards.slice(-150),
      borderColor: col,
      backgroundColor: col.replace(")", ",0.08)").replace("rgb","rgba"),
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: true,
    });
  }

  const ctx3 = document.getElementById("rewardChart").getContext("2d");
  if (_rewardChart) { _rewardChart.destroy(); _rewardChart=null; }
  if (!datasets.length) {
    document.getElementById("reward-curve-status").textContent = "No training data yet. Click Train PPO or Train SAC.";
    return;
  }
  _rewardChart = new Chart(ctx3, {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      responsive: true,
      plugins: {
        legend: { labels: { color:"#ccc", font:{size:11} } },
        tooltip: { mode:"index", intersect:false },
      },
      scales: {
        x: { display:false },
        y: { ticks:{color:"#aaa"}, grid:{color:"rgba(50,50,80,0.5)"} },
      },
    },
  });
  const n = datasets.reduce((s,d)=>s+d.data.length,0);
  document.getElementById("reward-curve-status").textContent =
    `Showing last 150 episodes per algo. Total plotted: ${n} eps.`;
}

// Auto-refresh reward curve every 5s
setInterval(loadRewardCurve, 5000);
loadRewardCurve();

async function planMission() {
  const nl = document.getElementById("mission-input").value.trim();
  if (!nl) return;
  document.getElementById("mission-result").innerHTML = "Planning...";
  try {
    const res = await fetch("/plan", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mission: nl}),
    });
    const data = await res.json();
    if (data.error) { document.getElementById("mission-result").textContent = "Error: " + data.error; return; }
    const diff = data.difficulty || "?";
    const dc   = data.domain_config || {};
    document.getElementById("mission-result").innerHTML = `
      <b>${data.mission_name || "Mission"}</b>
      <span class="difficulty-badge diff-${diff}">${diff}</span><br>
      Static obs: ${dc.n_static||0} &nbsp; Moving: ${dc.n_moving||0} &nbsp;
      Wind: ${(dc.wind_strength||0).toFixed(3)} &nbsp;
      Motor fail: ${((dc.motor_fail_prob||0)*100).toFixed(0)}%<br>
      Goal dist: ${dc.goal_dist_min||0}&ndash;${dc.goal_dist_max||0} m &nbsp;
      Algo: <b>${(data.algo||"ppo").toUpperCase()}</b><br>
      <i>${data.reasoning || ""}</i>
    `;
  } catch(e) {
    document.getElementById("mission-result").textContent = "Error: " + e.message;
  }
}
</script>

<!-- ============================================================ -->
<!-- ADVANCED ENV SIMULATION + REWARD CURVE                      -->
<!-- ============================================================ -->
<div class="row" style="margin-top:16px">

  <!-- ADVANCED CANVAS -->
  <div class="card" style="flex:0 0 auto">
    <h2>Advanced Env Simulation
      <span class="adv-badge" id="adv-algo-badge">—</span>
      <span id="adv-sim-status" style="color:#aaa;font-size:0.72rem;margin-left:8px">idle</span>
    </h2>
    <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap">
      <button class="btn-sim"       onclick="startAdvSim()">&#x25B6; Advanced Sim</button>
      <button class="btn-stop"      onclick="stopAdvSim()">&#x23F9; Stop</button>
      <button class="btn-train-adv" onclick="launchTrain('ppo')">&#x1F9E0; Train PPO</button>
      <button class="btn-train-sac" onclick="launchTrain('sac')">&#x1F9E0; Train SAC</button>
    </div>
    <canvas id="advCanvas" width="500" height="500"></canvas>
    <div style="margin-top:6px">
      <div class="adv-stat">Episode: <span id="adv-ep">—</span>  Step: <span id="adv-step">—</span>  Reward: <span id="adv-rew">—</span>  Ep Total: <span id="adv-eprew">—</span></div>
      <div class="adv-stat">Energy:
        <div class="energy-bar-wrap" style="display:inline-block;width:120px;vertical-align:middle">
          <div class="energy-bar-fill" id="adv-energy-bar" style="width:100%;background:#2ecc71"></div>
        </div>
        <span id="adv-energy-val">100%</span>
      </div>
      <div class="adv-stat">Wind: <span id="adv-wind">—</span>  Motor: <span id="adv-motor">OK</span>  Goal: <span id="adv-goal-flag">—</span></div>
    </div>
  </div>

  <!-- REWARD CURVE -->
  <div class="card" style="flex:1;min-width:280px">
    <h2>Live Reward Curve
      <button class="btn-lidar" style="float:right;margin:0;padding:4px 10px;font-size:0.75rem"
              onclick="loadRewardCurve()">Refresh</button>
    </h2>
    <canvas id="rewardChart" style="width:100%;max-height:300px"></canvas>
    <div class="stat" id="reward-curve-status" style="margin-top:6px;font-size:0.74rem">
      Auto-refreshes every 5s during training.
    </div>
  </div>

</div><!-- /advanced row -->

<!-- ============================================================ -->
<!-- NEW PANELS — row 2                                           -->
<!-- ============================================================ -->
<div class="row" style="margin-top:16px">

  <!-- CURRICULUM -->
  <div class="card" style="min-width:240px;max-width:340px">
    <h2>Curriculum Progress</h2>
    <div class="stat">Stage: <span class="stage-badge" id="curr-stage">—</span></div>
    <div class="stat" style="margin-top:8px">
      Success Rate: <span id="curr-sr">—</span>
      <div style="height:8px;background:#222240;border-radius:4px;margin-top:4px;overflow:hidden">
        <div id="curr-sr-bar" style="height:100%;background:#27ae60;width:0%;border-radius:4px;transition:width 0.5s"></div>
      </div>
    </div>
    <div class="stat" style="margin-top:6px">Episodes at stage: <span id="curr-eps">—</span></div>
    <div style="margin-top:8px;line-height:1.8" id="curr-dots"></div>
  </div>

  <!-- MISSION PLANNER -->
  <div class="card" style="flex:2;min-width:300px">
    <h2>Mission Planner <span style="color:#666;font-size:0.7rem">(Groq AI)</span></h2>
    <textarea class="mission-input" id="mission-input" rows="2"
      placeholder="Describe a mission e.g. 'Storm run with 10 obstacles and motor failure'"></textarea>
    <button class="btn-sim" onclick="planMission()">&#x1F4CB; Plan Mission</button>
    <div class="mission-result" id="mission-result" style="display:block;min-height:40px">
      Enter a mission description above and click Plan.
    </div>
  </div>

</div><!-- /row 2 -->

<div class="row" style="margin-top:16px">

  <!-- XAI PANEL -->
  <div class="card" style="flex:2;min-width:300px">
    <h2>Feature Importance (XAI)
      <button class="btn-lidar" style="float:right;margin:0;padding:4px 10px;font-size:0.75rem"
              onclick="loadImportance()">Analyse</button>
    </h2>
    <div class="stat" id="xai-status" style="margin-bottom:8px">Click Analyse to compute.</div>
    <div id="xai-bars"></div>
  </div>

  <!-- BENCHMARK -->
  <div class="card" style="flex:2;min-width:300px">
    <h2>Algorithm Benchmark (PPO vs SAC)
      <button class="btn-lidar" style="float:right;margin:0;padding:4px 10px;font-size:0.75rem"
              onclick="loadBenchmark()">Load</button>
    </h2>
    <div class="stat" id="bench-status" style="margin-bottom:6px">
      Click Load (run multi_algo first).
    </div>
    <table class="bench-table">
      <thead><tr><th>Algo</th><th>Mean Reward</th><th>Std</th><th>Success</th><th>Seeds</th></tr></thead>
      <tbody id="bench-table-body"></tbody>
    </table>
  </div>

</div><!-- /row 3 -->

</body>
</html>
"""

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/train")
def train():
    s = read_state(); s["training"] = True; write_state(s)
    def job():
        try:
            subprocess.Popen("python -m train.train_ppo", shell=True,
                             cwd=BASE_DIR)
        finally:
            s2 = read_state(); s2["training"] = False; write_state(s2)
    threading.Thread(target=job, daemon=True).start()
    return jsonify({"status": "training started"})


@app.route("/simulate")
def simulate_pygame():
    """Launch local Pygame window."""
    s = read_state(); s["simulation"] = True; write_state(s)
    def job():
        try:
            subprocess.Popen("python -m evaluate.simulate", shell=True,
                             cwd=BASE_DIR)
        finally:
            s2 = read_state(); s2["simulation"] = False; write_state(s2)
    threading.Thread(target=job, daemon=True).start()
    return jsonify({"status": "pygame simulation started"})


def _best_model_path(use_lidar: bool) -> Optional[str]:
    """Return the freshest available model zip, preferring final > best > latest checkpoint."""
    if use_lidar:
        candidates = [
            os.path.join(MODEL_DIR, "lidar", "drone_ppo_lidar_final.zip"),
            os.path.join(MODEL_DIR, "lidar", "best_model", "best_model.zip"),
        ]
        ckpt_dir = os.path.join(MODEL_DIR, "lidar", "checkpoints")
    else:
        candidates = [
            os.path.join(MODEL_DIR, "drone_ppo_final.zip"),
            os.path.join(MODEL_DIR, "best_model", "best_model.zip"),
            os.path.join(MODEL_DIR, "drone_ppo.zip"),
        ]
        ckpt_dir = os.path.join(MODEL_DIR, "checkpoints")

    for p in candidates:
        if os.path.exists(p):
            return p

    # fall back to latest checkpoint by modification time
    if os.path.isdir(ckpt_dir):
        zips = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".zip")],
            key=os.path.getmtime,
        )
        if zips:
            return zips[-1]
    return None


@app.route("/simulate/start")
def simulate_start():
    """SSE browser simulation stream."""
    global _sim_thread
    use_lidar = request.args.get("lidar", "0") == "1"

    model_path = _best_model_path(use_lidar)

    if not model_path:
        return jsonify({"error": "No model found. Train first."}), 404

    # stop any existing sim
    _sim_stop.set()
    if _sim_thread and _sim_thread.is_alive():
        _sim_thread.join(timeout=2)
    while not _sim_queue.empty():
        try: _sim_queue.get_nowait()
        except queue.Empty: break

    _sim_stop.clear()
    _sim_thread = threading.Thread(
        target=_run_browser_sim,
        args=(model_path, use_lidar),
        daemon=True,
    )
    _sim_thread.start()

    def generate():
        yield "retry: 500\n\n"
        while not _sim_stop.is_set():
            try:
                frame = _sim_queue.get(timeout=1.0)
                yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/simulate/stop")
def simulate_stop():
    _sim_stop.set()
    return jsonify({"status": "stopped"})


@app.route("/state")
def state():
    return jsonify(read_state())


@app.route("/predict", methods=["POST"])
def predict():
    import numpy as np
    data      = request.json or {}
    obs       = data.get("state", [])
    use_lidar = data.get("lidar", False)

    model_path = (
        os.path.join(MODEL_DIR, "lidar", "drone_ppo_lidar_final.zip")
        if use_lidar
        else os.path.join(MODEL_DIR, "drone_ppo.zip")
    )

    if not os.path.exists(model_path):
        # fallback: dummy action
        action = int(sum(obs)) % 4
        result = {
            "action": action,
            "action_name": {0:"up",1:"down",2:"left",3:"right"}[action],
            "probs": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
            "obs": obs,
        }
    else:
        from inference.predict import get_action_with_probs
        result = get_action_with_probs(obs, model_path=model_path)

    # persist probs to state so dashboard bars update
    s = read_state()
    s["last_action"] = result["action"]
    s["probs"]       = result["probs"]
    write_state(s)

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------
# NEW ROUTES — Phase 8
# ------------------------------------------------------------------

# Lazy model cache so the XAI routes don't re-load on every request
_model_cache: dict = {}
_xai_result_cache: dict = {}


def _load_advanced_model(model_path: Optional[str] = None):
    """Load the best available advanced model, caching by path."""
    import glob
    if model_path is None:
        patterns = [
            os.path.join(MODEL_DIR, "advanced", "**", "*final*.zip"),
            os.path.join(MODEL_DIR, "advanced", "**", "best_model.zip"),
        ]
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(pat, recursive=True))
        if not candidates:
            return None, None
        model_path = sorted(candidates, key=os.path.getmtime)[-1]

    if model_path in _model_cache:
        return _model_cache[model_path], model_path

    try:
        from stable_baselines3 import PPO, SAC
        algo = "sac" if "sac" in os.path.basename(model_path).lower() else "ppo"
        AlgoClass = SAC if algo == "sac" else PPO
        model = AlgoClass.load(model_path)
        _model_cache[model_path] = model
        return model, model_path
    except Exception as e:
        return None, None


@app.route("/curriculum/state")
def curriculum_state():
    """Return curriculum stage info from live_state.json."""
    s = read_state()
    return jsonify({
        "stage": s.get("curriculum_stage", 1),
        "stage_name": s.get("curriculum_stage_name", "Clear Sky"),
        "success_rate": s.get("curriculum_success_rate", 0.0),
        "n_episodes": s.get("curriculum_episodes_at_stage", 0),
        "history": s.get("curriculum_history", []),
        "promoted": s.get("curriculum_promoted", False),
        "demoted": s.get("curriculum_demoted", False),
    })


@app.route("/xai/importance")
def xai_importance():
    """Run feature importance on the best available advanced model."""
    n_samples = int(request.args.get("n", 150))
    model, model_path = _load_advanced_model()
    if model is None:
        return jsonify({"error": "No advanced model found. Run --train first."}), 404

    cache_key = f"importance_{model_path}_{n_samples}"
    if cache_key in _xai_result_cache:
        return jsonify(_xai_result_cache[cache_key])

    try:
        from drone.xai.explainer import XAIExplainer
        from drone.envs.advanced import DroneEnvAdvanced, DomainConfig, ObstacleConfig
        env = DroneEnvAdvanced(config=DomainConfig(
            obstacles=ObstacleConfig(n_static=0, n_moving=0),
            wind_strength=0.0, motor_fail_prob=0.0,
        ))
        xai = XAIExplainer(model, obs_space=env.observation_space)
        env.close()
        importance = xai.feature_importance(n_samples=n_samples, top_k=20)
        entropy_stats = xai.action_entropy_stats(n_samples=50)
        result = {
            "importance": importance,
            "entropy": entropy_stats,
            "model_path": model_path,
        }
        _xai_result_cache[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/xai/explain", methods=["POST"])
def xai_explain():
    """Generate NL trajectory explanation from a trajectory list."""
    data = request.json or {}
    trajectory = data.get("trajectory", [])
    if not trajectory:
        return jsonify({"explanation": "No trajectory data provided."}), 400

    model, _ = _load_advanced_model()
    if model is None:
        return jsonify({"explanation": "No model loaded.", "error": True}), 404

    try:
        from drone.xai.explainer import XAIExplainer
        xai = XAIExplainer(model)
        text = xai.explain_trajectory(trajectory)
        return jsonify({"explanation": text})
    except Exception as e:
        return jsonify({"explanation": f"Error: {e}"}), 500


@app.route("/whatif", methods=["POST"])
def whatif():
    """What-if analysis: modify obs features, see policy response."""
    data = request.json or {}
    obs = data.get("obs")
    modifications = {int(k): float(v) for k, v in data.get("modifications", {}).items()}

    if not obs:
        return jsonify({"error": "obs required"}), 400

    import numpy as np
    model, _ = _load_advanced_model()
    if model is None:
        return jsonify({"error": "No advanced model found."}), 404

    try:
        from drone.xai.explainer import XAIExplainer
        xai = XAIExplainer(model)
        result = xai.whatif(np.array(obs, dtype=np.float32), modifications)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/benchmark")
def benchmark():
    """Return cached PPO vs SAC benchmark results."""
    results_path = os.path.join(LOG_DIR, "multi_algo_results.json")
    if not os.path.exists(results_path):
        return jsonify({"error": "No benchmark results. Run: python -m train.multi_algo"}), 404
    try:
        with open(results_path) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/missions")
def missions():
    """Return logged NL missions."""
    log_path = os.path.join(LOG_DIR, "mission_log.jsonl")
    if not os.path.exists(log_path):
        return jsonify([])
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify(entries[-20:])  # last 20


@app.route("/simulate/advanced/start")
def simulate_advanced_start():
    """SSE stream for DroneEnvAdvanced canvas."""
    global _adv_thread
    model_path = _best_advanced_model_path()
    if not model_path:
        return jsonify({"error": "No advanced model found. Run: python train/run.py --train"}), 404

    _adv_stop.set()
    if _adv_thread and _adv_thread.is_alive():
        _adv_thread.join(timeout=2)
    while not _adv_queue.empty():
        try: _adv_queue.get_nowait()
        except queue.Empty: break

    _adv_stop.clear()
    _adv_thread = threading.Thread(target=_run_advanced_sim, args=(model_path,), daemon=True)
    _adv_thread.start()

    def generate():
        yield "retry: 300\n\n"
        while not _adv_stop.is_set():
            try:
                frame = _adv_queue.get(timeout=1.0)
                yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/simulate/advanced/stop")
def simulate_advanced_stop():
    _adv_stop.set()
    return jsonify({"status": "stopped"})


@app.route("/metrics")
def metrics():
    """Return episode reward history for the reward curve chart."""
    out = {}
    for algo in ("ppo", "sac"):
        fpath = os.path.join(LOG_DIR, f"metrics_advanced_{algo}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                out[algo] = {
                    "episodes": data.get("episodes", []),
                    "rewards":  data.get("rewards", []),
                    "successes":data.get("successes", []),
                }
            except Exception:
                pass
    # Also check basic metrics
    basic = os.path.join(LOG_DIR, "metrics.json")
    if os.path.exists(basic):
        try:
            with open(basic) as f:
                data = json.load(f)
            out["coord"] = {
                "episodes": data.get("episodes", []),
                "rewards":  data.get("rewards", []),
            }
        except Exception:
            pass
    return jsonify(out)


@app.route("/train/advanced", methods=["POST"])
def train_advanced_route():
    """Launch advanced training as a background subprocess."""
    data = request.json or {}
    algo = data.get("algo", "ppo")
    timesteps = int(data.get("timesteps", 300_000))
    curriculum = bool(data.get("curriculum", True))

    s = read_state()
    s["training"] = True
    s["algo"] = algo
    write_state(s)

    cmd = (
        f"python -m train.run --train --algo {algo} "
        f"--timesteps {timesteps} {'--curriculum' if curriculum else ''}"
    )

    def job():
        try:
            subprocess.run(cmd, shell=True, cwd=BASE_DIR)
        finally:
            s2 = read_state()
            s2["training"] = False
            write_state(s2)

    threading.Thread(target=job, daemon=True).start()
    return jsonify({"status": "training started", "algo": algo,
                    "timesteps": timesteps, "curriculum": curriculum})


@app.route("/plan", methods=["POST"])
def plan():
    """Parse NL mission via Groq and return MissionSpec JSON."""
    data = request.json or {}
    nl = data.get("mission", "").strip()
    if not nl:
        return jsonify({"error": "mission text required"}), 400

    try:
        from drone.llm.planner import MissionPlanner, MissionSpec
        planner = MissionPlanner(root_dir=None)
        spec = planner.plan(nl)
        result = spec.to_dict()
        result["difficulty"] = MissionSpec.difficulty_label(spec)
        result["domain_config"] = {
            "n_static": spec.n_static_obstacles,
            "n_moving": spec.n_moving_obstacles,
            "wind_strength": spec.wind_strength,
            "motor_fail_prob": spec.motor_fail_prob,
            "goal_dist_min": spec.goal_dist_min,
            "goal_dist_max": spec.goal_dist_max,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)
