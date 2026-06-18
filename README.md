# Autonomous Drone RL — Publication-Quality Navigation Agent

Check it out:- https://autonomous-drone-rl-control-system.onrender.com/

> **Abstract.** We present a deep reinforcement learning framework for autonomous drone navigation
> under realistic physical constraints including Dryden wind turbulence, partial motor failure,
> moving obstacles, and energy depletion. A 64-beam LIDAR sensor with sinusoidal angle encoding
> feeds a Transformer-based feature extractor (2-layer, 4-head, pre-LayerNorm) that learns
> spatially-aware obstacle representations. A 10-stage progressive curriculum with automatic
> promotion and demotion drives the agent from a clear-sky hover to a storm-level combat zone.
> We compare PPO and SAC with statistical significance (Mann-Whitney U, p < 0.05) across
> 3 seeds each, and provide XAI analysis (correlation-based feature importance, action entropy,
> natural-language trajectory explanation) alongside a Groq-powered natural-language mission
> planner. The full system is deployable as a Flask web dashboard with Server-Sent Event
> streaming, suitable for real-time human supervision of autonomous agents.

---

## Results

| Model | Env | Algorithm | Steps | Mean Reward | Success Rate | Seeds |
|-------|-----|-----------|-------|-------------|--------------|-------|
| Coord baseline | DroneEnv (10x10) | PPO | 200k | ~+84 | — | 1 |
| LIDAR baseline | DroneEnvLidar (8-ray) | PPO | 400k | ~+84 | — | 1 |
| Advanced (Stage 5) | DroneEnvAdvanced | PPO | 500k | TBD | TBD | 1 |
| Advanced (Stage 5) | DroneEnvAdvanced | SAC | 300k | TBD | TBD | 1 |
| Transformer + PPO | DroneEnvAdvanced | PPO+Transformer | 500k | TBD | TBD | 1 |

*Run `python -m train.multi_algo` to populate PPO vs SAC rows with significance test.*

---

## Architecture

```
                        Natural Language Mission
                               |
                      MissionPlanner (Groq API)
                      llama-3.3-70b-versatile
                               |
                          MissionSpec
                               |
                     ┌─────────────────┐
                     │  DomainConfig   │
                     │  (curriculum-   │
                     │   aware)        │
                     └────────┬────────┘
                              |
             ┌────────────────▼─────────────────┐
             │         DroneEnvAdvanced          │
             │  64-beam LIDAR  +  Dryden wind   │
             │  Motor failure  +  Moving obs    │
             │  Energy budget  +  Domain rand   │
             │  obs: Box(74,)  act: Box(2,)     │
             └────────────────┬─────────────────┘
                              |
             ┌────────────────▼─────────────────┐
             │     StagedCurriculum (SB3 CB)    │
             │  10 stages: Clear Sky -> Combat  │
             │  promote @ SR>80%, demote <30%   │
             └────────────────┬─────────────────┘
                              |
    ┌─────────────────────────▼──────────────────────────┐
    │          LidarTransformerExtractor                  │
    │  LIDAR[0:64] → [dist, cos(2pi*i/64), sin(...)]    │
    │  Linear(3, 64) → sequence of 64 beam tokens        │
    │  Context[64:74] → Linear(10,64) → CLS token        │
    │  [CLS|beam_0|...|beam_63] → TransformerEncoder    │
    │  2 layers, 4 heads, pre-LN, d_ff=128, GELU        │
    │  Mean-pool → Linear(64,128)+LayerNorm+ReLU         │
    │  Output: features (128,)                           │
    └─────────────────────────┬──────────────────────────┘
                              |
             ┌────────────────▼─────────────────┐
             │  PPO / SAC  (Stable Baselines3)  │
             │  MlpPolicy: [64] or [256,256]    │
             │  Optuna HPO (20 trials, TPE)     │
             └────────────────┬─────────────────┘
                              |
             ┌────────────────▼─────────────────┐
             │         XAIExplainer             │
             │  Feature importance (corr-based) │
             │  Action entropy (Gaussian H)     │
             │  NL trajectory explanation       │
             │  What-if analysis API            │
             └────────────────┬─────────────────┘
                              |
             ┌────────────────▼─────────────────┐
             │      Flask Dashboard (app.py)    │
             │  SSE canvas sim  /simulate/start │
             │  Curriculum card /curriculum/state│
             │  XAI panel       /xai/importance │
             │  Mission planner /plan           │
             │  Benchmark       /benchmark      │
             └──────────────────────────────────┘
```

---

## Quickstart

### 1. Install

```bash
git clone <repo-url> && cd Autonomous-Drone-RL
pip install -r requirements.txt
```

Set your Groq API key for the NL mission planner (optional):

```bash
export GROQ_API_KEY="gsk_..."   # Linux/Mac
set GROQ_API_KEY=gsk_...        # Windows CMD
```

### 2. Train

```bash
# Basic PPO on the original 10x10 grid
python -m train.train_ppo

# Advanced env: PPO with 10-stage curriculum
python train/run.py --train --algo ppo --timesteps 500000 --curriculum

# Advanced env: SAC (better for complex continuous tasks)
python train/run.py --train --algo sac --timesteps 300000

# Transformer policy (LIDAR-aware attention)
python train/run.py --train --algo ppo --transformer --timesteps 500000

# Hyperparameter optimisation (20 Optuna trials)
python train/run.py --hpo --algo ppo --trials 20

# PPO vs SAC statistical comparison
python train/run.py --multi-algo --timesteps 100000 --seeds 42 123 456
```

### 3. NL Mission Planning

```bash
# Plan via Groq -> MissionSpec -> auto-train
python train/run.py --plan "Storm run through 12 moving obstacles with motor failure"

# Via dashboard: open http://localhost:5000 -> Mission Planner card
```

### 4. Evaluate & Explain

```bash
# XAI analysis on best saved model
python train/run.py --xai

# Evaluate deterministically
python train/run.py --eval --episodes 100
```

### 5. Run Dashboard

```bash
python app.py
# Open http://localhost:5000
```

Dashboard panels:
- **Canvas Sim** — real-time SSE simulation with LIDAR rays
- **Live Stats** — episode / reward / model type
- **Policy Confidence** — action distribution bars
- **Curriculum Progress** — stage, success rate, episode history
- **Mission Planner** — NL -> MissionSpec via Groq
- **Feature Importance** — XAI bar chart (click Analyse)
- **Algorithm Benchmark** — PPO vs SAC comparison table

---

## File Structure

```
Autonomous-Drone-RL/
|
|-- app.py                        Flask dashboard (15 routes, SSE, XAI, planner)
|-- requirements.txt              Python dependencies
|
|-- env/                          ORIGINAL envs (backward compatible, untouched)
|   |-- drone_env.py              10x10 grid, Discrete(4), obs=Box(4,)
|   `-- drone_env_lidar.py        Same, 8-beam LIDAR, obs=Box(10,)
|
|-- drone/                        NEW advanced package
|   |-- envs/
|   |   `-- advanced.py           DroneEnvAdvanced: 64-beam LIDAR, wind, obs=Box(74,)
|   |-- curriculum/
|   |   `-- staged.py             StagedCurriculum: 10 stages, auto promote/demote
|   |-- policies/
|   |   `-- transformer.py        LidarTransformerExtractor (SB3 BaseFeaturesExtractor)
|   |-- xai/
|   |   `-- explainer.py          XAIExplainer: importance, entropy, NL, whatif
|   `-- llm/
|       `-- planner.py            MissionPlanner: NL -> MissionSpec via Groq API
|
|-- train/
|   |-- run.py                    MASTER CLI: all modes in one command
|   |-- train_ppo.py              Original PPO trainer (coord env)
|   |-- train_lidar.py            Original PPO trainer (LIDAR env)
|   |-- train_advanced.py         PPO/SAC trainer (DroneEnvAdvanced)
|   |-- hpo.py                    Optuna HPO: 20 trials, TPE sampler
|   `-- multi_algo.py             PPO vs SAC + Mann-Whitney U significance test
|
|-- evaluate/
|   |-- simulate.py               Pygame visual playback
|   |-- benchmark.py              100-ep eval across checkpoints
|   |-- policy_heatmap.py         10x10 arrow heatmap
|   `-- plot_variance.py          Shaded reward curve across seeds
|
|-- inference/
|   `-- predict.py                get_action() + get_action_with_probs()
|
|-- models/
|   |-- drone_ppo_final.zip       Coord baseline (200k steps)
|   |-- lidar/                    LIDAR model + checkpoints
|   |-- advanced/ppo/             Advanced PPO model + best + checkpoints
|   |-- advanced/sac/             Advanced SAC model + best + checkpoints
|   `-- benchmark/                Per-seed models for statistical comparison
|
`-- logs/
    |-- live_state.json           Live bridge: trainer <-> dashboard
    |-- metrics_advanced_ppo.json Episode history (rolling 200)
    |-- hpo_results_ppo.json      Optuna trial results
    |-- multi_algo_results.json   PPO vs SAC comparison
    `-- mission_log.jsonl         All NL mission logs
```

---

## Environment Details

### DroneEnvAdvanced — Observation Space (74-dim)

| Index | Feature | Description |
|-------|---------|-------------|
| 0–63 | `lidar_00`..`lidar_63` | 64-beam LIDAR: normalised distance [0,1] |
| 64–65 | `goal_cos`, `goal_sin` | Goal direction as unit vector |
| 66 | `goal_dist` | Goal distance, normalised |
| 67–68 | `vel_x`, `vel_y` | Current velocity |
| 69 | `energy` | Remaining energy [0,1] |
| 70–71 | `wind_x`, `wind_y` | Current wind vector |
| 72–73 | `motor_x`, `motor_y` | Motor status (1=ok, 0=failed) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Goal reached | +500 |
| Progress toward goal | +10 * delta |
| Obstacle collision | -50 |
| Near-miss (1.5x radius) | up to -20 |
| Energy expenditure | -0.05 per step |
| Excessive action jerk | -2.0 |
| Survival penalty | -0.1 per step |
| Boundary hit | -5.0 |

### Curriculum Stages

| Stage | Name | Static | Moving | Wind | Motor Fail |
|-------|------|--------|--------|------|-----------|
| 1 | Clear Sky | 2 | 0 | 0.000 | 0% |
| 2 | Light Traffic | 4 | 0 | 0.010 | 0% |
| 3 | First Mover | 6 | 1 | 0.020 | 0% |
| 4 | Gusty | 8 | 2 | 0.040 | 0% |
| 5 | Moderate | 8 | 3 | 0.060 | 0% |
| 6 | Windy Obstacle Course | 10 | 4 | 0.080 | 5% |
| 7 | Storm | 12 | 5 | 0.100 | 10% |
| 8 | Motor Trouble | 14 | 6 | 0.100 | 20% |
| 9 | GPS Denied | 16 | 8 | 0.120 | 30% |
| 10 | Combat Zone | 18 | 10 | 0.140 | 40% |

Auto-promote: success rate > 80% over 100 episodes.
Auto-demote: success rate < 30%.

---

## Transformer Policy

```
Input obs (74,) per step
  |
  +-- LIDAR [0:64] --------> Linear(3, 64) -> 64 beam tokens (B, 64, 64)
  |   [dist_i, cos(2pi*i/64), sin(2pi*i/64)]
  |
  `-- Context [64:74] -----> Linear(10, 64) -> CLS token (B, 1, 64)
                                               |
                             cat([CLS, beams]) (B, 65, 64)
                                               |
                         TransformerEncoder x2 (pre-LN, 4 heads, d_ff=128)
                                               |
                              Mean-pool (B, 64)
                                               |
                           Linear(64,128) + LayerNorm + ReLU
                                               |
                                    features (128,)
Total parameters: ~93,317 (extractor) + policy heads
```

---

## BibTeX

```bibtex
@misc{dronerl2026,
  title   = {Autonomous Drone Navigation with Curriculum RL,
             Transformer LIDAR Policy, and XAI Analysis},
  author  = {Autonomous Drone RL Team},
  year    = {2026},
  note    = {Continuous-action DroneEnvAdvanced: 64-beam LIDAR,
             Dryden wind, motor failure, 10-stage curriculum,
             PPO/SAC comparison with Mann-Whitney U significance test,
             Groq-powered NL mission planning.
             \url{https://github.com/Akshad1234/Autonomous-Drone-RL}},
}
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| stable-baselines3 | >=2.0 | PPO, SAC, callbacks, VecEnv |
| gymnasium | >=0.29 | Environment API |
| torch | >=2.0 | Transformer policy, GPU training |
| flask | >=3.0 | Web dashboard |
| optuna | 3.6.1 | Hyperparameter optimisation |
| groq | 0.11.0 | NL mission planner (Llama 3) |
| scipy | 1.14.1 | Mann-Whitney U significance test |
| matplotlib | any | Plots, heatmaps |
| numpy | any | Array operations |
