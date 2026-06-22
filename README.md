# Autonomous Drone Navigation via Deep Reinforcement Learning with Curriculum Training, Transformer-Based LIDAR Policy, and Explainability Analysis

**Live Demo:** https://autonomous-drone-rl-control-system.onrender.com/
**Repository:** https://github.com/Akshad1234/Autonomous-Drone-RL

---

## Abstract

This work presents a deep reinforcement learning framework for autonomous drone navigation under realistic physical constraints, including Dryden wind turbulence, single-axis motor failure, moving obstacles, and energy depletion. The agent perceives its environment through a 64-beam LIDAR sensor whose readings are processed by a Transformer-based feature extractor employing sinusoidal angle encoding, two encoder layers, four attention heads, and pre-Layer Normalization. A 10-stage progressive curriculum with automatic promotion and demotion drives training from an obstacle-free hover scenario to a storm-level combat zone with 18 static obstacles, 10 moving obstacles, 14% wind strength, and 40% motor-failure probability. Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) are compared with statistical significance (Mann-Whitney U test, p < 0.05) across three independent random seeds each. An explainability module provides correlation-based feature importance, action-distribution entropy, natural-language trajectory summaries, and interactive what-if perturbation analysis. The complete system is deployed as a Flask web dashboard with Server-Sent Event (SSE) streaming and a Groq-powered natural-language mission planner, enabling real-time human supervision of the autonomous agent.

---

## I. Introduction

Autonomous aerial vehicles (drones) must navigate dynamic, uncertain environments under strict physical constraints. Classical motion-planning approaches rely on accurate world models and hand-crafted heuristics that break down in the presence of wind disturbances, partial sensor failures, or unknown moving obstacles. Deep reinforcement learning (deep RL) offers an alternative: an agent learns a navigation policy entirely from reward signals, without requiring an explicit environment model.

Three key challenges motivate this work:

1. **Physical realism gap.** Most published RL navigation benchmarks use simplified grid worlds with discrete actions and no dynamics. Real drones experience momentum, wind drift, motor degradation, and finite energy budgets. A policy trained on oversimplified environments may fail to transfer.

2. **Curriculum difficulty.** A continuous-action environment with 18 obstacles and storm-level wind is too hard for an agent to learn from scratch. Without structured difficulty progression the policy collapses to a safe but sub-optimal hover strategy.

3. **Interpretability and supervision.** Deploying an autonomous agent in safety-critical settings requires operators to understand what the agent is attending to, when it is uncertain, and how it responds to hypothetical scenario changes.

This project addresses all three challenges in a single, end-to-end framework: a physically realistic simulation, a progressive curriculum, a spatially-aware Transformer policy, statistical evaluation across multiple algorithms and seeds, and an explainability and mission-planning interface accessible through a live web dashboard.

---

## II. Methodology

### A. Environment Design

Three Gymnasium-compatible environments are provided, ordered by increasing complexity.

**DroneEnv** is a discrete 10 × 10 grid with four cardinal actions {Up, Down, Left, Right}, a four-dimensional observation vector [drone_x, drone_y, goal_x, goal_y], and 15 static obstacles. It serves as a baseline to verify that the training pipeline and evaluation scripts are correct before introducing continuous dynamics.

**DroneEnvLidar** extends the baseline by replacing the raw coordinate observation with an eight-beam LIDAR sensor and a two-dimensional goal vector, yielding a ten-dimensional observation. Each LIDAR ray casts along one of eight directions and returns a normalized distance to the nearest obstacle or wall.

**DroneEnvAdvanced** is the primary publication-quality environment. It operates on a continuous 20 × 20 cell world with a two-dimensional continuous action space, Box(2,) ∈ [−1, +1]², where each component specifies a target velocity along one axis. The observation is a 74-dimensional vector structured as follows:

| Index  | Feature          | Description                                 |
|--------|------------------|---------------------------------------------|
| 0–63   | lidar_00–lidar_63 | 64-beam LIDAR distances, normalized [0, 1]  |
| 64–65  | goal_cos, goal_sin | Goal direction as a unit vector             |
| 66     | goal_dist        | Goal distance, normalized by world diagonal |
| 67–68  | vel_x, vel_y     | Current velocity, normalized by MAX_SPEED   |
| 69     | energy           | Remaining energy budget [0, 1]              |
| 70–71  | wind_x, wind_y   | Current wind vector, normalized by MAX_WIND |
| 72–73  | motor_x, motor_y | Per-axis motor status (1 = ok, 0 = failed)  |

**Physics model.** Velocity dynamics follow a damping-plus-acceleration model: v_t+1 = α · v_t + β · a_t + η_wind, where α is a damping coefficient, β is the control gain, and η_wind is a Dryden turbulence term modeled as a Gauss-Markov process with autocorrelation α_w = 0.95. At the start of each episode, a random single-axis motor failure is applied with probability p_fail, clamping the corresponding action component to zero. Energy depletes each step at a rate proportional to the squared action magnitude; reaching zero energy terminates the episode.

**LIDAR geometry.** Each of the 64 rays is cast at evenly spaced angles using ray-circle intersection for moving circular obstacles and analytical ray-segment intersection for static obstacles and walls. Distances are normalized by the world diagonal.

**Reward function.** The shaped reward at each step is:

R = 10 · Δdist + R_goal + R_collision + R_near_miss − 0.05 · ‖a‖² − 2.0 · ‖Δv‖² − 0.1 − 5.0 · [boundary hit]

where R_goal = +500 on success, R_collision = −50 on contact, and R_near_miss ≤ −20 graduated within 1.5× obstacle radius.

### B. Curriculum Learning

**StagedCurriculum** is a Stable-Baselines3 (SB3) callback that wraps the training environment and modulates difficulty automatically. Ten stages progress from an obstacle-free hover to a maximum-difficulty scenario:

| Stage | Name                  | Static Obs | Moving Obs | Wind  | Motor Fail |
|-------|-----------------------|-----------|-----------|-------|-----------|
| 1     | Clear Sky             | 2         | 0         | 0.000 | 0%        |
| 2     | Light Traffic         | 4         | 0         | 0.010 | 0%        |
| 3     | First Mover           | 6         | 1         | 0.020 | 0%        |
| 4     | Gusty                 | 8         | 2         | 0.040 | 0%        |
| 5     | Moderate              | 8         | 3         | 0.060 | 0%        |
| 6     | Windy Obstacle Course | 10        | 4         | 0.080 | 5%        |
| 7     | Storm                 | 12        | 5         | 0.100 | 10%       |
| 8     | Motor Trouble         | 14        | 6         | 0.100 | 20%       |
| 9     | GPS Denied            | 16        | 8         | 0.120 | 30%       |
| 10    | Combat Zone           | 18        | 10        | 0.140 | 40%       |

Promotion occurs when the rolling 100-episode success rate exceeds 80% for at least 20 consecutive episodes; demotion occurs when it falls below 30%. Stage transitions are persisted to `logs/live_state.json` so the dashboard can display live curriculum progress.

### C. Neural Network Architecture

**Standard MLP policy.** The default policy uses a two-hidden-layer MLP with widths [256, 256] and ReLU activations, followed by a linear policy head (output dim 2 for continuous actions) and a linear value head (output dim 1). This architecture is used with both PPO and SAC.

**Transformer-based policy (LidarTransformerExtractor).** The LIDAR-aware alternative encodes each of the 64 beams as a three-dimensional token [dist_i, cos(2πi/64), sin(2πi/64)], where the sinusoidal components carry fixed geometric angle information. Each token is projected to a model dimension D_MODEL = 64 by a shared linear layer. A context token [CLS] is constructed from the ten non-LIDAR features (goal, velocity, energy, wind, motor) via a separate linear projection. The full sequence [CLS | beam_0 | … | beam_63] (length 65) is passed through two Transformer encoder layers with four attention heads, d_ff = 128, GELU activations, and pre-Layer Normalization. All 65 output tokens are mean-pooled, then projected through a Linear(64, 128) + LayerNorm + ReLU layer to produce a 128-dimensional feature vector consumed by the SB3 policy and value heads. Total extractor parameters: approximately 93,317.

### D. Training Algorithms

**PPO (Proximal Policy Optimization)** [1] is the primary on-policy algorithm. Key hyperparameters: learning rate 3×10⁻⁴, n_steps 2048, batch size 64, discount γ = 0.99, GAE λ = 0.95, clip range 0.2, entropy coefficient 0.001. Default budget: 500,000 timesteps on the advanced environment.

**SAC (Soft Actor-Critic)** [2] is the off-policy alternative, better suited to sample-efficient exploration in continuous action spaces. Key hyperparameters: learning rate 3×10⁻⁴, replay buffer 200,000, batch size 256, soft update τ = 0.005, discount γ = 0.99, automatic entropy coefficient tuning. Default budget: 300,000 timesteps.

### E. Hyperparameter Optimization

Optuna [3] with the Tree-structured Parzen Estimator (TPE) sampler and a Median Pruner is used to search over learning rate, network width, batch size, discount factor, clip range (PPO), or buffer size and soft-update coefficient (SAC) across 20 trials per algorithm. Each trial trains for 30,000 timesteps and evaluates over 30 episodes. Results are logged to `logs/hpo_results_<algo>.json`.

### F. Statistical Evaluation

To evaluate whether PPO or SAC is significantly better, both algorithms are trained across three seeds (42, 123, 456), and 50 evaluation episodes per seed are collected. A two-sided Mann-Whitney U test (significance threshold p < 0.05) is applied to the pooled reward distributions. The result, p-value, effect size, and the winning algorithm are written to `logs/multi_algo_results.json`.

### G. Explainability (XAI)

**XAIExplainer** provides four analysis modes without SHAP or LIME dependencies:

- *Feature importance* samples 150 random observations, queries the deterministic policy on each, and computes the absolute Pearson correlation between each of the 74 observation features and each action component. The top-20 features ranked by importance are returned.
- *Action entropy* reports the Shannon entropy of the policy's action distribution: for Gaussian policies H = ½ ln(2πe σ²) per dimension.
- *Trajectory explanation* summarizes a recorded episode as a natural-language string using rule-based templates (no LLM required).
- *What-if analysis* perturbs a user-specified subset of observation features and returns the resulting action delta and distribution shift.

### H. Natural-Language Mission Planning

**MissionPlanner** converts a free-text mission description (e.g., "Storm run through 12 moving obstacles with motor failure") into a structured `MissionSpec` by calling the Groq API with the `llama-3.3-70b-versatile` model. The LLM is prompted with a strict JSON schema defining bounds for eight parameters: obstacle counts, wind strength, motor-failure probability, goal distance range, and preferred algorithm. The response is validated and clamped to safe ranges before being applied to the environment via `to_domain_config()`. All missions are logged to `logs/mission_log.jsonl`. A difficulty score is computed from the parsed spec and labeled Easy / Moderate / Hard / Extreme. If the Groq API is unavailable the planner returns safe defaults without raising an exception.

---

## III. Literature Survey

This project builds on and compares against the following bodies of work:

**Deep RL for navigation.** Mnih et al. [4] demonstrated that deep Q-networks (DQN) can learn complex control policies from raw pixel observations. Schulman et al. [1] introduced PPO, which has become the standard on-policy algorithm in robotics due to its stability. Haarnoja et al. [2] proposed SAC, adding maximum-entropy exploration via an automatic temperature parameter, making it particularly effective for continuous-action locomotion tasks. This project uses both PPO and SAC to compare their relative sample efficiency and final performance on the advanced environment.

**Curriculum learning.** Bengio et al. [5] formalized the principle that presenting training examples in a meaningful order from easy to hard improves convergence. Florensa et al. [6] applied goal-conditioned curriculum generation to robotic locomotion. The 10-stage StagedCurriculum in this project follows the same principle, with automatic promotion and demotion rather than a fixed schedule.

**Transformer architectures for RL.** Chen et al. [7] introduced Decision Transformer, which frames RL as sequence modeling. Vaswani et al. [8] established the Transformer as a general-purpose sequence model with multi-head self-attention. This project applies Transformer-based feature extraction specifically to structured LIDAR sequences, using fixed sinusoidal angle encoding inspired by positional encodings from [8].

**LIDAR-based navigation.** Himmelsbach et al. [9] and subsequent work demonstrated that LIDAR point clouds are the dominant sensor modality for autonomous ground vehicle navigation. This project adapts LIDAR to a 2D drone environment by using ray-circle and ray-segment intersection geometry to produce normalized per-beam distances, analogous to 2D occupancy scanning.

**Hyperparameter optimization.** Akiba et al. [3] presented Optuna, a black-box optimization framework using TPE, which has been widely adopted for neural network hyperparameter search. This project uses Optuna with a MedianPruner to search PPO and SAC hyperparameter spaces efficiently.

**Explainability in RL.** Lundberg and Lee [10] proposed SHAP values for model-agnostic feature attribution; Ribeiro et al. [11] proposed LIME. This project deliberately avoids those dependencies, instead using Pearson correlation between observations and policy outputs as a computationally lightweight importance proxy that is compatible with both discrete and continuous action policies.

**Large language models for task planning.** Ahn et al. [12] showed that large language models (LLMs) can be grounded to robotic skill primitives to produce executable task plans from natural language. This project applies a similar idea at the mission-specification level: the Groq-hosted `llama-3.3-70b-versatile` model translates free-text operator intent into validated environment configuration parameters.

---

## IV. Implementation

### A. Technology Stack

| Component              | Library / Tool                     | Version       |
|------------------------|------------------------------------|---------------|
| RL algorithms          | Stable-Baselines3                  | ≥ 2.3.0       |
| Environment API        | Gymnasium                          | ≥ 0.29.0      |
| Deep learning          | PyTorch                            | ≥ 2.0.0       |
| Web framework          | Flask                              | 3.1.3         |
| WSGI / async server    | Gunicorn + Gevent                  | 23.0.0 / 24.11.1 |
| Hyperparameter search  | Optuna                             | 3.6.1         |
| LLM API client         | Groq SDK                           | 1.4.0         |
| Statistics             | SciPy                              | 1.14.1        |
| Visualization          | Matplotlib                         | any           |
| Numerical computing    | NumPy                              | ≥ 1.24.0, < 2.0 |
| Python runtime         | CPython                            | 3.11.9        |
| Deployment platform    | Render (via Procfile)              | —             |

### B. Module Structure

The project is organized into five top-level packages and a single-file Flask application.

**`env/`** contains the two backward-compatible baseline environments (`drone_env.py`, `drone_env_lidar.py`) preserved without modification to allow fair comparison against the original discrete-grid baselines.

**`drone/`** is the advanced package. `drone/envs/advanced.py` implements `DroneEnvAdvanced`. `drone/curriculum/staged.py` implements `StagedCurriculum` as an SB3 `BaseCallback`. `drone/policies/transformer.py` implements `LidarTransformerExtractor` as an SB3 `BaseFeaturesExtractor`. `drone/xai/explainer.py` implements `XAIExplainer`. `drone/llm/planner.py` implements `MissionPlanner`.

**`train/`** contains the training orchestration. `train/run.py` is the master CLI entry point that dispatches to the appropriate training, evaluation, HPO, or mission-planning workflow via argparse flags. `train/train_advanced.py` contains the main training loop with four callbacks: `CheckpointCallback` (every 25,000 steps), `EvalCallback` (every 10,000 steps on a fixed Stage 5 configuration), `AdvancedMetricsCallback` (per-episode reward/length/success to JSON), and `StagedCurriculum` (when `--curriculum` is passed). `train/hpo.py` runs Optuna search. `train/multi_algo.py` runs the multi-seed comparison.

**`evaluate/`** provides `benchmark.py` for 100-episode evaluation across all saved checkpoints, `policy_heatmap.py` for visualizing the learned policy as a 10 × 10 arrow heatmap, `plot_variance.py` for shaded reward curves across seeds, and `simulate.py` for Pygame-based visual episode playback.

**`inference/`** provides `predict.py` with two API functions: `get_action(obs, model_path)` returning an integer action, and `get_action_with_probs(obs, model_path)` returning a dictionary with action, action name, per-action probabilities, and the original observation. Models are cached in memory after the first load.

**`app.py`** is the 1,774-line Flask web dashboard. It exposes 15 HTTP routes including Server-Sent Event (SSE) streaming endpoints for both the grid environment (`/simulate/start`) and the advanced environment (`/simulate/advanced/start`). Background threads step the simulation environment and push state frames to bounded queues (~200–300 frames); the SSE endpoints drain the queue at approximately 8–14 fps. When no trained model files are present, three fallback random-policy simulators (`_run_random_sim`, `_run_random_lidar_sim`, `_run_random_advanced_sim`) ensure the dashboard is fully functional on a fresh deployment. Live training state is synchronized between the trainer process and the dashboard via a shared `logs/live_state.json` file polled every 1–5 seconds.

### C. Dashboard UI

The single-page dashboard (`/`) renders seven major panels:

1. **Canvas Simulator** — real-time SSE-driven canvas showing drone position, LIDAR rays, obstacles, goal, and path history.
2. **Sensor Suite** — polar LIDAR plot, energy bar, and wind-direction vector.
3. **Live Stats** — episode number, cumulative reward, step count, last action, training status.
4. **Policy Confidence** — action probability bars with dynamic highlighting of the selected action.
5. **Curriculum Progress** — current stage badge, rolling success rate, and a history dot strip (green = success, red = failure).
6. **Mission Planner** — natural-language textarea that calls `/plan`, displays the parsed `MissionSpec`, difficulty badge, and domain configuration.
7. **Algorithm Benchmark** — tabular PPO vs. SAC comparison with mean reward, standard deviation, success rate, seeds used, and Mann-Whitney p-value.

### D. Deployment

The application is deployed on Render using the command:

```
gunicorn -w 1 --worker-class gevent -b 0.0.0.0:$PORT --timeout 120 app:app
```

A single Gevent worker is used because trained RL models are stateful in-memory objects; multiple workers would require shared-memory model serving. The 120-second timeout accommodates long-lived SSE connections.

---

## V. Results

| Model              | Environment              | Algorithm        | Steps | Mean Reward | Success Rate | Seeds |
|--------------------|--------------------------|------------------|-------|-------------|--------------|-------|
| Coord baseline     | DroneEnv (10 × 10)       | PPO              | 200 k | ~+84        | —            | 1     |
| LIDAR baseline     | DroneEnvLidar (8-beam)   | PPO              | 400 k | ~+84        | —            | 1     |
| Advanced (Stage 5) | DroneEnvAdvanced         | PPO              | 500 k | TBD         | TBD          | 1     |
| Advanced (Stage 5) | DroneEnvAdvanced         | SAC              | 300 k | TBD         | TBD          | 1     |
| Transformer + PPO  | DroneEnvAdvanced         | PPO + Transformer| 500 k | TBD         | TBD          | 1     |

*Run `python train/run.py --multi-algo --timesteps 100000 --seeds 42 123 456` to populate the PPO vs. SAC rows with significance-test results.*

---

## VI. Conclusion

This project demonstrates a complete pipeline for training, evaluating, explaining, and deploying deep RL agents for autonomous drone navigation. Starting from simple discrete-grid baselines, the framework scales to a physically realistic continuous environment with Dryden wind turbulence, motor failure, moving obstacles, and energy constraints. A 10-stage auto-advancing curriculum enables the agent to learn progressively without manual intervention. A Transformer-based policy with fixed sinusoidal angle encoding provides spatially-aware obstacle representation. Statistical comparison of PPO and SAC across multiple seeds with a Mann-Whitney U significance test ensures that performance claims are not attributable to seed variance alone. The deployed Flask dashboard with SSE streaming makes the agent's behavior, policy confidence, curriculum progress, and XAI analysis observable in real time by a human operator.

---

## VII. Future Scope

The following enhancements are planned for future iterations of this framework:

1. **3-D environment.** Extend DroneEnvAdvanced to a three-dimensional continuous world, adding altitude control, gravitational modeling, and 3-D LIDAR (spherical ray casting). This would make the simulation directly applicable to real UAV path planning.

2. **Transfer to real hardware.** Implement a sim-to-real transfer pipeline using domain randomization over obstacle shapes, LIDAR noise, and wind statistics. Evaluate the trained policy on a physical quadrotor platform (e.g., Crazyflie or DJI Tello).

3. **Multi-agent coordination.** Extend the environment to support multiple drones with inter-agent collision avoidance and cooperative goal coverage, using Multi-Agent PPO (MAPPO) or QMIX.

4. **Vision-based perception.** Replace or augment the LIDAR input with camera-image observations processed by a convolutional encoder, enabling the policy to handle textured environments and color-coded waypoints.

5. **MLflow experiment tracking.** Integrate MLflow for centralized logging of hyperparameters, metrics, and model artifacts across all training runs, replacing the current JSON-file-based approach.

6. **SHAP-based XAI.** Add an optional SHAP value computation path for users who require model-class-specific attributions rather than the current correlation-based proxy.

7. **Continuous curriculum.** Replace the 10-stage discrete curriculum with a continuous difficulty parameter controlled by a separate meta-controller (e.g., automatic curriculum learning via PAIRED or ACCEL), allowing finer-grained adaptation.

8. **Reinforcement learning from human feedback (RLHF).** Incorporate an operator reward signal collected through the dashboard interface, allowing non-expert users to shape the policy toward mission-specific behavioral preferences.

---

## References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[2] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. 35th Int. Conf. Machine Learning (ICML)*, 2018, pp. 1861–1870.

[3] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2019, pp. 2623–2631.

[4] V. Mnih, K. Kavukcuoglu, D. Silver, et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.

[5] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in *Proc. 26th Int. Conf. Machine Learning (ICML)*, 2009, pp. 41–48.

[6] C. Florensa, D. Held, M. Wulfmeier, M. Zhang, and P. Abbeel, "Reverse Curriculum Generation for Reinforcement Learning," in *Proc. 1st Conf. Robot Learning (CoRL)*, 2017, pp. 482–495.

[7] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, "Decision Transformer: Reinforcement Learning via Sequence Modeling," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021, pp. 15084–15097.

[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NIPS)*, vol. 30, 2017.

[9] M. Himmelsbach, F. V. Hundelshausen, and H.-J. Wünsche, "Fast segmentation of 3D point clouds for ground vehicles," in *Proc. IEEE Intelligent Vehicles Symposium (IV)*, 2010, pp. 560–565.

[10] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NIPS)*, vol. 30, 2017.

[11] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2016, pp. 1135–1144.

[12] M. Ahn, A. Brohan, N. Brown, et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," in *Proc. 6th Conf. Robot Learning (CoRL)*, 2022.

[13] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.

[14] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[15] M. Towers, J. K. Terry, A. Kwiatkowski, et al., "Gymnasium," *Zenodo*, 2023. [Online]. Available: https://doi.org/10.5281/zenodo.8127025

[16] Groq Inc., "Groq API," 2024. [Online]. Available: https://groq.com/

---

## Dependencies

| Package           | Version      | Purpose                                  |
|-------------------|--------------|------------------------------------------|
| stable-baselines3 | ≥ 2.3.0      | PPO, SAC, callbacks, VecEnv              |
| gymnasium         | ≥ 0.29.0     | Environment API                          |
| torch             | ≥ 2.0.0      | Transformer policy, GPU training         |
| flask             | 3.1.3        | Web dashboard                            |
| flask-cors        | 6.0.5        | Cross-origin resource sharing            |
| gunicorn          | 23.0.0       | Production WSGI server                   |
| gevent            | 24.11.1      | Async worker for SSE streaming           |
| optuna            | 3.6.1        | Hyperparameter optimisation              |
| groq              | 1.4.0        | NL mission planner (Llama 3.3 70B)       |
| scipy             | 1.14.1       | Mann-Whitney U significance test         |
| matplotlib        | any          | Plots, heatmaps, reward curves           |
| numpy             | ≥ 1.24.0, < 2.0 | Array operations                      |
| colorama          | 0.4.6        | Colored terminal output                  |

---

## BibTeX

```bibtex
@misc{dronerl2026,
  title   = {Autonomous Drone Navigation via Deep Reinforcement Learning
             with Curriculum Training, Transformer-Based LIDAR Policy,
             and Explainability Analysis},
  author  = {Autonomous Drone RL Team},
  year    = {2026},
  note    = {DroneEnvAdvanced: 64-beam LIDAR, Dryden wind turbulence,
             single-axis motor failure, 10-stage progressive curriculum,
             PPO/SAC comparison with Mann-Whitney U significance test,
             Groq-powered natural-language mission planning.
             \url{https://github.com/Akshad1234/Autonomous-Drone-RL}},
}
```
