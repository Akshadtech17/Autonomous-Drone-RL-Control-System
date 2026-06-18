"""
Train PPO or SAC on DroneEnvAdvanced (continuous 74-dim obs, Box(2,) actions).

Saves to models/advanced/<algo>/

Usage:
    python -m train.train_advanced --algo ppo --timesteps 500000 --curriculum
    python -m train.train_advanced --algo sac --timesteps 300000 --seed 42
    python train/run.py --algo sac --env advanced --curriculum   # via master entrypoint
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone.envs.advanced import DomainConfig, DroneEnvAdvanced, ObstacleConfig
from drone.curriculum.staged import STAGES, StagedCurriculum

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS_PPO: int = 500_000
DEFAULT_TIMESTEPS_SAC: int = 300_000
DEFAULT_SEED: int = 42
EVAL_FREQ: int = 10_000
CKPT_FREQ: int = 25_000
METRICS_KEEP: int = 200

# Medium-difficulty eval config (Stage 5) — fixed so eval is always comparable
EVAL_CONFIG = DomainConfig(
    obstacles=ObstacleConfig(n_static=8, n_moving=3, speed_max=0.07),
    wind_strength=0.06,
    motor_fail_prob=0.0,
    goal_dist_min=10.0,
    goal_dist_max=14.0,
)

PPO_HYPERPARAMS: dict = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)

SAC_HYPERPARAMS: dict = dict(
    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=2_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# METRICS CALLBACK
# ---------------------------------------------------------------------------

class AdvancedMetricsCallback(BaseCallback):
    """Per-episode metrics to disk + live state for dashboard."""

    def __init__(
        self,
        metrics_file: Path,
        state_file: Path,
        algo: str,
    ) -> None:
        super().__init__()
        self.metrics_file = metrics_file
        self.state_file = state_file
        self.algo = algo
        self.ep_reward: float = 0.0
        self.ep_length: int = 0
        self.ep_count: int = 0
        self._flush_every: int = 10

    def __repr__(self) -> str:
        return f"AdvancedMetricsCallback(algo={self.algo}, ep={self.ep_count})"

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [{}])
        if rewards is None or dones is None:
            return True

        self.ep_reward += float(rewards[0])
        self.ep_length += 1

        if bool(dones[0]):
            self.ep_count += 1
            goal_reached = infos[0].get("goal_reached", False)
            energy = infos[0].get("energy", 1.0)
            marker = " [GOAL]" if goal_reached else ""
            print(
                f"[{self.algo.upper()}] Ep {self.ep_count:4d} | "
                f"Reward {self.ep_reward:8.2f} | "
                f"Len {self.ep_length:4d} | "
                f"Energy {energy:.2f}{marker}"
            )

            m = _load_json(self.metrics_file)
            m.setdefault("episodes", []).append(self.ep_count)
            m.setdefault("rewards", []).append(self.ep_reward)
            m.setdefault("lengths", []).append(self.ep_length)
            m.setdefault("successes", []).append(int(goal_reached))
            for k in ("episodes", "rewards", "lengths", "successes"):
                m[k] = m[k][-METRICS_KEEP:]

            if self.ep_count % self._flush_every == 0:
                _save_json(self.metrics_file, m)
                s = _load_json(self.state_file)
                s.update(
                    episode=self.ep_count,
                    reward=float(self.ep_reward),
                    episode_length=self.ep_length,
                    model_type=f"advanced_{self.algo}",
                    goal_reached=bool(goal_reached),
                )
                _save_json(self.state_file, s)

            self.ep_reward = 0.0
            self.ep_length = 0
        return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(
    algo: str = "ppo",
    timesteps: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    use_curriculum: bool = False,
    use_mlflow: bool = False,
    root_dir: Optional[Path] = None,
    policy_kwargs_override: Optional[dict] = None,
    domain_config_override: Optional[DomainConfig] = None,
) -> Path:
    """
    Train PPO or SAC on DroneEnvAdvanced.
    Returns path to the saved final model.

    policy_kwargs_override: if set, replaces the default net_arch policy_kwargs
        (use this to inject LidarTransformerExtractor)
    domain_config_override: if set, overrides the starting DomainConfig
        (used by the NL mission planner to inject a planned config)
    """
    algo = algo.lower()
    if algo not in ("ppo", "sac"):
        raise ValueError(f"algo must be 'ppo' or 'sac', got '{algo}'")

    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS_PPO if algo == "ppo" else DEFAULT_TIMESTEPS_SAC

    root = root_dir or Path(__file__).parent.parent
    log_dir = root / "logs"
    model_dir = root / "models" / "advanced" / algo
    best_dir = model_dir / "best_model"
    ckpt_dir = model_dir / "checkpoints"
    metrics_file = log_dir / f"metrics_advanced_{algo}.json"
    state_file = log_dir / "live_state.json"

    for d in (log_dir, model_dir, best_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        _save_json(metrics_file, {"episodes": [], "rewards": [], "lengths": [], "successes": []})

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Training env — use override if provided, else stage-based default
    if domain_config_override is not None:
        start_config = domain_config_override
    elif use_curriculum:
        start_config = STAGES[0].config
    else:
        start_config = STAGES[4].config

    def make_train_env() -> Monitor:
        return Monitor(DroneEnvAdvanced(config=start_config))

    def make_eval_env() -> Monitor:
        return Monitor(DroneEnvAdvanced(config=EVAL_CONFIG))

    train_vec = DummyVecEnv([make_train_env])
    eval_vec = DummyVecEnv([make_eval_env])

    # Callbacks
    metrics_cb = AdvancedMetricsCallback(metrics_file, state_file, algo)

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=str(ckpt_dir),
        name_prefix=f"drone_{algo}_advanced",
    )

    callbacks = [eval_cb, ckpt_cb, metrics_cb]

    if use_curriculum:
        curriculum_cb = StagedCurriculum(
            vec_env=train_vec,
            state_file=state_file,
            verbose=1,
        )
        callbacks.append(curriculum_cb)
        print(f"[Curriculum] 10-stage progressive curriculum enabled")

    # Model — policy_kwargs_override replaces the default net_arch dict entirely
    ppo_params = {**PPO_HYPERPARAMS}
    sac_params = {**SAC_HYPERPARAMS}
    if policy_kwargs_override is not None:
        ppo_params["policy_kwargs"] = policy_kwargs_override
        sac_params["policy_kwargs"] = policy_kwargs_override

    if algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_vec,
            tensorboard_log=str(log_dir),
            verbose=0,
            seed=seed,
            **ppo_params,
        )
    else:  # sac
        model = SAC(
            policy="MlpPolicy",
            env=train_vec,
            tensorboard_log=str(log_dir),
            verbose=0,
            seed=seed,
            **sac_params,
        )

    # MLflow (optional)
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            run_name = f"{algo}_advanced_{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            params = {
                "algo": algo,
                "seed": seed,
                "timesteps": timesteps,
                "curriculum": use_curriculum,
                "env": "DroneEnvAdvanced",
            }
            params.update(PPO_HYPERPARAMS if algo == "ppo" else SAC_HYPERPARAMS)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow_run = mlflow
            print(f"[MLflow] Run '{run_name}' started")
        except ImportError:
            print("[MLflow] Not installed — skipping")

    # Mark training active in state file
    s = _load_json(state_file)
    s["training"] = True
    s["algo"] = algo
    _save_json(state_file, s)

    tb_name = f"drone_{algo}_advanced" + ("_curriculum" if use_curriculum else "")
    print(
        f"\n[TRAIN ADVANCED] algo={algo.upper()}  timesteps={timesteps}  "
        f"seed={seed}  curriculum={use_curriculum}"
    )
    print(f"  obs_dim=74  action_dim=2  env=DroneEnvAdvanced")

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=tb_name,
    )

    s = _load_json(state_file)
    s["training"] = False
    _save_json(state_file, s)

    final_path = model_dir / f"drone_{algo}_advanced_final"
    model.save(str(final_path))

    print(f"\nADVANCED TRAINING COMPLETE ({algo.upper()})")
    print(f"   Final : {final_path}.zip")
    print(f"   Best  : {best_dir}/best_model.zip")
    print(f"   Ckpts : {ckpt_dir}/")

    # Log to MLflow
    if mlflow_run:
        try:
            m = _load_json(metrics_file)
            if m.get("rewards"):
                mlflow_run.log_metric("final_mean_reward", float(np.mean(m["rewards"][-50:])))
                succ = m.get("successes", [])
                if succ:
                    mlflow_run.log_metric("final_success_rate", float(np.mean(succ[-50:])))
            mlflow_run.log_artifact(str(final_path) + ".zip")
            mlflow_run.end_run()
        except Exception as e:
            print(f"[MLflow] Error: {e}")

    return Path(str(final_path) + ".zip")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO or SAC on DroneEnvAdvanced")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args, _ = parser.parse_known_args()
    main(
        algo=args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        use_curriculum=args.curriculum,
        use_mlflow=args.mlflow,
    )
