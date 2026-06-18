"""
Hyperparameter optimisation using Optuna (TPE sampler + MedianPruner).

Searches PPO or SAC hyperparameters to maximise mean reward over a fixed
eval set.  Each trial trains for `timesteps_per_trial` steps, then evaluates
for `eval_episodes` deterministic episodes.

Usage:
    python -m train.hpo --algo ppo --trials 30
    python -m train.hpo --algo sac --trials 20 --timesteps 30000
    python train/run.py --hpo --algo ppo

Results saved to:
    logs/hpo_results_<algo>.json   — best params + all trial results
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone.envs.advanced import DomainConfig, DroneEnvAdvanced, ObstacleConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_ALGO: str = "ppo"
DEFAULT_N_TRIALS: int = 20
DEFAULT_TIMESTEPS_PER_TRIAL: int = 30_000   # short runs for speed
DEFAULT_EVAL_EPISODES: int = 30
N_STARTUP_TRIALS: int = 5   # Optuna random exploration before TPE

# Fixed eval config (Stage 3 difficulty — fast episodes, still meaningful)
EVAL_CONFIG = DomainConfig(
    obstacles=ObstacleConfig(n_static=6, n_moving=1, speed_max=0.05),
    wind_strength=0.02,
    motor_fail_prob=0.0,
    goal_dist_min=8.0,
    goal_dist_max=12.0,
)


# ---------------------------------------------------------------------------
# SEARCH SPACES
# ---------------------------------------------------------------------------

def _ppo_params(trial: Any) -> Dict[str, Any]:
    """Optuna suggests PPO hyperparameters."""
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    # Ensure batch_size <= n_steps
    if batch_size > n_steps:
        batch_size = n_steps // 2 or 32

    net_arch_key = trial.suggest_categorical(
        "net_arch", ["small", "medium", "large", "deep"]
    )
    net_arch = {
        "small":  [64, 64],
        "medium": [128, 128],
        "large":  [256, 256],
        "deep":   [256, 256, 128],
    }[net_arch_key]

    return {
        "learning_rate": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.8),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0]),
        "policy_kwargs": dict(net_arch=net_arch),
    }


def _sac_params(trial: Any) -> Dict[str, Any]:
    """Optuna suggests SAC hyperparameters."""
    net_arch_key = trial.suggest_categorical(
        "net_arch", ["small", "medium", "large"]
    )
    net_arch = {
        "small":  [64, 64],
        "medium": [128, 128],
        "large":  [256, 256],
    }[net_arch_key]

    return {
        "learning_rate": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "learning_starts": trial.suggest_categorical("learning_starts", [500, 1000, 2000]),
        "ent_coef": "auto",
        "policy_kwargs": dict(net_arch=net_arch),
    }


# ---------------------------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------------------------

def _make_objective(
    algo: str,
    timesteps_per_trial: int,
    eval_episodes: int,
    seed: int,
):
    """Return an Optuna objective function closure."""

    def objective(trial: Any) -> float:
        params = _ppo_params(trial) if algo == "ppo" else _sac_params(trial)

        def make_env() -> Monitor:
            return Monitor(DroneEnvAdvanced(config=EVAL_CONFIG))

        train_env = DummyVecEnv([make_env])
        eval_env = DummyVecEnv([make_env])

        try:
            if algo == "ppo":
                model = PPO(
                    "MlpPolicy", train_env, verbose=0, seed=seed, **params
                )
            else:
                model = SAC(
                    "MlpPolicy", train_env, verbose=0, seed=seed, **params
                )

            model.learn(total_timesteps=timesteps_per_trial)
            mean_reward, _ = evaluate_policy(
                model, eval_env,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                warn=False,
            )
        except Exception as e:
            print(f"  [HPO] Trial {trial.number} failed: {e}")
            return float("-inf")
        finally:
            train_env.close()
            eval_env.close()

        return float(mean_reward)

    return objective


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_hpo(
    algo: str = DEFAULT_ALGO,
    n_trials: int = DEFAULT_N_TRIALS,
    timesteps_per_trial: int = DEFAULT_TIMESTEPS_PER_TRIAL,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 42,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run Optuna HPO. Returns dict with best_params and all trial results.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    root = root_dir or Path(__file__).parent.parent
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"hpo_results_{algo}.json"

    print(
        f"\n[HPO] algo={algo.upper()}  trials={n_trials}  "
        f"timesteps_per_trial={timesteps_per_trial}  eval_eps={eval_episodes}"
    )
    print(f"      Results -> {out_path}")

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"drone_{algo}_hpo",
    )

    objective = _make_objective(algo, timesteps_per_trial, eval_episodes, seed)

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n[HPO] Best trial #{best.number}: reward={best.value:.2f}")
    print(f"      Params: {best.params}")
    print(f"      Elapsed: {elapsed:.0f}s ({elapsed/n_trials:.1f}s/trial)")

    all_results: List[Dict] = [
        {
            "trial": t.number,
            "value": t.value if t.value is not None else None,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    output: Dict[str, Any] = {
        "algo": algo,
        "best_trial": best.number,
        "best_reward": float(best.value),
        "best_params": best.params,
        "n_trials": n_trials,
        "timesteps_per_trial": timesteps_per_trial,
        "eval_episodes": eval_episodes,
        "elapsed_seconds": round(elapsed, 1),
        "all_trials": all_results,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"[HPO] Saved -> {out_path}")

    return output


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for drone RL")
    parser.add_argument("--algo", type=str, default=DEFAULT_ALGO, choices=["ppo", "sac"])
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS_PER_TRIAL)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    run_hpo(
        algo=args.algo,
        n_trials=args.trials,
        timesteps_per_trial=args.timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
