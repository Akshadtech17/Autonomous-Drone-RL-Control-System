"""
Train PPO on DroneEnvLidar (8-ray LIDAR + goal direction, Box(10,)).
Saves to models/lidar/ — does NOT touch existing exact-coord models.

Usage:
    python -m train.train_lidar [--timesteps 400000] [--seed 42] [--obstacles 15]
    python train/run.py --algo ppo --env lidar          # via master entrypoint
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.drone_env_lidar import DroneEnvLidar

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS: int = 400_000
DEFAULT_SEED: int = 42
DEFAULT_OBSTACLES: int = 15
PPO_HYPERPARAMS: dict = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)
EVAL_FREQ: int = 5_000
CKPT_FREQ: int = 10_000
METRICS_KEEP: int = 200


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
# CALLBACK
# ---------------------------------------------------------------------------
class LiveMetricsCallback(BaseCallback):
    """Logs per-episode metrics to disk for the LIDAR model dashboard."""

    def __init__(self, metrics_file: Path, state_file: Path) -> None:
        super().__init__()
        self.metrics_file = metrics_file
        self.state_file = state_file
        self.ep_reward: float = 0.0
        self.ep_length: int = 0
        self.ep_count: int = 0
        self._flush_every: int = 10

    def __repr__(self) -> str:
        return f"LiveMetricsCallback(lidar, ep={self.ep_count})"

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        self.ep_reward += float(rewards[0])
        self.ep_length += 1

        if bool(dones[0]):
            self.ep_count += 1
            print(
                f"[LIDAR] Ep {self.ep_count:4d} | "
                f"Reward {self.ep_reward:7.2f} | Len {self.ep_length:3d}"
            )

            m = _load_json(self.metrics_file)
            m.setdefault("episodes", []).append(self.ep_count)
            m.setdefault("rewards", []).append(self.ep_reward)
            m.setdefault("lengths", []).append(self.ep_length)
            for k in ("episodes", "rewards", "lengths"):
                m[k] = m[k][-METRICS_KEEP:]

            if self.ep_count % self._flush_every == 0:
                _save_json(self.metrics_file, m)
                s = _load_json(self.state_file)
                s.update(
                    episode=self.ep_count,
                    reward=float(self.ep_reward),
                    episode_length=self.ep_length,
                    model_type="lidar",
                )
                _save_json(self.state_file, s)

            self.ep_reward = 0.0
            self.ep_length = 0
        return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(
    timesteps: int = DEFAULT_TIMESTEPS,
    seed: int = DEFAULT_SEED,
    n_obstacles: int = DEFAULT_OBSTACLES,
    root_dir: Optional[Path] = None,
) -> Path:
    """Train PPO on DroneEnvLidar. Returns path to the saved final model."""
    root = root_dir or Path(__file__).parent.parent
    log_dir = root / "logs"
    model_dir = root / "models" / "lidar"
    best_dir = model_dir / "best_model"
    ckpt_dir = model_dir / "checkpoints"
    metrics_file = log_dir / "metrics_lidar.json"
    state_file = log_dir / "live_state.json"

    for d in (log_dir, model_dir, best_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        _save_json(metrics_file, {"episodes": [], "rewards": [], "lengths": []})

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env() -> Monitor:
        return Monitor(DroneEnvLidar(n_obstacles=n_obstacles))

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(log_dir),
            eval_freq=EVAL_FREQ,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=CKPT_FREQ,
            save_path=str(ckpt_dir),
            name_prefix="drone_ppo_lidar",
        ),
        LiveMetricsCallback(metrics_file, state_file),
    ]

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=seed,
        **PPO_HYPERPARAMS,
    )

    print(f"\n[LIDAR TRAIN] timesteps={timesteps}  seed={seed}  obstacles={n_obstacles}")

    s = _load_json(state_file)
    s["training"] = True
    _save_json(state_file, s)

    model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name="drone_rl_lidar")

    s = _load_json(state_file)
    s["training"] = False
    _save_json(state_file, s)

    final_path = model_dir / "drone_ppo_lidar_final"
    model.save(str(final_path))
    print(f"\nLIDAR TRAINING COMPLETE")
    print(f"   Final : {final_path}.zip")
    print(f"   Best  : {best_dir}")
    print(f"   Ckpts : {ckpt_dir}")

    return Path(f"{final_path}.zip")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on DroneEnvLidar")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--obstacles", type=int, default=DEFAULT_OBSTACLES)
    args, _ = parser.parse_known_args()
    main(timesteps=args.timesteps, seed=args.seed, n_obstacles=args.obstacles)
