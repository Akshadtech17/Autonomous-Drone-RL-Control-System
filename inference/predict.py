"""
Inference module — loads a trained PPO model and exposes:
    get_action(obs)          -> int
    get_action_with_probs(obs) -> dict

Usage:
    python -m inference.predict --obs 0 0 9 9
    python -m inference.predict --lidar
    python -m inference.predict --help
"""

import argparse
import os
import numpy as np
import torch as th

from stable_baselines3 import PPO

ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_coord_model_path = os.path.join(BASE_DIR, "models", "drone_ppo.zip")
_lidar_model_path = os.path.join(BASE_DIR, "models", "lidar", "drone_ppo_lidar_final.zip")

_model_cache: dict = {}


def _load_model(path: str) -> PPO:
    if path not in _model_cache:
        _model_cache[path] = PPO.load(path)
    return _model_cache[path]


def get_action(observation, model_path: str = _coord_model_path) -> int:
    """Return the greedy action (int 0-3) for the given observation."""
    model = _load_model(model_path)
    action, _ = model.predict(np.array(observation, dtype=np.float32),
                               deterministic=True)
    return int(action)


def get_action_with_probs(observation, model_path: str = _coord_model_path) -> dict:
    """
    Return action + softmax probabilities for each action.

    Returns:
        {
          "action":      int,
          "action_name": str,
          "probs":       {"up": f, "down": f, "left": f, "right": f},
          "obs":         list[float]
        }
    """
    model = _load_model(model_path)
    obs_np = np.array(observation, dtype=np.float32)

    # greedy action
    action, _ = model.predict(obs_np, deterministic=True)
    action = int(action)

    # softmax distribution over actions
    obs_tensor = th.tensor(obs_np).float().unsqueeze(0).to(model.device)
    with th.no_grad():
        dist  = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.squeeze().cpu().tolist()

    return {
        "action":      action,
        "action_name": ACTION_NAMES[action],
        "probs": {
            ACTION_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
        "obs": obs_np.tolist(),
    }


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single inference step")
    parser.add_argument("--obs",   type=float, nargs="+",
                        default=[0.0, 0.0, 9.0, 9.0],
                        help="Observation vector (4 floats for coord, 10 for lidar)")
    parser.add_argument("--lidar", action="store_true",
                        help="Use LIDAR model")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model path")
    args = parser.parse_args()

    path = args.model or (_lidar_model_path if args.lidar else _coord_model_path)
    result = get_action_with_probs(args.obs, model_path=path)
    print(result)
