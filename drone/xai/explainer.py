"""
drone.xai.explainer — Explainability toolkit for trained RL policies.

Provides four analysis modes:

  1. feature_importance(n_samples)
       Correlation-based: sample N observations, get deterministic actions,
       compute |Pearson correlation| between each obs feature and each action
       component. Fast, model-agnostic, no SHAP dependency.

  2. action_entropy(obs)
       Shannon entropy of the policy distribution at a single observation.
       Low entropy = confident; high entropy = uncertain.
       Works for discrete (PPO on grid envs) and continuous (PPO/SAC on advanced env).

  3. explain_trajectory(trajectory)
       Natural-language summary of a recorded trajectory. Rule-based template,
       no LLM required. Suitable for dashboard display.

  4. whatif(obs, modifications)
       Query the policy on a user-modified observation and return the delta
       in action and confidence. Powers the dashboard what-if simulator.

Compatible with all SB3 algorithms and both discrete and continuous policies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_LIDAR_BEAMS: int = 64         # must match DroneEnvAdvanced
IMPORTANCE_EPSILON: float = 0.05  # finite-difference step size
NEAR_MISS_REWARD_THRESHOLD: float = -5.0   # reward below this → near-miss event
COLLISION_REWARD_THRESHOLD: float = -40.0  # reward below this → collision event


# ---------------------------------------------------------------------------
# FEATURE NAME REGISTRY
# ---------------------------------------------------------------------------

def _feature_names(obs_dim: int) -> List[str]:
    """
    Return human-readable names for each observation dimension.
    Matches the layout of DroneEnvAdvanced for 74-dim obs.
    Gracefully handles legacy envs (10-dim LIDAR, 4-dim coord).
    """
    n_lidar = min(N_LIDAR_BEAMS, obs_dim)
    names: List[str] = [f"lidar_{i:02d}" for i in range(n_lidar)]
    tail_dim = obs_dim - n_lidar
    tail_labels = [
        "goal_cos", "goal_sin", "goal_dist",
        "vel_x", "vel_y", "energy",
        "wind_x", "wind_y", "motor_x", "motor_y",
    ]
    names += tail_labels[:tail_dim]
    while len(names) < obs_dim:
        names.append(f"feat_{len(names)}")
    return names[:obs_dim]


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class XAIExplainer:
    """
    Model-agnostic explainability wrapper for SB3 policies.

    Works with PPO (discrete or continuous) and SAC.
    Does not require SHAP, LIME, or any extra ML libraries.
    """

    def __init__(
        self,
        model: BaseAlgorithm,
        obs_space: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.obs_space = obs_space or model.observation_space
        self._obs_dim: int = int(np.prod(self.obs_space.shape))
        self._action_dim: int = int(np.prod(model.action_space.shape)) \
            if hasattr(model.action_space, "shape") else 1
        self._device = model.device
        self._is_discrete: bool = not hasattr(model.action_space, "high")
        self.feature_names: List[str] = _feature_names(self._obs_dim)

    def __repr__(self) -> str:
        algo = type(self.model).__name__
        return (
            f"XAIExplainer(algo={algo}, obs_dim={self._obs_dim}, "
            f"action_dim={self._action_dim}, "
            f"discrete={self._is_discrete})"
        )

    # ------------------------------------------------------------------
    # 1. FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        n_samples: int = 300,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Correlation-based feature importance.

        Samples `n_samples` random observations from the observation space,
        queries the deterministic policy, then computes max |Pearson r| between
        each obs feature and each action component.

        Returns a dict {feature_name: importance_0_to_1} sorted descending.
        Pass top_k to return only the most important features.
        """
        # Sample random observations
        samples = np.array(
            [self.obs_space.sample() for _ in range(n_samples)],
            dtype=np.float32,
        )

        # Batch predict (SB3 accepts 2-D obs array)
        actions_list = []
        for s in samples:
            a, _ = self.model.predict(s, deterministic=True)
            actions_list.append(np.asarray(a, dtype=np.float32).flatten())
        actions = np.stack(actions_list)   # (N, action_dim)

        # |Pearson r| between each feature and each action component
        importance = np.zeros(self._obs_dim, dtype=np.float32)
        for feat_i in range(self._obs_dim):
            col = samples[:, feat_i]
            if col.std() < 1e-8:
                continue
            max_corr = 0.0
            for act_j in range(actions.shape[1]):
                if actions[:, act_j].std() < 1e-8:
                    continue
                corr = float(np.corrcoef(col, actions[:, act_j])[0, 1])
                if not math.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
            importance[feat_i] = max_corr

        # Normalise to [0, 1]
        max_val = importance.max()
        if max_val > 0:
            importance = importance / max_val

        result = {
            name: float(score)
            for name, score in zip(self.feature_names, importance)
        }
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        if top_k:
            result = dict(list(result.items())[:top_k])
        return result

    # ------------------------------------------------------------------
    # 2. ACTION ENTROPY
    # ------------------------------------------------------------------

    def action_entropy(self, obs: np.ndarray) -> float:
        """
        Shannon entropy of the policy distribution at a given observation.

        Discrete policy: H = -Σ p_i log p_i  (nats)
        Continuous policy: differential entropy from the Gaussian parameters.
        Returns 0.0 on any failure.
        """
        obs_np = np.asarray(obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[None]

        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(self._device)

        try:
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                entropy = dist.entropy().mean().item()
            return float(entropy)
        except Exception:
            pass

        # SAC fallback: compute entropy from actor Gaussian params
        try:
            with torch.no_grad():
                mean, log_std, _ = self.model.actor.get_action_dist_params(obs_t)
                # Gaussian entropy per dim: 0.5 * log(2πe * σ²)
                std = log_std.exp()
                entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std).sum(-1).mean()
                return float(entropy.item())
        except Exception:
            return 0.0

    def action_entropy_stats(
        self, n_samples: int = 200
    ) -> Dict[str, float]:
        """
        Run entropy over n_samples random observations.
        Returns mean, std, min, max.
        """
        entropies = []
        for _ in range(n_samples):
            obs = self.obs_space.sample()
            entropies.append(self.action_entropy(obs))
        arr = np.array(entropies)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # 3. NATURAL LANGUAGE TRAJECTORY EXPLANATION
    # ------------------------------------------------------------------

    def explain_trajectory(
        self, trajectory: List[Dict[str, Any]]
    ) -> str:
        """
        Produce a natural-language paragraph summarising a recorded trajectory.

        Each element of `trajectory` should be a dict with keys:
            step, obs, action, reward, done, info
        """
        if not trajectory:
            return "No trajectory data provided."

        total_steps = len(trajectory)
        total_reward = sum(float(t.get("reward", 0)) for t in trajectory)
        goal_reached = any(t.get("info", {}).get("goal_reached", False) for t in trajectory)

        # Count events from rewards
        rewards = [float(t.get("reward", 0)) for t in trajectory]
        collisions = sum(1 for r in rewards if r <= COLLISION_REWARD_THRESHOLD)
        near_misses = sum(
            1 for r in rewards
            if COLLISION_REWARD_THRESHOLD < r <= NEAR_MISS_REWARD_THRESHOLD
        )

        # Action statistics
        actions = []
        for t in trajectory:
            a = t.get("action")
            if a is not None:
                actions.append(np.asarray(a, dtype=np.float32).flatten())
        if actions:
            action_arr = np.stack(actions)
            mean_magnitude = float(np.linalg.norm(action_arr, axis=1).mean())
            action_smoothness = float(np.diff(action_arr, axis=0).std()) if len(actions) > 1 else 0.0
        else:
            mean_magnitude = 0.0
            action_smoothness = 0.0

        # Energy and wind from infos
        energies = [
            float(t.get("info", {}).get("energy", 1.0))
            for t in trajectory
            if "info" in t and t["info"]
        ]
        final_energy = energies[-1] if energies else 1.0
        energy_used = 1.0 - final_energy

        winds = []
        for t in trajectory:
            w = t.get("info", {}).get("wind")
            if w:
                winds.append(np.linalg.norm(w))
        mean_wind = float(np.mean(winds)) if winds else 0.0

        # Motor status
        motor_statuses = []
        for t in trajectory:
            m = t.get("info", {}).get("motor_ok")
            if m:
                motor_statuses.append(m)
        motor_failure = any(
            any(v < 0.5 for v in m) for m in motor_statuses
        ) if motor_statuses else False

        # Build explanation
        outcome = "reached the goal" if goal_reached else "did NOT reach the goal"
        smoothness_desc = (
            "smooth" if action_smoothness < 0.15
            else "moderate" if action_smoothness < 0.35
            else "aggressive"
        )
        efficiency = (
            "efficient" if mean_magnitude < 0.4
            else "moderate" if mean_magnitude < 0.7
            else "high-thrust"
        )
        wind_desc = (
            "calm conditions" if mean_wind < 0.02
            else f"light wind ({mean_wind:.3f} avg)"
            if mean_wind < 0.06
            else f"strong wind ({mean_wind:.3f} avg)"
        )

        lines = [
            f"The drone flew for {total_steps} steps and {outcome} "
            f"(total reward: {total_reward:+.1f}).",
        ]
        if collisions > 0:
            lines.append(
                f"It suffered {collisions} obstacle collision(s) — "
                f"bounce-back recovery was used each time."
            )
        if near_misses > 0:
            lines.append(f"There were {near_misses} near-miss event(s) within the danger radius.")
        if motor_failure:
            lines.append(
                "A motor axis failure was active during this episode, "
                "requiring asymmetric thrust compensation."
            )
        lines.append(
            f"Flight style: {smoothness_desc} control, {efficiency} thrust "
            f"(avg action magnitude {mean_magnitude:.2f}/1.0)."
        )
        lines.append(
            f"Environment: {wind_desc}. "
            f"Energy consumed: {energy_used:.0%} of budget."
        )
        if not goal_reached:
            lines.append("Suggestion: increase training timesteps or reduce stage difficulty.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # 4. WHAT-IF ANALYSIS
    # ------------------------------------------------------------------

    def whatif(
        self,
        obs: np.ndarray,
        modifications: Dict[int, float],
    ) -> Dict[str, Any]:
        """
        Query the policy on a modified observation and return the delta.

        Args:
            obs:           Base observation, shape (obs_dim,)
            modifications: {feature_index: new_value}  — features to change.

        Returns:
            {
              "baseline_action":  list[float],
              "modified_action":  list[float],
              "action_delta":     list[float],
              "delta_magnitude":  float,
              "baseline_entropy": float,
              "modified_entropy": float,
              "entropy_delta":    float,
              "modified_obs":     list[float],
              "changed_features": {feature_name: {from: float, to: float}},
            }
        """
        obs = np.asarray(obs, dtype=np.float32)
        modified = obs.copy()

        changed: Dict[str, Dict[str, float]] = {}
        for idx, val in modifications.items():
            if 0 <= idx < len(obs):
                changed[self.feature_names[idx]] = {
                    "from": float(obs[idx]),
                    "to": float(val),
                }
                modified[idx] = float(val)

        base_action, _ = self.model.predict(obs, deterministic=True)
        mod_action, _ = self.model.predict(modified, deterministic=True)

        base_action = np.asarray(base_action, dtype=np.float32).flatten()
        mod_action = np.asarray(mod_action, dtype=np.float32).flatten()
        delta = mod_action - base_action

        return {
            "baseline_action": base_action.tolist(),
            "modified_action": mod_action.tolist(),
            "action_delta": delta.tolist(),
            "delta_magnitude": float(np.linalg.norm(delta)),
            "baseline_entropy": self.action_entropy(obs),
            "modified_entropy": self.action_entropy(modified),
            "entropy_delta": self.action_entropy(modified) - self.action_entropy(obs),
            "modified_obs": modified.tolist(),
            "changed_features": changed,
        }
