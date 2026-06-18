"""
CurriculumCallback — progressively increases obstacle count during training.

Starts at n_obstacles=3, adds +2 every step_interval timesteps, capped at max_obstacles=25.
Works with both DroneEnv (coord) and DroneEnvLidar.

Usage in training script:
    from train.curriculum import CurriculumCallback
    cb = CurriculumCallback(vec_env, state_file=STATE_FILE)
    model.learn(..., callback=[..., cb])
"""

import json
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class CurriculumCallback(BaseCallback):
    """
    Increments n_obstacles on all envs in a VecEnv every `step_interval` steps.

    Also writes current difficulty to `state_file` so the dashboard can display it.
    """

    def __init__(
        self,
        vec_env: VecEnv,
        start_obstacles: int = 3,
        max_obstacles:   int = 25,
        step_interval:   int = 25_000,
        state_file=None,
        verbose:         int = 1,
    ):
        super().__init__(verbose)
        self.vec_env        = vec_env
        self.start_obstacles = start_obstacles
        self.max_obstacles   = max_obstacles
        self.step_interval   = step_interval
        self.state_file      = state_file
        self.current_n       = start_obstacles

        # apply initial difficulty immediately
        self._set_obstacles(start_obstacles)

    # ------------------------------------------------------------------
    def _set_obstacles(self, n: int):
        for wrapped_env in self.vec_env.envs:
            inner = self._unwrap(wrapped_env)
            if hasattr(inner, "n_obstacles"):
                inner.n_obstacles = n
        if self.verbose:
            print(f"[Curriculum] n_obstacles -> {n}")
        self._write_state(n)

    @staticmethod
    def _unwrap(env):
        """Unwrap Monitor / other wrappers to get the raw env."""
        while hasattr(env, "env"):
            env = env.env
        return env

    def _write_state(self, n: int):
        if self.state_file is None:
            return
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    s = json.load(f)
            else:
                s = {}
            s["n_obstacles"] = n
            with open(self.state_file, "w") as f:
                json.dump(s, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        n_increases  = self.num_timesteps // self.step_interval
        target_n     = min(self.start_obstacles + n_increases * 2, self.max_obstacles)
        if target_n != self.current_n:
            self.current_n = target_n
            self._set_obstacles(target_n)
        return True


if __name__ == "__main__":
    print("CurriculumCallback — not a standalone script.")
    print("Import and use inside a training script.")
    print("  from train.curriculum import CurriculumCallback")
