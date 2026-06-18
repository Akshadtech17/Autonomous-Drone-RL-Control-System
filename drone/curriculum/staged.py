"""
drone.curriculum.staged — 10-stage progressive curriculum for DroneEnvAdvanced.

Stage progression (1 = easiest → 10 = hardest):
  1  Clear Sky      — 2 static obs, no wind, no motor failure
  2  Light Traffic  — 4 static, whisper of wind
  3  First Mover    — 6 static + 1 moving, light wind
  4  Gusty          — 8 static + 2 moving, gusty wind
  5  Moderate       — 8 static + 3 moving, moderate wind
  6  Windy Course   — 10 static + 4 moving, wind + 5% motor failure
  7  Storm          — 12 static + 5 moving, strong wind + 10% failure
  8  Motor Trouble  — 14 static + 6 moving, 20% motor failure
  9  GPS Denied     — 16 static + 8 moving, strong wind + 30% failure
  10 Combat Zone    — 18 static + 10 moving, max wind + 40% failure

Auto-promote: rolling success rate > 80% over last 100 episodes.
Auto-demote:  rolling success rate < 30% over last 100 episodes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from drone.envs.advanced import DomainConfig, DroneEnvAdvanced, ObstacleConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EVAL_WINDOW: int = 100          # rolling window size for success rate
MIN_EPISODES_BEFORE_EVAL: int = 20   # don't promote/demote with fewer eps
MAX_HISTORY_STORED: int = 20    # stage-change events kept in state file


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Stage:
    """One curriculum stage with its environment config and thresholds."""

    number: int
    name: str
    config: DomainConfig
    promote_threshold: float = 0.80
    demote_threshold: float = 0.30

    def __repr__(self) -> str:
        obs = self.config.obstacles
        return (
            f"Stage({self.number}/10 '{self.name}' "
            f"obs={obs.n_static}s+{obs.n_moving}m "
            f"wind={self.config.wind_strength:.3f} "
            f"fail={self.config.motor_fail_prob:.0%})"
        )


@dataclasses.dataclass
class CurriculumState:
    """Mutable runtime state tracked by StagedCurriculum."""

    stage_idx: int = 0
    total_episodes: int = 0
    stage_episodes: int = 0
    recent_successes: List[int] = dataclasses.field(default_factory=list)
    stage_history: List[Dict] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CurriculumState(stage={self.stage_idx + 1}, "
            f"stage_eps={self.stage_episodes}, "
            f"sr={self.success_rate():.1%})"
        )

    def success_rate(self) -> float:
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)


# ---------------------------------------------------------------------------
# STAGE DEFINITIONS — all 10 stages in one place
# ---------------------------------------------------------------------------

def _build_stages() -> List[Stage]:
    return [
        Stage(
            1, "Clear Sky",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=2, n_moving=0, speed_min=0.0, speed_max=0.0),
                wind_strength=0.0, motor_fail_prob=0.0,
                goal_dist_min=5.0, goal_dist_max=9.0,
            ),
        ),
        Stage(
            2, "Light Traffic",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=4, n_moving=0, speed_min=0.0, speed_max=0.0),
                wind_strength=0.01, motor_fail_prob=0.0,
                goal_dist_min=7.0, goal_dist_max=11.0,
            ),
        ),
        Stage(
            3, "First Mover",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=6, n_moving=1, speed_max=0.05),
                wind_strength=0.02, motor_fail_prob=0.0,
                goal_dist_min=8.0, goal_dist_max=12.0,
            ),
        ),
        Stage(
            4, "Gusty",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=8, n_moving=2, speed_max=0.06),
                wind_strength=0.04, motor_fail_prob=0.0,
                goal_dist_min=9.0, goal_dist_max=13.0,
            ),
        ),
        Stage(
            5, "Moderate",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=8, n_moving=3, speed_max=0.07),
                wind_strength=0.06, motor_fail_prob=0.0,
                goal_dist_min=10.0, goal_dist_max=14.0,
            ),
        ),
        Stage(
            6, "Windy Obstacle Course",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=10, n_moving=4, speed_max=0.08),
                wind_strength=0.08, motor_fail_prob=0.05,
                goal_dist_min=10.0, goal_dist_max=15.0,
            ),
        ),
        Stage(
            7, "Storm",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=12, n_moving=5, speed_max=0.08),
                wind_strength=0.10, motor_fail_prob=0.10,
                goal_dist_min=11.0, goal_dist_max=15.0,
            ),
        ),
        Stage(
            8, "Motor Trouble",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=14, n_moving=6, speed_max=0.09),
                wind_strength=0.10, motor_fail_prob=0.20,
                goal_dist_min=12.0, goal_dist_max=16.0,
            ),
        ),
        Stage(
            9, "GPS Denied",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=16, n_moving=8, speed_max=0.09),
                wind_strength=0.12, motor_fail_prob=0.30,
                goal_dist_min=13.0, goal_dist_max=17.0,
            ),
        ),
        Stage(
            10, "Combat Zone",
            DomainConfig(
                obstacles=ObstacleConfig(n_static=18, n_moving=10, speed_min=0.05, speed_max=0.10),
                wind_strength=0.14, motor_fail_prob=0.40,
                goal_dist_min=14.0, goal_dist_max=18.0,
            ),
        ),
    ]


STAGES: List[Stage] = _build_stages()


# ---------------------------------------------------------------------------
# CALLBACK
# ---------------------------------------------------------------------------

class StagedCurriculum(BaseCallback):
    """
    SB3 callback that drives 10-stage progressive curriculum on DroneEnvAdvanced.

    Attach to model.learn() alongside other callbacks. Requires access to the
    underlying VecEnv to push DomainConfig updates at episode boundaries.
    """

    def __init__(
        self,
        vec_env: VecEnv,
        state_file: Optional[Path] = None,
        start_stage: int = 1,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.vec_env = vec_env
        self.state_file = state_file
        self.state = CurriculumState(stage_idx=max(0, start_stage - 1))
        self._apply_stage(self.current_stage, reason="init")

    def __repr__(self) -> str:
        return (
            f"StagedCurriculum(stage={self.current_stage.number}/10, "
            f"state={self.state})"
        )

    @property
    def current_stage(self) -> Stage:
        return STAGES[self.state.stage_idx]

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        if bool(dones[0]):
            goal_reached = bool(infos[0].get("goal_reached", False))
            self._record_episode(success=goal_reached)
        return True

    # ------------------------------------------------------------------
    def _record_episode(self, success: bool) -> None:
        s = self.state
        s.total_episodes += 1
        s.stage_episodes += 1
        s.recent_successes.append(int(success))
        if len(s.recent_successes) > EVAL_WINDOW:
            s.recent_successes.pop(0)

        if len(s.recent_successes) < MIN_EPISODES_BEFORE_EVAL:
            self._write_state()
            return

        sr = s.success_rate()
        stage = self.current_stage

        if sr > stage.promote_threshold and s.stage_idx < len(STAGES) - 1:
            s.stage_idx += 1
            s.stage_episodes = 0
            s.recent_successes.clear()
            self._apply_stage(STAGES[s.stage_idx], reason=f"promoted sr={sr:.1%}")

        elif sr < stage.demote_threshold and s.stage_idx > 0:
            s.stage_idx -= 1
            s.stage_episodes = 0
            s.recent_successes.clear()
            self._apply_stage(STAGES[s.stage_idx], reason=f"demoted sr={sr:.1%}")

        else:
            self._write_state()

    def _apply_stage(self, stage: Stage, reason: str = "") -> None:
        """Push the new DomainConfig to every env in the VecEnv."""
        for wrapped_env in self.vec_env.envs:
            inner = _unwrap(wrapped_env)
            if isinstance(inner, DroneEnvAdvanced):
                inner.set_config(stage.config)

        if self.verbose:
            print(
                f"[Curriculum] Stage {stage.number}/10 '{stage.name}' ({reason})"
                f" | static={stage.config.obstacles.n_static}"
                f" moving={stage.config.obstacles.n_moving}"
                f" wind={stage.config.wind_strength:.3f}"
                f" fail={stage.config.motor_fail_prob:.0%}"
            )

        self.state.stage_history.append({
            "stage": stage.number,
            "name": stage.name,
            "timestep": int(self.num_timesteps),
            "reason": reason,
            "success_rate": round(self.state.success_rate(), 4),
        })
        self._write_state()

    def _write_state(self) -> None:
        if self.state_file is None:
            return
        stage = self.current_stage
        payload: Dict = {
            "curriculum_stage": stage.number,
            "curriculum_stage_name": stage.name,
            "curriculum_stage_episodes": self.state.stage_episodes,
            "curriculum_success_rate": round(self.state.success_rate(), 4),
            "curriculum_total_episodes": self.state.total_episodes,
            "curriculum_stage_history": self.state.stage_history[-MAX_HISTORY_STORED:],
        }
        try:
            existing: Dict = {}
            if self.state_file.exists():
                existing = json.loads(self.state_file.read_text())
            existing.update(payload)
            self.state_file.write_text(json.dumps(existing))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _unwrap(env):
    """Peel Monitor and VecEnvWrapper layers to reach the raw environment."""
    while hasattr(env, "env"):
        env = env.env
    return env


def get_stage(number: int) -> Stage:
    """Return a Stage by 1-based number (1–10)."""
    if not 1 <= number <= 10:
        raise ValueError(f"Stage number must be 1–10, got {number}")
    return STAGES[number - 1]
