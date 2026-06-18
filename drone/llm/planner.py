"""
drone.llm.planner — Natural-language mission planner powered by Groq API.

Translates a free-text mission description into a structured MissionSpec
that can be directly dispatched to DroneEnvAdvanced or train_advanced.

Flow:
    user text
      → MissionPlanner.plan(text)
          → Groq (llama-3.3-70b-versatile)
              → JSON extraction
          → MissionSpec (validated, clamped)
          → DomainConfig (ready for env or training)
      → logged to logs/mission_log.jsonl

Usage:
    export GROQ_API_KEY="gsk_..."
    python -m drone.llm.planner "Fly through a storm with 10 obstacles and 30% motor failure"

Requires:
    pip install groq>=0.11.0
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from drone.envs.advanced import DomainConfig, ObstacleConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "llama-3.3-70b-versatile"   # Groq model ID
FALLBACK_MODEL: str = "llama3-8b-8192"            # cheaper fallback
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.1                           # near-deterministic extraction
MISSION_LOG: str = "logs/mission_log.jsonl"

# Clamp bounds — values outside these are unsafe for the env
_CLAMP = dict(
    n_static=(0, 18),
    n_moving=(0, 10),
    wind_strength=(0.0, 0.14),
    motor_fail_prob=(0.0, 0.40),
    goal_dist_min=(6.0, 16.0),
    goal_dist_max=(8.0, 18.0),
)

# System prompt: define the JSON schema for the LLM
_SYSTEM_PROMPT = """\
You are a drone mission planner. The user gives you a natural-language mission description.
You must respond with ONLY a valid JSON object — no prose, no markdown, no extra keys.

JSON schema:
{
  "mission_name": "short name (3-6 words)",
  "n_static_obstacles": <integer 0-18>,
  "n_moving_obstacles": <integer 0-10>,
  "wind_strength": <float 0.0-0.14, where 0=calm, 0.05=gusty, 0.10=storm, 0.14=extreme>,
  "motor_fail_prob": <float 0.0-0.40, where 0=all motors OK, 0.1=10% chance of one axis failing>,
  "goal_dist_min": <float 6.0-16.0, minimum distance to goal>,
  "goal_dist_max": <float 8.0-18.0, maximum distance to goal — must be > goal_dist_min>,
  "algo": <"ppo" or "sac">,
  "reasoning": "one sentence explaining your parameter choices"
}

Guidelines:
- "easy mission" → few obstacles, calm wind, full motors, short goal
- "storm" → wind_strength ≥ 0.10
- "motor failure" or "damaged" → motor_fail_prob ≥ 0.15
- "long range" or "far goal" → goal_dist_min ≥ 12, goal_dist_max ≥ 15
- "crowded" or "obstacle course" → n_static ≥ 10, n_moving ≥ 3
- "combat" or "extreme" → max everything
- SAC is better for complex continuous tasks; PPO for simpler exploration
Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MissionSpec:
    """Structured mission parameters extracted from NL input."""
    mission_name: str
    n_static_obstacles: int
    n_moving_obstacles: int
    wind_strength: float
    motor_fail_prob: float
    goal_dist_min: float
    goal_dist_max: float
    algo: str
    reasoning: str
    raw_nl: str
    timestamp: float = dataclasses.field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"MissionSpec(name={self.mission_name!r}, "
            f"static={self.n_static_obstacles}, moving={self.n_moving_obstacles}, "
            f"wind={self.wind_strength:.3f}, motor_fail={self.motor_fail_prob:.2f}, "
            f"algo={self.algo})"
        )

    def to_domain_config(self) -> DomainConfig:
        """Convert to a DroneEnvAdvanced DomainConfig."""
        return DomainConfig(
            obstacles=ObstacleConfig(
                n_static=self.n_static_obstacles,
                n_moving=self.n_moving_obstacles,
            ),
            wind_strength=self.wind_strength,
            motor_fail_prob=self.motor_fail_prob,
            goal_dist_min=self.goal_dist_min,
            goal_dist_max=self.goal_dist_max,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def difficulty_label(spec: "MissionSpec") -> str:
        """Human-readable difficulty estimate."""
        score = (
            spec.n_static_obstacles / 18.0 * 3
            + spec.n_moving_obstacles / 10.0 * 3
            + spec.wind_strength / 0.14 * 2
            + spec.motor_fail_prob / 0.40 * 2
        )
        if score < 2:
            return "Easy"
        elif score < 4:
            return "Moderate"
        elif score < 7:
            return "Hard"
        else:
            return "Extreme"


# ---------------------------------------------------------------------------
# PARSER
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str, raw_nl: str) -> MissionSpec:
    """Extract JSON from LLM response and build MissionSpec."""

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    data: Dict[str, Any] = json.loads(match.group())

    def clamp(key: str, lo: float, hi: float, default: float) -> float:
        return float(max(lo, min(hi, data.get(key, default))))

    def clampi(key: str, lo: int, hi: int, default: int) -> int:
        return int(max(lo, min(hi, int(data.get(key, default)))))

    n_static = clampi("n_static_obstacles", *_CLAMP["n_static"], 4)
    n_moving = clampi("n_moving_obstacles", *_CLAMP["n_moving"], 1)
    wind = clamp("wind_strength", *_CLAMP["wind_strength"], 0.03)
    fail = clamp("motor_fail_prob", *_CLAMP["motor_fail_prob"], 0.0)
    dist_min = clamp("goal_dist_min", *_CLAMP["goal_dist_min"], 8.0)
    dist_max = clamp("goal_dist_max", *_CLAMP["goal_dist_max"], 12.0)

    # Ensure max > min
    if dist_max <= dist_min:
        dist_max = dist_min + 2.0

    algo = str(data.get("algo", "ppo")).lower()
    if algo not in ("ppo", "sac"):
        algo = "ppo"

    return MissionSpec(
        mission_name=str(data.get("mission_name", "Custom Mission")),
        n_static_obstacles=n_static,
        n_moving_obstacles=n_moving,
        wind_strength=wind,
        motor_fail_prob=fail,
        goal_dist_min=dist_min,
        goal_dist_max=dist_max,
        algo=algo,
        reasoning=str(data.get("reasoning", "")),
        raw_nl=raw_nl,
    )


# ---------------------------------------------------------------------------
# PLANNER
# ---------------------------------------------------------------------------

class MissionPlanner:
    """
    Natural-language → MissionSpec dispatcher using Groq API.

    Requires GROQ_API_KEY environment variable (or pass api_key directly).
    Falls back gracefully when Groq is unavailable (network error, no key).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        log_path: Optional[Path] = None,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self._root = root_dir or Path(__file__).parent.parent.parent
        self._log_path: Path = (
            log_path if log_path is not None
            else self._root / MISSION_LOG
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazy-loaded

    def __repr__(self) -> str:
        has_key = bool(self.api_key)
        return f"MissionPlanner(model={self.model!r}, api_key_set={has_key})"

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("groq package not installed. Run: pip install groq")
        return self._client

    def plan(self, nl_mission: str) -> MissionSpec:
        """
        Parse a natural-language mission into a MissionSpec.

        If the Groq API is unavailable (no key, network error), returns a
        safe default MissionSpec rather than crashing.
        """
        if not nl_mission.strip():
            raise ValueError("Empty mission description")

        try:
            spec = self._call_groq(nl_mission)
        except Exception as exc:
            print(f"[MissionPlanner] Groq call failed: {exc}. Using fallback.")
            spec = self._fallback_spec(nl_mission)

        self._log(spec)
        return spec

    def _call_groq(self, nl_mission: str) -> MissionSpec:
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Mission: {nl_mission}"},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        raw = response.choices[0].message.content or ""
        return _parse_llm_response(raw, nl_mission)

    def _fallback_spec(self, nl_mission: str) -> MissionSpec:
        """Rule-based fallback when Groq is unavailable."""
        text = nl_mission.lower()
        n_static = 4
        n_moving = 1
        wind = 0.03
        fail = 0.0
        goal_min = 8.0
        goal_max = 12.0
        algo = "ppo"

        if any(w in text for w in ("storm", "extreme", "combat")):
            n_static, n_moving, wind, fail = 14, 7, 0.12, 0.25
            goal_min, goal_max = 12.0, 16.0
        elif any(w in text for w in ("hard", "difficult", "challenging")):
            n_static, n_moving, wind, fail = 10, 4, 0.08, 0.10
            goal_min, goal_max = 10.0, 14.0
        elif any(w in text for w in ("easy", "simple", "beginner")):
            n_static, n_moving, wind, fail = 2, 0, 0.0, 0.0
            goal_min, goal_max = 6.0, 9.0
        if "sac" in text:
            algo = "sac"
        if "motor" in text or "failure" in text:
            fail = max(fail, 0.15)
        if "wind" in text:
            wind = max(wind, 0.06)
        if "long" in text or "far" in text:
            goal_min = max(goal_min, 12.0)
            goal_max = max(goal_max, 15.0)

        return MissionSpec(
            mission_name="Fallback Mission",
            n_static_obstacles=n_static,
            n_moving_obstacles=n_moving,
            wind_strength=wind,
            motor_fail_prob=fail,
            goal_dist_min=goal_min,
            goal_dist_max=goal_max,
            algo=algo,
            reasoning="Fallback rule-based parsing (Groq unavailable)",
            raw_nl=nl_mission,
        )

    def _log(self, spec: MissionSpec) -> None:
        entry = spec.to_dict()
        entry["difficulty"] = MissionSpec.difficulty_label(spec)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def list_missions(self) -> List[Dict[str, Any]]:
        """Return all logged missions as a list of dicts."""
        if not self._log_path.exists():
            return []
        missions = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    missions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return missions


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Groq mission planner CLI")
    parser.add_argument("mission", nargs="?",
                        default="Fly through a gusty urban canyon with 8 moving obstacles")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--list", action="store_true", help="List all logged missions")
    args = parser.parse_args()

    planner = MissionPlanner(model=args.model)
    print(repr(planner))

    if args.list:
        missions = planner.list_missions()
        print(f"{len(missions)} logged missions:")
        for m in missions[-5:]:
            print(f"  {m['mission_name']} [{m['difficulty']}] — {m['raw_nl'][:60]}")
    else:
        print(f"\nPlanning: {args.mission!r}")
        spec = planner.plan(args.mission)
        print(repr(spec))
        print(f"Difficulty: {MissionSpec.difficulty_label(spec)}")
        print(f"Reasoning: {spec.reasoning}")
        print(f"DomainConfig: {spec.to_domain_config()}")
