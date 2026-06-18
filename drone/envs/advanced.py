"""
DroneEnvAdvanced — publication-quality continuous drone navigation environment.

Observation space  Box(shape=(74,), dtype=float32):
  [0 :64]  64-beam LIDAR rays, normalized 0-1 by world diagonal
  [64:66]  goal direction unit vector (cos θ, sin θ)
  [66]     goal distance, normalized 0-1 by world diagonal
  [67:69]  current velocity (vx, vy), normalized by MAX_SPEED
  [69]     energy remaining, 0-1
  [70:72]  wind vector (wx, wy), normalized by MAX_WIND
  [72:74]  motor status per axis (1.0 = ok, 0.0 = failed)

Action space  Box(low=-1, high=1, shape=(2,)):
  [0]  target vx  (−1 = full left,  +1 = full right)
  [1]  target vy  (−1 = full down,  +1 = full up)

Reward shaping:
  progress  ×10.0   — per-step distance-to-goal reduction
  goal      +500    — episode terminates
  obstacle  −50     — bounce-back, no termination
  near-miss ≤−20    — graduated within 1.5× obstacle radius
  energy    −0.05 × |a|²
  smooth    −2.0  × |Δv|²  (jerk penalty for smooth trajectories)
  survival  −0.1   per step
  boundary  −5.0   on wall bounce

Compatible with: StagedCurriculum (set_config), all SB3 algorithms.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# CONFIG — all tuneable constants live here, never buried in methods
# ---------------------------------------------------------------------------

N_LIDAR_BEAMS: int = 64
OBS_DIM: int = 74          # 64 + 2 + 1 + 2 + 1 + 2 + 2
MAX_SPEED: float = 0.40    # cells per step
MAX_WIND: float = 0.15     # max wind magnitude for normalization
NEAR_MISS_FACTOR: float = 1.5  # near-miss radius = factor × obstacle.radius

# Reward coefficients
R_GOAL: float = 500.0
R_PROGRESS: float = 10.0
R_OBSTACLE: float = -50.0
R_NEAR_MISS_MAX: float = -20.0
R_ENERGY: float = -0.05
R_SMOOTH: float = -2.0
R_SURVIVAL: float = -0.1
R_BOUNDARY: float = -5.0
GOAL_RADIUS: float = 1.0   # cells — counts as "reached" if dist < this


@dataclasses.dataclass
class ObstacleConfig:
    """Obstacle distribution parameters for one curriculum stage."""

    n_static: int = 8
    n_moving: int = 4
    radius_min: float = 0.4
    radius_max: float = 0.9
    speed_min: float = 0.02
    speed_max: float = 0.08

    def __repr__(self) -> str:
        return (
            f"ObstacleConfig(static={self.n_static}, moving={self.n_moving}, "
            f"r=[{self.radius_min:.1f},{self.radius_max:.1f}], "
            f"v=[{self.speed_min:.2f},{self.speed_max:.2f}])"
        )


@dataclasses.dataclass
class DomainConfig:
    """Full domain randomization config for one curriculum stage."""

    grid_size: float = 20.0
    obstacles: ObstacleConfig = dataclasses.field(default_factory=ObstacleConfig)
    wind_strength: float = 0.05       # noise amplitude in Gauss-Markov process
    wind_correlation: float = 0.95    # autocorrelation α
    motor_fail_prob: float = 0.0      # per-episode P(one axis fails)
    energy_budget: float = 1.0        # starting energy (normalised)
    energy_rate: float = 0.003        # depletion = rate × sum(|action|) per step
    goal_dist_min: float = 10.0       # domain-randomised goal distance range
    goal_dist_max: float = 16.0

    def __repr__(self) -> str:
        return (
            f"DomainConfig(grid={self.grid_size}, "
            f"wind={self.wind_strength:.3f}, "
            f"motor_fail={self.motor_fail_prob:.2f}, "
            f"obstacles={self.obstacles})"
        )


@dataclasses.dataclass
class ObstacleState:
    """Runtime state of a single obstacle."""

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ObstacleState(pos=({self.x:.1f},{self.y:.1f}), "
            f"r={self.radius:.2f}, v=({self.vx:.3f},{self.vy:.3f}))"
        )


# ---------------------------------------------------------------------------
# RAY GEOMETRY HELPERS (module-level for speed, no class overhead)
# ---------------------------------------------------------------------------

def _ray_circle_intersection(
    ox: float, oy: float,
    dx: float, dy: float,
    cx: float, cy: float,
    r: float,
) -> Optional[float]:
    """First positive t where ray (origin + t*dir) enters circle. None if no hit."""
    fx = ox - cx
    fy = oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t = (-b - sq) * 0.5
    if t > 1e-6:
        return t
    t = (-b + sq) * 0.5
    if t > 1e-6:
        return t
    return None


def _ray_aabb_t(
    ox: float, oy: float,
    dx: float, dy: float,
    xmin: float, ymin: float,
    xmax: float, ymax: float,
) -> float:
    """Smallest positive t where ray hits the AABB boundary from inside."""
    t_min = float("inf")
    eps = 1e-9

    if abs(dx) > eps:
        for wall_x in (xmin, xmax):
            t = (wall_x - ox) / dx
            if t > eps:
                hy = oy + t * dy
                if ymin <= hy <= ymax:
                    t_min = min(t_min, t)

    if abs(dy) > eps:
        for wall_y in (ymin, ymax):
            t = (wall_y - oy) / dy
            if t > eps:
                hx = ox + t * dx
                if xmin <= hx <= xmax:
                    t_min = min(t_min, t)

    return t_min


# ---------------------------------------------------------------------------
# MAIN ENVIRONMENT
# ---------------------------------------------------------------------------

class DroneEnvAdvanced(gym.Env):
    """
    Advanced continuous-space drone navigation with 64-beam LIDAR,
    Dryden wind, motor failure, moving obstacles, and energy tracking.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or DomainConfig()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._world_diag: float = math.sqrt(2.0) * self.config.grid_size

        # Action space: target velocity command (vx, vy) ∈ [−1, 1]²
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: [LIDAR(64), goal_dir(2), goal_dist(1),
        #                     vel(2), energy(1), wind(2), motor(2)]
        low = np.concatenate([
            np.zeros(N_LIDAR_BEAMS, dtype=np.float32),   # LIDAR ≥ 0
            np.full(2, -1.0, dtype=np.float32),           # goal_dir
            np.zeros(1, dtype=np.float32),                 # goal_dist
            np.full(2, -1.0, dtype=np.float32),           # vel
            np.zeros(1, dtype=np.float32),                 # energy
            np.full(2, -1.0, dtype=np.float32),           # wind
            np.zeros(2, dtype=np.float32),                 # motor
        ])
        high = np.concatenate([
            np.ones(N_LIDAR_BEAMS, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Runtime state (initialised in reset)
        self.drone_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.drone_vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.obstacles: List[ObstacleState] = []
        self.wind: np.ndarray = np.zeros(2, dtype=np.float32)
        self.motor_ok: np.ndarray = np.ones(2, dtype=np.float32)
        self.energy: float = 1.0
        self.steps: int = 0
        self._prev_vel: np.ndarray = np.zeros(2, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"DroneEnvAdvanced(obs_dim={OBS_DIM}, action_dim=2, "
            f"max_steps={self.max_steps}, config={self.config})"
        )

    # ------------------------------------------------------------------
    # PUBLIC API — used by curriculum and dashboard
    # ------------------------------------------------------------------

    def set_config(self, config: DomainConfig) -> None:
        """Hot-swap domain config (called by StagedCurriculum between episodes)."""
        self.config = config
        self._world_diag = math.sqrt(2.0) * config.grid_size

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        gs = cfg.grid_size

        self.steps = 0
        self.energy = float(cfg.energy_budget)
        self.wind = np.zeros(2, dtype=np.float32)
        self.drone_vel = np.zeros(2, dtype=np.float32)
        self._prev_vel = np.zeros(2, dtype=np.float32)

        # Domain-randomised start (lower-left quadrant, near origin)
        self.drone_pos = np.array(
            [
                random.uniform(0.5, gs * 0.15),
                random.uniform(0.5, gs * 0.15),
            ],
            dtype=np.float32,
        )

        # Domain-randomised goal (upper-right quadrant, varying distance)
        for _ in range(50):
            dist = random.uniform(cfg.goal_dist_min, cfg.goal_dist_max)
            angle = random.uniform(math.pi * 0.15, math.pi * 0.65)
            gx = self.drone_pos[0] + dist * math.cos(angle)
            gy = self.drone_pos[1] + dist * math.sin(angle)
            if 1.0 <= gx <= gs - 1.0 and 1.0 <= gy <= gs - 1.0:
                self.goal_pos = np.array([gx, gy], dtype=np.float32)
                break
        else:
            self.goal_pos = np.array([gs * 0.85, gs * 0.85], dtype=np.float32)

        # Motor failure injection
        if random.random() < cfg.motor_fail_prob:
            self.motor_ok = np.ones(2, dtype=np.float32)
            self.motor_ok[random.randint(0, 1)] = 0.0
        else:
            self.motor_ok = np.ones(2, dtype=np.float32)

        # Generate obstacles
        self.obstacles = self._generate_obstacles()

        return self._get_obs(), {"config": str(cfg)}

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        prev_pos = self.drone_pos.copy()
        prev_dist = float(np.linalg.norm(self.goal_pos - self.drone_pos))
        self._prev_vel = self.drone_vel.copy()

        # Physics updates
        self._update_wind()
        self._move_obstacles()

        # Motor failure masks one axis
        effective_action = action * self.motor_ok

        # Drone velocity: inertia (75%) + commanded (25%), scaled to MAX_SPEED
        self.drone_vel = (
            0.75 * self.drone_vel + 0.25 * effective_action * MAX_SPEED
        ).astype(np.float32)

        # Proposed new position (velocity + wind disturbance)
        new_pos = (self.drone_pos + self.drone_vel + self.wind).astype(np.float32)

        reward: float = 0.0
        terminated: bool = False
        gs = self.config.grid_size

        # Boundary bounce
        bounced = False
        for axis in range(2):
            if new_pos[axis] < 0.0:
                new_pos[axis] = 0.01
                self.drone_vel[axis] = abs(self.drone_vel[axis]) * 0.5
                bounced = True
            elif new_pos[axis] > gs:
                new_pos[axis] = gs - 0.01
                self.drone_vel[axis] = -abs(self.drone_vel[axis]) * 0.5
                bounced = True
        if bounced:
            reward += R_BOUNDARY

        # Obstacle collision → bounce back, keep running
        if self._check_collision(new_pos):
            new_pos = prev_pos.copy()
            self.drone_vel = np.zeros(2, dtype=np.float32)
            reward += R_OBSTACLE
        else:
            self.drone_pos = new_pos

        # Near-miss graduated penalty
        for obs in self.obstacles:
            d = math.sqrt(
                (self.drone_pos[0] - obs.x) ** 2 + (self.drone_pos[1] - obs.y) ** 2
            )
            near_r = obs.radius * NEAR_MISS_FACTOR
            if obs.radius < d < near_r:
                closeness = 1.0 - (d - obs.radius) / (near_r - obs.radius)
                reward += R_NEAR_MISS_MAX * closeness

        # Energy depletion and penalty
        e_used = float(np.sum(np.abs(action))) * self.config.energy_rate
        self.energy = max(0.0, self.energy - e_used)
        reward += float(np.sum(action**2)) * R_ENERGY

        # Smooth flight: penalise jerk (change in velocity)
        delta_vel = self.drone_vel - self._prev_vel
        reward += float(np.sum(delta_vel**2)) * R_SMOOTH

        # Progress toward goal
        new_dist = float(np.linalg.norm(self.goal_pos - self.drone_pos))
        reward += (prev_dist - new_dist) * R_PROGRESS

        # Survival penalty (encourages efficiency)
        reward += R_SURVIVAL

        # Goal reached
        if new_dist < GOAL_RADIUS:
            reward += R_GOAL
            terminated = True

        # Max steps
        if self.steps >= self.max_steps:
            terminated = True

        info: Dict[str, Any] = {
            "dist_to_goal": new_dist,
            "energy": self.energy,
            "motor_ok": self.motor_ok.tolist(),
            "wind": self.wind.tolist(),
            "n_obstacles": len(self.obstacles),
            "step": self.steps,
            "goal_reached": new_dist < GOAL_RADIUS,
        }
        return self._get_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # RENDERING HELPERS (used by dashboard and Pygame sim)
    # ------------------------------------------------------------------

    def get_lidar_endpoints(self) -> List[Tuple[float, float]]:
        """Return 64 (x, y) world-space ray endpoints for visualisation."""
        diag = self._world_diag
        gs = self.config.grid_size
        ox, oy = float(self.drone_pos[0]), float(self.drone_pos[1])
        endpoints: List[Tuple[float, float]] = []

        for i in range(N_LIDAR_BEAMS):
            angle = 2.0 * math.pi * i / N_LIDAR_BEAMS
            dx = math.cos(angle)
            dy = math.sin(angle)
            t = _ray_aabb_t(ox, oy, dx, dy, 0.0, 0.0, gs, gs)
            for obs in self.obstacles:
                t_hit = _ray_circle_intersection(ox, oy, dx, dy, obs.x, obs.y, obs.radius)
                if t_hit is not None and 0.0 < t_hit < t:
                    t = t_hit
            endpoints.append((ox + dx * min(t, diag), oy + dy * min(t, diag)))

        return endpoints

    def get_obstacle_info(self) -> List[Dict[str, float]]:
        """Serialisable obstacle data for the dashboard SSE stream."""
        return [
            {"x": o.x, "y": o.y, "radius": o.radius, "vx": o.vx, "vy": o.vy}
            for o in self.obstacles
        ]

    # ------------------------------------------------------------------
    # INTERNAL METHODS
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        diag = self._world_diag
        rays = self._cast_rays()

        goal_vec = self.goal_pos - self.drone_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-6:
            goal_dir = (goal_vec / goal_dist).astype(np.float32)
        else:
            goal_dir = np.array([1.0, 0.0], dtype=np.float32)

        obs = np.concatenate([
            rays,
            goal_dir,
            np.array([goal_dist / diag], dtype=np.float32),
            (self.drone_vel / MAX_SPEED).astype(np.float32),
            np.array([self.energy], dtype=np.float32),
            (self.wind / MAX_WIND).astype(np.float32),
            self.motor_ok,
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _cast_rays(self) -> np.ndarray:
        """64-beam LIDAR: returns normalised distances [0, 1]."""
        diag = self._world_diag
        gs = self.config.grid_size
        ox, oy = float(self.drone_pos[0]), float(self.drone_pos[1])
        rays = np.empty(N_LIDAR_BEAMS, dtype=np.float32)

        for i in range(N_LIDAR_BEAMS):
            angle = 2.0 * math.pi * i / N_LIDAR_BEAMS
            dx = math.cos(angle)
            dy = math.sin(angle)
            t = _ray_aabb_t(ox, oy, dx, dy, 0.0, 0.0, gs, gs)
            for obs in self.obstacles:
                t_hit = _ray_circle_intersection(ox, oy, dx, dy, obs.x, obs.y, obs.radius)
                if t_hit is not None and 0.0 < t_hit < t:
                    t = t_hit
            rays[i] = float(min(t, diag)) / diag

        return rays

    def _generate_obstacles(self) -> List[ObstacleState]:
        cfg = self.config
        obs_cfg = cfg.obstacles
        gs = cfg.grid_size
        result: List[ObstacleState] = []
        n_total = obs_cfg.n_static + obs_cfg.n_moving

        for attempt in range(n_total * 30):
            if len(result) >= n_total:
                break
            r = random.uniform(obs_cfg.radius_min, obs_cfg.radius_max)
            x = random.uniform(r + 1.0, gs - r - 1.0)
            y = random.uniform(r + 1.0, gs - r - 1.0)

            # Clearance from drone start
            if math.hypot(x - self.drone_pos[0], y - self.drone_pos[1]) < 3.0 + r:
                continue
            # Clearance from goal
            if math.hypot(x - self.goal_pos[0], y - self.goal_pos[1]) < 2.5 + r:
                continue
            # No overlap with existing
            if any(
                math.hypot(x - o.x, y - o.y) < r + o.radius + 0.3
                for o in result
            ):
                continue

            is_moving = len(result) >= obs_cfg.n_static
            if is_moving:
                speed = random.uniform(obs_cfg.speed_min, obs_cfg.speed_max)
                angle = random.uniform(0.0, 2.0 * math.pi)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)
            else:
                vx = vy = 0.0

            result.append(ObstacleState(x=x, y=y, radius=r, vx=vx, vy=vy))

        return result

    def _check_collision(self, pos: np.ndarray) -> bool:
        return any(
            math.hypot(pos[0] - o.x, pos[1] - o.y) < o.radius
            for o in self.obstacles
        )

    def _update_wind(self) -> None:
        cfg = self.config
        noise = np.random.normal(0.0, cfg.wind_strength, 2).astype(np.float32)
        self.wind = (cfg.wind_correlation * self.wind + (1.0 - cfg.wind_correlation) * noise)
        self.wind = np.clip(self.wind, -MAX_WIND, MAX_WIND).astype(np.float32)

    def _move_obstacles(self) -> None:
        gs = self.config.grid_size
        for obs in self.obstacles:
            if obs.vx == 0.0 and obs.vy == 0.0:
                continue
            obs.x += obs.vx
            obs.y += obs.vy
            if obs.x - obs.radius <= 0.0:
                obs.x = obs.radius + 0.01
                obs.vx = abs(obs.vx)
            elif obs.x + obs.radius >= gs:
                obs.x = gs - obs.radius - 0.01
                obs.vx = -abs(obs.vx)
            if obs.y - obs.radius <= 0.0:
                obs.y = obs.radius + 0.01
                obs.vy = abs(obs.vy)
            elif obs.y + obs.radius >= gs:
                obs.y = gs - obs.radius - 0.01
                obs.vy = -abs(obs.vy)

    def render(self) -> None:
        gs = int(self.config.grid_size)
        grid = [["." for _ in range(gs)] for _ in range(gs)]

        gx, gy = int(self.goal_pos[0]), int(min(gs - 1, self.goal_pos[1]))
        dx, dy = int(self.drone_pos[0]), int(min(gs - 1, self.drone_pos[1]))
        if 0 <= gy < gs and 0 <= gx < gs:
            grid[gy][gx] = "G"
        if 0 <= dy < gs and 0 <= dx < gs:
            grid[dy][dx] = "D"

        for obs in self.obstacles:
            for ry in range(max(0, int(obs.y - obs.radius) - 1), min(gs, int(obs.y + obs.radius) + 2)):
                for rx in range(max(0, int(obs.x - obs.radius) - 1), min(gs, int(obs.x + obs.radius) + 2)):
                    if math.hypot(rx - obs.x, ry - obs.y) <= obs.radius:
                        grid[ry][rx] = "X"

        print(
            f"\nStep {self.steps:3d} | "
            f"dist={np.linalg.norm(self.goal_pos - self.drone_pos):.2f} | "
            f"energy={self.energy:.2f} | "
            f"wind=({self.wind[0]:.3f},{self.wind[1]:.3f}) | "
            f"motor={self.motor_ok.tolist()}"
        )
        for row in reversed(grid):   # y=0 at bottom like a proper coordinate system
            print(" ".join(row))


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = DroneEnvAdvanced()
    print(repr(env))
    obs, info = env.reset(seed=42)
    print(f"obs shape: {obs.shape}  obs[:5]: {obs[:5].round(3)}")
    print(f"action_space: {env.action_space}")
    print(f"obstacles: {len(env.obstacles)}")
    env.render()

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"step={step+1}  reward={reward:.2f}  dist={info['dist_to_goal']:.2f}  done={done}")
        if done:
            obs, _ = env.reset()
            total_reward = 0.0
    print("DroneEnvAdvanced OK")
