import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0,  1), (-1, 1), (-1, 0), (-1, -1),
]
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class DroneEnvLidar(gym.Env):
    """
    10x10 grid drone env with 8-ray LIDAR observation.
    obs = [ray_N, ray_NE, ..., ray_NW, goal_dx, goal_dy]  -> Box(10,)
    rays are normalized 0-1 by max_dist=9; goal deltas normalized by max_dist.
    """

    def __init__(self, n_obstacles=15):
        super().__init__()
        self.grid_size = 10
        self.max_steps = 200
        self.n_obstacles = n_obstacles

        self.action_space = spaces.Discrete(4)

        low  = np.array([0.0]*8 + [-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0]*10,                dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.drone_pos  = np.array([0, 0], dtype=np.int32)
        self.goal_pos   = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)
        self.obstacles  = set()
        self.steps      = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps     = 0
        self.drone_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos  = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)
        self.obstacles = self._generate_obstacles()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _generate_obstacles(self):
        obstacles = set()
        attempts  = 0
        while len(obstacles) < self.n_obstacles and attempts < self.n_obstacles * 10:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in [(0, 0), tuple(self.goal_pos)]:
                obstacles.add((x, y))
            attempts += 1
        return obstacles

    # ------------------------------------------------------------------
    def _cast_rays(self):
        """Return list of 8 normalized ray distances."""
        max_dist = self.grid_size - 1  # 9
        rays = []
        for dx, dy in DIRECTIONS:
            dist = 0
            x, y = int(self.drone_pos[0]), int(self.drone_pos[1])
            while dist < max_dist:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                    break
                if (nx, ny) in self.obstacles:
                    dist += 1
                    break
                x, y = nx, ny
                dist += 1
            rays.append(dist / max_dist)
        return rays

    def get_lidar_endpoints(self):
        """Return list of 8 (end_x, end_y) grid coords for rendering."""
        max_dist = self.grid_size - 1
        endpoints = []
        for dx, dy in DIRECTIONS:
            dist = 0
            x, y = int(self.drone_pos[0]), int(self.drone_pos[1])
            while dist < max_dist:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                    break
                if (nx, ny) in self.obstacles:
                    x, y = nx, ny
                    break
                x, y = nx, ny
                dist += 1
            endpoints.append((x, y))
        return endpoints

    # ------------------------------------------------------------------
    def _get_obs(self):
        rays = self._cast_rays()
        max_dist = self.grid_size - 1
        goal_dx = (self.goal_pos[0] - self.drone_pos[0]) / max_dist
        goal_dy = (self.goal_pos[1] - self.drone_pos[1]) / max_dist
        return np.array(rays + [goal_dx, goal_dy], dtype=np.float32)

    def _distance_to_goal(self):
        return np.linalg.norm(self.drone_pos - self.goal_pos)

    # ------------------------------------------------------------------
    def step(self, action):
        self.steps += 1
        prev_pos  = self.drone_pos.copy()
        prev_dist = self._distance_to_goal()

        if action == 0:
            self.drone_pos[1] -= 1
        elif action == 1:
            self.drone_pos[1] += 1
        elif action == 2:
            self.drone_pos[0] -= 1
        elif action == 3:
            self.drone_pos[0] += 1

        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size - 1)
        new_dist = self._distance_to_goal()

        reward     = 0.0
        terminated = False

        reward += (prev_dist - new_dist) * 5.0
        reward -= 0.05

        # obstacle collision — bounce back, keep going (no episode end)
        if tuple(self.drone_pos) in self.obstacles:
            reward -= 10
            self.drone_pos = prev_pos.copy()  # push back to safe cell

        if np.array_equal(self.drone_pos, self.goal_pos):
            reward += 100
            terminated = True

        if (
            self.drone_pos[0] == 0 or self.drone_pos[1] == 0 or
            self.drone_pos[0] == self.grid_size - 1 or
            self.drone_pos[1] == self.grid_size - 1
        ):
            reward -= 1

        if self.steps >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        gx, gy = self.goal_pos
        dx, dy = self.drone_pos
        grid[gy, gx] = "G"
        grid[dy, dx] = "D"
        for (x, y) in self.obstacles:
            grid[y, x] = "X"
        print()
        for row in grid:
            print(" ".join(row))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke-test DroneEnvLidar")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    env = DroneEnvLidar(n_obstacles=10)
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}  obs: {obs}")
    env.render()
    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        rays = env.get_lidar_endpoints()
        print(f"step={i+1}  action={action}  reward={reward:.2f}  done={done}")
        print(f"  lidar endpoints: {rays[:4]} ...")
        if done:
            print("Episode ended, resetting.")
            obs, _ = env.reset()
