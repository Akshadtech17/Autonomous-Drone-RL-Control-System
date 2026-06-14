import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class DroneEnv(gym.Env):
    """
    Grid-based Drone Navigation Environment (PPO-ready)
    Goal: Reach target while avoiding obstacles
    """

    def __init__(self):
        super(DroneEnv, self).__init__()

        # =========================
        # ENV CONFIG
        # =========================
        self.grid_size = 10
        self.max_steps = 200

        # Actions: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Observation: drone_x, drone_y, goal_x, goal_y
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        self.drone_pos = np.array([0, 0], dtype=np.int32)

        self.goal_pos = np.array(
            [self.grid_size - 1, self.grid_size - 1],
            dtype=np.int32
        )

        self.obstacles = self._generate_obstacles()

        return self._get_obs(), {}

    # =========================
    # OBSTACLES
    # =========================
    def _generate_obstacles(self):
        obstacles = set()

        for _ in range(15):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)

            if (x, y) not in [(0, 0), tuple(self.goal_pos)]:
                obstacles.add((x, y))

        return obstacles

    # =========================
    # OBSERVATION
    # =========================
    def _get_obs(self):
        return np.array(
            np.concatenate([self.drone_pos, self.goal_pos]),
            dtype=np.float32
        )

    # =========================
    # DISTANCE
    # =========================
    def _distance_to_goal(self):
        return np.linalg.norm(self.drone_pos - self.goal_pos)

    # =========================
    # STEP FUNCTION (CORE FIXED LOGIC)
    # =========================
    def step(self, action):

        self.steps += 1

        prev_dist = self._distance_to_goal()

        # -------------------------
        # MOVEMENT LOGIC
        # -------------------------
        if action == 0:   # up
            self.drone_pos[1] -= 1
        elif action == 1: # down
            self.drone_pos[1] += 1
        elif action == 2: # left
            self.drone_pos[0] -= 1
        elif action == 3: # right
            self.drone_pos[0] += 1

        # keep inside grid
        self.drone_pos = np.clip(
            self.drone_pos,
            0,
            self.grid_size - 1
        )

        new_dist = self._distance_to_goal()

        # =========================
        # 🔥 REWARD FUNCTION (FIXED)
        # =========================
        reward = 0.0
        terminated = False

        # 1. progress reward (MOST IMPORTANT SIGNAL)
        reward += (prev_dist - new_dist) * 5.0

        # 2. small survival penalty
        reward -= 0.05

        # 3. obstacle collision
        if tuple(self.drone_pos) in self.obstacles:
            reward -= 50
            terminated = True

        # 4. goal reached
        if np.array_equal(self.drone_pos, self.goal_pos):
            reward += 100
            terminated = True

        # 5. boundary penalty (prevents wall hugging)
        if (
            self.drone_pos[0] == 0 or
            self.drone_pos[1] == 0 or
            self.drone_pos[0] == self.grid_size - 1 or
            self.drone_pos[1] == self.grid_size - 1
        ):
            reward -= 1

        # 6. max steps termination
        if self.steps >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    # =========================
    # RENDER (DEBUG MODE)
    # =========================
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        gx, gy = self.goal_pos
        dx, dy = self.drone_pos

        grid[gy, gx] = "G"
        grid[dy, dx] = "D"

        for (x, y) in self.obstacles:
            grid[y, x] = "X"

        print("\n")
        for row in grid:
            print(" ".join(row))