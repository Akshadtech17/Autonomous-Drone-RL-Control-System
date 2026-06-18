"""
Visualise the trained policy as a 10x10 grid of directional arrows.

For each cell (x,y) with goal fixed at (9,9):
  - build the correct observation (LIDAR or exact-coord)
  - call model.predict(deterministic=True)
  - record chosen action + confidence

Output: evaluate/policy_heatmap.png

Usage:
    python -m evaluate.policy_heatmap
    python -m evaluate.policy_heatmap --lidar
    python -m evaluate.policy_heatmap --model models/checkpoints/drone_ppo_100000_steps.zip
    python -m evaluate.policy_heatmap --help
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch as th
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Policy heatmap visualiser")
parser.add_argument("--model",  type=str, default=None,
                    help="Path to model .zip (default: best available)")
parser.add_argument("--lidar",  action="store_true",
                    help="Use LIDAR observation builder")
parser.add_argument("--out",    type=str,
                    default=os.path.join(BASE_DIR, "evaluate", "policy_heatmap.png"),
                    help="Output PNG path")
parser.add_argument("--n-obstacles", type=int, default=0,
                    help="Obstacles to place for obs build in lidar mode (default 0 = none)")
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------------
if args.model:
    model_path = args.model
elif args.lidar:
    model_path = os.path.join(BASE_DIR, "models", "lidar", "drone_ppo_lidar_final.zip")
else:
    model_path = os.path.join(BASE_DIR, "models", "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", "drone_ppo.zip")

if not os.path.exists(model_path):
    print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
    sys.exit(1)

print(f"Loading model: {model_path}")
model = PPO.load(model_path)

# ------------------------------------------------------------------
# OBSERVATION BUILDERS
# ------------------------------------------------------------------
GRID     = 10
GOAL     = (9, 9)
ACTION_ARROWS = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}
ACTION_NAMES  = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}


def build_obs_coord(x, y):
    return np.array([x, y, GOAL[0], GOAL[1]], dtype=np.float32)


def build_obs_lidar(x, y, obstacles):
    from env.drone_env_lidar import DroneEnvLidar, DIRECTIONS
    max_d = GRID - 1
    rays = []
    for dx, dy in DIRECTIONS:
        dist = 0
        cx, cy = x, y
        while dist < max_d:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
                break
            if (nx, ny) in obstacles:
                dist += 1
                break
            cx, cy = nx, ny
            dist += 1
        rays.append(dist / max_d)
    goal_dx = (GOAL[0] - x) / max_d
    goal_dy = (GOAL[1] - y) / max_d
    return np.array(rays + [goal_dx, goal_dy], dtype=np.float32)


def get_probs(obs_np):
    obs_t = th.tensor(obs_np).float().unsqueeze(0).to(model.device)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_t)
        return dist.distribution.probs.squeeze().cpu().numpy()


# ------------------------------------------------------------------
# COLLECT POLICY
# ------------------------------------------------------------------
def collect(use_lidar: bool, obstacles: set):
    actions    = np.zeros((GRID, GRID), dtype=int)
    confidence = np.zeros((GRID, GRID), dtype=float)

    for y in range(GRID):
        for x in range(GRID):
            if use_lidar:
                obs = build_obs_lidar(x, y, obstacles)
            else:
                obs = build_obs_coord(x, y)

            probs  = get_probs(obs)
            action = int(np.argmax(probs))
            actions[y, x]    = action
            confidence[y, x] = float(probs[action])

    return actions, confidence


# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
def plot(actions, confidence, obstacles, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#111122")
    fig.patch.set_facecolor("#0d0d1a")

    # grid
    for i in range(GRID + 1):
        ax.axhline(i, color="#2a2a4a", linewidth=0.8)
        ax.axvline(i, color="#2a2a4a", linewidth=0.8)

    for y in range(GRID):
        for x in range(GRID):
            cx, cy = x + 0.5, y + 0.5

            # obstacle
            if (x, y) in obstacles:
                rect = plt.Rectangle((x, y), 1, 1, color="#c0392b", alpha=0.7)
                ax.add_patch(rect)
                ax.text(cx, cy, "✕", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
                continue

            # goal
            if x == GOAL[0] and y == GOAL[1]:
                circle = plt.Circle((cx, cy), 0.38, color="#27ae60", zorder=3)
                ax.add_patch(circle)
                ax.text(cx, cy, "G", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold", zorder=4)
                continue

            action = actions[y, x]
            conf   = confidence[y, x]

            # cell background shaded by confidence
            shade = int(conf * 80)
            color = f"#{shade:02x}{shade:02x}{min(shade + 40, 255):02x}"
            rect  = plt.Rectangle((x, y), 1, 1, color=color, alpha=0.5)
            ax.add_patch(rect)

            # arrow
            arrow_color = plt.cm.Blues(0.3 + conf * 0.7)
            dx, dy = ACTION_ARROWS[action]
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                arrowprops=dict(
                    arrowstyle="->", color=arrow_color,
                    lw=1.5 + conf * 1.5,
                ),
            )

            # confidence text
            ax.text(cx, cy - 0.33, f"{conf:.0%}",
                    ha="center", va="center",
                    color="white", fontsize=6, alpha=0.7)

    # axes
    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.set_xticklabels(range(GRID), color="white", fontsize=8)
    ax.set_yticklabels(range(GRID), color="white", fontsize=8)
    ax.set_title(
        f"Policy Heatmap — goal at {GOAL}  ({'LIDAR' if args.lidar else 'exact-coord'})",
        color="white", fontsize=12, pad=10,
    )
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")

    # legend
    handles = [
        mpatches.Patch(color="#27ae60", label="Goal"),
        mpatches.Patch(color="#c0392b", label="Obstacle"),
        mpatches.Patch(color="#3399ff", label="High confidence"),
        mpatches.Patch(color="#334466", label="Low confidence"),
    ]
    ax.legend(handles=handles, loc="upper left",
              facecolor="#141428", labelcolor="white",
              edgecolor="#333355", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Heatmap saved → {out_path}")
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    obstacles = set()
    if args.lidar and args.n_obstacles > 0:
        import random
        while len(obstacles) < args.n_obstacles:
            x = random.randint(0, GRID - 1)
            y = random.randint(0, GRID - 1)
            if (x, y) not in [(0, 0), GOAL]:
                obstacles.add((x, y))

    print(f"Grid={GRID}x{GRID}  goal={GOAL}  obstacles={len(obstacles)}")
    actions, confidence = collect(args.lidar, obstacles)
    plot(actions, confidence, obstacles, args.out)


if __name__ == "__main__":
    main()
