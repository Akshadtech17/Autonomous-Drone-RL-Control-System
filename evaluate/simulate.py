"""
Pygame visual simulator.

Usage:
    python -m evaluate.simulate              # exact-coord model (models/drone_ppo.zip)
    python -m evaluate.simulate --lidar      # LIDAR model (models/lidar/drone_ppo_lidar_final.zip)
    python -m evaluate.simulate --model PATH # any checkpoint
    python -m evaluate.simulate --help
"""

import argparse
import os
import time
import numpy as np

import pygame

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pygame drone simulator")
parser.add_argument("--lidar",  action="store_true",
                    help="Use LIDAR env + model and draw ray lines")
parser.add_argument("--model",  type=str, default=None,
                    help="Path to .zip model (overrides --lidar default path)")
parser.add_argument("--fps",    type=int, default=5, help="Playback speed (default 5)")
parser.add_argument("--episodes", type=int, default=0,
                    help="Number of episodes to run (0 = infinite)")
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# ENV + MODEL SELECTION
# ------------------------------------------------------------------
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if args.lidar:
    from env.drone_env_lidar import DroneEnvLidar
    env = DroneEnvLidar()
    default_model = os.path.join(BASE_DIR, "models", "lidar", "drone_ppo_lidar_final.zip")
else:
    from env.drone_env import DroneEnv
    env = DroneEnv()
    default_model = os.path.join(BASE_DIR, "models", "drone_ppo.zip")

model_path = args.model if args.model else default_model

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}\n"
        f"Train first: python -m train.train_ppo  (or --lidar: python -m train.train_lidar)"
    )

model = PPO.load(model_path)
print(f"Loaded model: {model_path}")

# ------------------------------------------------------------------
# PYGAME SETUP
# ------------------------------------------------------------------
pygame.init()

CELL   = 55
GRID   = env.grid_size
W = H  = GRID * CELL

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("RL Drone Simulator" + (" (LIDAR)" if args.lidar else ""))

WHITE  = (255, 255, 255)
BLACK  = (30,  30,  30)
BG     = (20,  20,  35)
RED    = (200, 60,  60)
GREEN  = (60,  200, 80)
BLUE   = (60,  120, 220)
YELLOW = (255, 220, 0)
GRID_C = (50,  50,  70)
LIDAR_C= (255, 230, 50, 60)   # semi-transparent yellow

clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 13)


# ------------------------------------------------------------------
# DRAW
# ------------------------------------------------------------------
def draw(env, step, reward, done):
    screen.fill(BG)

    # grid lines
    for i in range(GRID + 1):
        pygame.draw.line(screen, GRID_C, (i * CELL, 0), (i * CELL, H))
        pygame.draw.line(screen, GRID_C, (0, i * CELL), (W, i * CELL))

    # obstacles
    for (x, y) in env.obstacles:
        pygame.draw.rect(screen, RED,
                         (x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4))

    # LIDAR rays (lidar mode only)
    if args.lidar and hasattr(env, "get_lidar_endpoints"):
        endpoints = env.get_lidar_endpoints()
        dx, dy = int(env.drone_pos[0]), int(env.drone_pos[1])
        cx, cy = dx * CELL + CELL // 2, dy * CELL + CELL // 2
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for (ex, ey) in endpoints:
            ex_px = ex * CELL + CELL // 2
            ey_px = ey * CELL + CELL // 2
            pygame.draw.line(surf, (255, 230, 50, 70), (cx, cy), (ex_px, ey_px), 1)
            pygame.draw.circle(surf, (255, 230, 50, 130), (ex_px, ey_px), 3)
        screen.blit(surf, (0, 0))

    # goal
    gx, gy = env.goal_pos
    pygame.draw.rect(screen, GREEN,
                     (gx * CELL + 2, gy * CELL + 2, CELL - 4, CELL - 4))
    g_lbl = font.render("G", True, BLACK)
    screen.blit(g_lbl, (gx * CELL + CELL // 2 - 5, gy * CELL + CELL // 2 - 7))

    # drone
    drx, dry = int(env.drone_pos[0]), int(env.drone_pos[1])
    pygame.draw.circle(screen, BLUE,
                       (drx * CELL + CELL // 2, dry * CELL + CELL // 2),
                       CELL // 3)

    # HUD
    hud = font.render(
        f"step={step:3d}  reward={reward:6.2f}  {'DONE' if done else '    '}",
        True, WHITE
    )
    screen.blit(hud, (6, 4))

    pygame.display.flip()


# ------------------------------------------------------------------
# RUN LOOP
# ------------------------------------------------------------------
def run():
    obs, _ = env.reset()
    done      = False
    step      = 0
    ep_reward = 0.0
    ep_count  = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            step      += 1
            ep_reward += reward

        draw(env, step, ep_reward, done)
        clock.tick(args.fps)

        if done:
            time.sleep(0.8)
            ep_count += 1
            if args.episodes and ep_count >= args.episodes:
                pygame.quit()
                return
            obs, _ = env.reset()
            done      = False
            step      = 0
            ep_reward = 0.0


if __name__ == "__main__":
    run()
