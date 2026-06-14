import pygame
import time
import numpy as np
from env.drone_env import DroneEnv
from stable_baselines3 import PPO

# load trained model
model = PPO.load("models/drone_ppo")

env = DroneEnv()

pygame.init()

CELL_SIZE = 50
GRID_SIZE = env.grid_size

WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚁 RL Drone Navigation Simulator")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)

clock = pygame.time.Clock()


def draw_grid(env):
    screen.fill(WHITE)

    # draw grid lines
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(screen, BLACK, rect, 1)

    # draw obstacles
    for (x, y) in env.obstacles:
        pygame.draw.rect(
            screen,
            RED,
            (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    # draw goal
    gx, gy = env.goal_pos
    pygame.draw.rect(
        screen,
        GREEN,
        (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # draw drone
    dx, dy = env.drone_pos
    pygame.draw.circle(
        screen,
        BLUE,
        (
            dx * CELL_SIZE + CELL_SIZE // 2,
            dy * CELL_SIZE + CELL_SIZE // 2
        ),
        CELL_SIZE // 3
    )

    pygame.display.update()


def run():
    obs, _ = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)

        draw_grid(env)
        clock.tick(5)  # speed control

        if done:
            time.sleep(1)
            obs, _ = env.reset()
            done = False


if __name__ == "__main__":
    run()