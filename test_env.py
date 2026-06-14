from env.drone_env import DroneEnv
import random

env = DroneEnv()

obs, _ = env.reset()

for step in range(20):

    action = random.randint(0, 3)

    obs, reward, done, _, _ = env.step(action)

    env.render()

    print(
        f"Action={action}, "
        f"Reward={reward}"
    )

    if done:
        print("Goal Reached!")
        break