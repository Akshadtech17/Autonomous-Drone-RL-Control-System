from stable_baselines3 import PPO
import numpy as np

# Load trained model once (IMPORTANT)
model = PPO.load("models/drone_ppo")

def get_action(observation):
    """
    Takes environment state → returns RL action
    """
    action, _ = model.predict(np.array(observation))
    return int(action)