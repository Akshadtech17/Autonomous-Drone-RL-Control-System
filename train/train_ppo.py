import os
import json
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv

from env.drone_env import DroneEnv

# =========================================================
# 🔒 REPRODUCIBILITY
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# 📁 PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

METRICS_FILE = os.path.join(LOG_DIR, "metrics.json")
STATE_FILE = os.path.join(LOG_DIR, "live_state.json")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================================
# 📊 INIT FILES
# =========================================================
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w") as f:
        json.dump({"episodes": [], "rewards": [], "lengths": []}, f)

if not os.path.exists(STATE_FILE):
    with open(STATE_FILE, "w") as f:
        json.dump({
            "training": False,
            "episode": 0,
            "reward": 0.0,
            "episode_length": 0,
            "last_action": None
        }, f)

# =========================================================
# 📁 HELPERS
# =========================================================
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

# =========================================================
# 🧠 ENVIRONMENT
# =========================================================
def make_env():
    env = DroneEnv()
    return Monitor(env)

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# =========================================================
# 📊 EVAL CALLBACK
# =========================================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# =========================================================
# 💾 CHECKPOINT CALLBACK
# =========================================================
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="drone_ppo"
)

# =========================================================
# 📡 LIVE METRICS CALLBACK (🔥 MAIN ENGINE)
# =========================================================
class LiveMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")

        if rewards is None or dones is None:
            return True

        reward = float(rewards[0])
        done = bool(dones[0])

        self.episode_reward += reward
        self.episode_length += 1

        # track last action
        if actions is not None:
            state = load_json(STATE_FILE)
            state["last_action"] = int(actions[0])
            save_json(STATE_FILE, state)

        if done:
            self.episode_count += 1

            # =========================
            # PRINT LOG
            # =========================
            print(
                f"📊 Episode {self.episode_count} | "
                f"Reward: {self.episode_reward:.2f} | "
                f"Length: {self.episode_length}"
            )

            # =========================
            # UPDATE METRICS HISTORY
            # =========================
            metrics = load_json(METRICS_FILE)

            metrics["episodes"].append(self.episode_count)
            metrics["rewards"].append(self.episode_reward)
            metrics["lengths"].append(self.episode_length)

            metrics["episodes"] = metrics["episodes"][-200:]
            metrics["rewards"] = metrics["rewards"][-200:]
            metrics["lengths"] = metrics["lengths"][-200:]

            save_json(METRICS_FILE, metrics)

            # =========================
            # UPDATE LIVE STATE (FLASK)
            # =========================
            state = load_json(STATE_FILE)

            state["episode"] = self.episode_count
            state["reward"] = float(self.episode_reward)
            state["episode_length"] = self.episode_length

            save_json(STATE_FILE, state)

            # reset episode
            self.episode_reward = 0.0
            self.episode_length = 0

        return True

metrics_callback = LiveMetricsCallback()

# =========================================================
# 🧠 PPO MODEL
# =========================================================
model = PPO(
    policy="MlpPolicy",
    env=env,

    tensorboard_log=LOG_DIR,
    verbose=1,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,

    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,

    policy_kwargs=dict(net_arch=[256, 256]),

    seed=SEED
)

# =========================================================
# 🚀 TRAINING
# =========================================================
TOTAL_TIMESTEPS = 200000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[
        eval_callback,
        checkpoint_callback,
        metrics_callback
    ],
    tb_log_name="drone_rl_production"
)

# =========================================================
# 💾 SAVE FINAL MODEL
# =========================================================
FINAL_PATH = os.path.join(MODEL_DIR, "drone_ppo_final")
model.save(FINAL_PATH)

print("\n✅ TRAINING COMPLETE")
print(f"📦 Final Model: {FINAL_PATH}.zip")
print(f"🏆 Best Model: {BEST_MODEL_DIR}")
print(f"📁 Checkpoints: {CHECKPOINT_DIR}")
print(f"📊 Metrics: {METRICS_FILE}")
print(f"📡 Live State: {STATE_FILE}")