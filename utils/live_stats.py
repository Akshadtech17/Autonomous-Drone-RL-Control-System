import json
import os

STATS_FILE = "logs/live_stats.json"

def log_stats(reward, length, episode):
    data = {
        "reward": float(reward),
        "length": int(length),
        "episode": int(episode)
    }

    os.makedirs("logs", exist_ok=True)

    with open(STATS_FILE, "w") as f:
        json.dump(data, f)