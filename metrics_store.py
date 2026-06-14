import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "logs", "metrics.json")

def init():
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "w") as f:
            json.dump({"rewards": [], "episodes": []}, f)

def write(data):
    with open(FILE_PATH, "w") as f:
        json.dump(data, f)

def read():
    try:
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    except:
        return {"rewards": [], "episodes": []}