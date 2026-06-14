from flask import Flask, jsonify, render_template_string, request
import subprocess
import threading
import os
import json
import time

app = Flask(__name__)

# =========================================================
# 📁 FOLDERS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 📡 SHARED LIVE STATE FILE (🔥 REAL BRIDGE)
# =========================================================
STATE_FILE = os.path.join(LOG_DIR, "live_state.json")

def init_state():
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "training": False,
                "simulation": False,
                "episode": 0,
                "reward": 0.0,
                "episode_length": 0,
                "last_action": None
            }, f)

def read_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def write_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

init_state()

# =========================================================
# 🌐 UI
# =========================================================
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚁 RL Control Center</title>

    <style>
        body {
            font-family: Arial;
            background: #0f0f1a;
            color: white;
            text-align: center;
        }

        h1 { margin-top: 25px; }

        button {
            padding: 15px 30px;
            margin: 10px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }

        .train { background: #28a745; color: white; }
        .sim { background: #007bff; color: white; }

        .box {
            margin-top: 20px;
            font-size: 18px;
            color: #ccc;
        }
    </style>

    <script>
        async function startTraining() {
            fetch("/train");
        }

        async function startSim() {
            fetch("/simulate");
        }

        async function refresh() {
            const res = await fetch("/state");
            const data = await res.json();

            document.getElementById("status").innerText =
                "Episode: " + data.episode +
                " | Reward: " + data.reward +
                " | Length: " + data.episode_length +
                " | Action: " + data.last_action +
                " | Training: " + data.training;
        }

        setInterval(refresh, 1000);
    </script>
</head>

<body>

<h1>🚁 Autonomous Drone RL Control Center</h1>

<button class="train" onclick="startTraining()">▶ Start Training</button>
<button class="sim" onclick="startSim()">🎮 Run Simulation</button>

<div class="box">
    <p id="status">Loading...</p>
</div>

</body>
</html>
"""

# =========================================================
# ⚙️ PROCESS RUNNER
# =========================================================
def run_command(cmd):
    subprocess.Popen(cmd, shell=True)

# =========================================================
# 🏠 HOME
# =========================================================
@app.route("/")
def home():
    return render_template_string(HOME_PAGE)

# =========================================================
# 🧠 TRAIN
# =========================================================
@app.route("/train")
def train():
    state = read_state()
    state["training"] = True
    write_state(state)

    def job():
        try:
            run_command("python -m train.train_ppo")
        finally:
            state = read_state()
            state["training"] = False
            write_state(state)

    threading.Thread(target=job).start()
    return jsonify({"status": "training started"})

# =========================================================
# 🎮 SIM
# =========================================================
@app.route("/simulate")
def simulate():
    state = read_state()
    state["simulation"] = True
    write_state(state)

    def job():
        try:
            run_command("python -m evaluate.simulate")
        finally:
            state = read_state()
            state["simulation"] = False
            write_state(state)

    threading.Thread(target=job).start()
    return jsonify({"status": "simulation started"})

# =========================================================
# 📡 LIVE STATE API
# =========================================================
@app.route("/state")
def state():
    return jsonify(read_state())

# =========================================================
# 🤖 PREDICT API
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    state_vec = data.get("state", [])

    action = sum(state_vec) % 4

    s = read_state()
    s["last_action"] = action
    write_state(s)

    return jsonify({"action": action})

# =========================================================
# ❤️ HEALTH
# =========================================================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)