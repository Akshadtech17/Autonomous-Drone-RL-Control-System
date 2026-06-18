"""
Rigorous benchmark across all checkpoints (coord and lidar).

Usage:
    python -m evaluate.benchmark [--episodes 100] [--models-dir PATH]
    python -m evaluate.benchmark --help

Outputs:
    evaluate/benchmark_results.json
    evaluate/learning_curve.png
"""

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark all checkpoints")
parser.add_argument("--episodes",   type=int, default=100,
                    help="Episodes per checkpoint (default 100)")
parser.add_argument("--models-dir", type=str,
                    default=os.path.join(BASE_DIR, "models"),
                    help="Root models directory")
parser.add_argument("--out",        type=str,
                    default=os.path.join(BASE_DIR, "evaluate", "benchmark_results.json"),
                    help="Output JSON path")
parser.add_argument("--fig",        type=str,
                    default=os.path.join(BASE_DIR, "evaluate", "learning_curve.png"),
                    help="Output figure path")
args, _ = parser.parse_known_args()

N_EPISODES = args.episodes
MODELS_DIR = args.models_dir
OUT_JSON   = args.out
OUT_FIG    = args.fig

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def find_checkpoints(models_dir):
    """
    Returns list of (step, path, model_type) sorted by step ascending.
    model_type: 'coord' or 'lidar'
    """
    checkpoints = []
    for root, _, files in os.walk(models_dir):
        for f in files:
            if not f.endswith(".zip"):
                continue
            path = os.path.join(root, f)
            is_lidar = "lidar" in path.replace("\\", "/").lower()
            # extract step number from filename if possible
            step = _parse_step(f)
            checkpoints.append((step, path, "lidar" if is_lidar else "coord"))
    return sorted(checkpoints, key=lambda x: (x[2], x[0]))


def _parse_step(filename: str) -> int:
    """Try to extract training-step number from checkpoint filename."""
    import re
    m = re.search(r"_(\d+)_steps", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", filename)
    if m:
        return int(m.group(1))
    return 0


def _make_env(is_lidar: bool, seed: int):
    if is_lidar:
        from env.drone_env_lidar import DroneEnvLidar
        env = DroneEnvLidar()
    else:
        from env.drone_env import DroneEnv
        env = DroneEnv()
    return env


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(n) if n > 1 else 0.0


def run_episodes(model, is_lidar: bool, n: int):
    """Run n episodes, return (success_rate, mean_reward, mean_len, std_*)."""
    rewards, lengths, successes = [], [], []

    for ep in range(n):
        env = _make_env(is_lidar, seed=ep)
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        ep_len    = 0
        done      = False
        success   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            ep_len    += 1
            if done and reward > 50:  # reached goal reward = +100 -(small penalties)
                success = True

        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(int(success))
        env.close()

    r  = np.array(rewards)
    l  = np.array(lengths)
    s  = np.array(successes, dtype=float)
    return {
        "success_rate":      float(s.mean()),
        "success_rate_ci95": float(ci95(s.std(), n)),
        "mean_reward":       float(r.mean()),
        "mean_reward_std":   float(r.std()),
        "mean_reward_ci95":  float(ci95(r.std(), n)),
        "mean_length":       float(l.mean()),
        "mean_length_std":   float(l.std()),
        "mean_length_ci95":  float(ci95(l.std(), n)),
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    checkpoints = find_checkpoints(MODELS_DIR)
    if not checkpoints:
        print("No .zip model files found under", MODELS_DIR)
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints. Running {N_EPISODES} episodes each.\n")

    results       = []
    coord_pts     = []   # (step, success_rate, std) for plot
    lidar_pts     = []

    # header
    header = f"{'Model':<55} {'Type':<7} {'Step':>8} {'SuccessRate':>12} {'CI95':>8} {'MeanRew':>10} {'MeanLen':>9}"
    print(header)
    print("─" * len(header))

    for step, path, mtype in checkpoints:
        rel = os.path.relpath(path, MODELS_DIR)
        try:
            model    = PPO.load(path)
            is_lidar = (mtype == "lidar")
            stats    = run_episodes(model, is_lidar, N_EPISODES)
        except Exception as e:
            print(f"  SKIP {rel}: {e}")
            continue

        entry = {"path": rel, "step": step, "type": mtype, **stats}
        results.append(entry)

        sr  = stats["success_rate"]
        ci  = stats["success_rate_ci95"]
        mr  = stats["mean_reward"]
        ml  = stats["mean_length"]

        print(
            f"{rel:<55} {mtype:<7} {step:>8} "
            f"{sr:>11.1%} {ci:>8.3f} {mr:>10.2f} {ml:>9.1f}"
        )

        pt = (step, sr, stats["success_rate_ci95"])
        if is_lidar:
            lidar_pts.append(pt)
        else:
            coord_pts.append(pt)

    # ------------------------------------------------------------------
    # SAVE JSON
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    # ------------------------------------------------------------------
    # LEARNING CURVE PLOT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#111122")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    def _plot_series(pts, color, label):
        if not pts:
            return
        pts.sort(key=lambda x: x[0])
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=5)
        ax.fill_between(xs,
                        [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)],
                        alpha=0.2, color=color)

    _plot_series(coord_pts, "#4499ff", "Exact-coord model")
    _plot_series(lidar_pts, "#ff9944", "LIDAR model")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate")
    ax.set_title("Learning Curve — Success Rate vs Training Steps")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(facecolor="#141428", labelcolor="white", edgecolor="#333355")

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, facecolor=fig.get_facecolor())
    print(f"Learning curve saved → {OUT_FIG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
