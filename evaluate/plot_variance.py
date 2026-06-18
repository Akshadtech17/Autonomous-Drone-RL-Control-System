"""
Plot mean ± std reward curve across 5 seed runs.

Reads: logs/metrics_seed_{seed}.json  (written by train_ppo.py with --seed)
Output: evaluate/variance_plot.png

Usage:
    python -m evaluate.plot_variance [--seeds 42 123 456 789 1337] [--out PATH]
    python -m evaluate.plot_variance --help
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot reward variance across seeds")
parser.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 456, 789, 1337],
                    help="Seeds to include (default: 42 123 456 789 1337)")
parser.add_argument("--log-dir", type=str,
                    default=os.path.join(BASE_DIR, "logs"),
                    help="Directory containing metrics_seed_N.json files")
parser.add_argument("--out",     type=str,
                    default=os.path.join(BASE_DIR, "evaluate", "variance_plot.png"),
                    help="Output PNG path")
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
def load_rewards(seed: int, log_dir: str):
    """Load reward list for a given seed from its metrics JSON."""
    path = os.path.join(log_dir, f"metrics_seed_{seed}.json")
    if not os.path.exists(path):
        # fall back to generic metrics.json for seed 42
        if seed == 42:
            path = os.path.join(log_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("rewards", [])


def smooth(arr, window=10):
    if len(arr) < window:
        return np.array(arr)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ------------------------------------------------------------------
def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#111122")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    all_rewards = []
    max_len     = 0
    found_seeds = []

    for seed in args.seeds:
        rewards = load_rewards(seed, args.log_dir)
        if rewards is None:
            print(f"[SKIP] seed {seed}: no data found in {args.log_dir}")
            continue
        s = smooth(rewards)
        all_rewards.append(s)
        max_len = max(max_len, len(s))
        found_seeds.append(seed)

    if not all_rewards:
        print("No data found for any seed. Train with --seed first:")
        print("  python -m train.train_ppo --seed 42")
        print("  python -m train.train_ppo --seed 123")
        print("  ...")
        return

    # pad shorter arrays with their last value
    padded = []
    for arr in all_rewards:
        if len(arr) < max_len:
            pad = np.full(max_len - len(arr), arr[-1])
            arr = np.concatenate([arr, pad])
        padded.append(arr)

    matrix = np.stack(padded)           # shape (n_seeds, episodes)
    mean   = matrix.mean(axis=0)
    std    = matrix.std(axis=0)
    xs     = np.arange(max_len)

    ax.plot(xs, mean, color="#4499ff", linewidth=2,
            label=f"Mean reward  (seeds={found_seeds})")
    ax.fill_between(xs, mean - std, mean + std, alpha=0.25, color="#4499ff",
                    label="±1 std")

    # individual seed lines (faint)
    for i, (seed, arr) in enumerate(zip(found_seeds, padded)):
        ax.plot(xs, arr, alpha=0.25, linewidth=0.8,
                color=plt.cm.tab10(i / 10), label=f"seed {seed}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Multi-Seed Reward Variance")
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(facecolor="#141428", labelcolor="white", edgecolor="#333355",
              fontsize=8, loc="upper left")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Variance plot saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
