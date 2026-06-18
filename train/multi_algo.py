"""
Multi-algorithm benchmark: PPO vs SAC on DroneEnvAdvanced.

Trains each algorithm across multiple seeds, evaluates deterministically,
runs a Mann-Whitney U test for statistical significance, and produces a
violin + box comparison plot.

Usage:
    python -m train.multi_algo --timesteps 100000 --seeds 42 123 456
    python -m train.multi_algo --algos ppo sac --eval-episodes 50
    python train/run.py --multi-algo

Results:
    logs/multi_algo_results.json   — raw rewards, stats, p-value
    evaluate/multi_algo_plot.png   — violin/box comparison chart
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone.envs.advanced import DomainConfig, DroneEnvAdvanced, ObstacleConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_ALGOS: List[str] = ["ppo", "sac"]
DEFAULT_SEEDS: List[int] = [42, 123, 456]
DEFAULT_TIMESTEPS: int = 100_000
DEFAULT_EVAL_EPISODES: int = 50

# Fixed medium-difficulty eval config for fair comparison
BENCHMARK_CONFIG = DomainConfig(
    obstacles=ObstacleConfig(n_static=8, n_moving=3, speed_max=0.07),
    wind_strength=0.06,
    motor_fail_prob=0.0,
    goal_dist_min=10.0,
    goal_dist_max=14.0,
)

PPO_PARAMS: Dict = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)

SAC_PARAMS: Dict = dict(
    learning_rate=3e-4, buffer_size=100_000, learning_starts=1_000,
    batch_size=256, tau=0.005, gamma=0.99,
    train_freq=1, gradient_steps=1, ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)


# ---------------------------------------------------------------------------
# TRAIN + EVAL ONE RUN
# ---------------------------------------------------------------------------

def _train_and_eval(
    algo: str,
    seed: int,
    timesteps: int,
    eval_episodes: int,
    model_dir: Path,
) -> Tuple[float, float, float]:
    """
    Train one algorithm×seed combination and evaluate.
    Returns (mean_reward, std_reward, success_rate).
    """

    def make_env() -> Monitor:
        return Monitor(DroneEnvAdvanced(config=BENCHMARK_CONFIG))

    train_env = DummyVecEnv([make_env])
    eval_env_vec = DummyVecEnv([make_env])

    if algo == "ppo":
        model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, **PPO_PARAMS)
    else:
        model = SAC("MlpPolicy", train_env, seed=seed, verbose=0, **SAC_PARAMS)

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0

    save_path = model_dir / f"drone_{algo}_seed{seed}"
    model.save(str(save_path))

    rewards, _ = evaluate_policy(
        model, eval_env_vec,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    rewards = np.asarray(rewards, dtype=np.float32)

    # Success = reward above goal-bonus threshold (~400 after penalties)
    success_rate = float((rewards > 400.0).mean())

    train_env.close()
    eval_env_vec.close()

    print(
        f"  [{algo.upper()} seed={seed}] "
        f"mean={rewards.mean():.1f} ± {rewards.std():.1f}  "
        f"success={success_rate:.1%}  "
        f"time={elapsed:.0f}s"
    )
    return float(rewards.mean()), float(rewards.std()), success_rate


# ---------------------------------------------------------------------------
# STATISTICAL TEST
# ---------------------------------------------------------------------------

def _mann_whitney(
    rewards_a: List[float],
    rewards_b: List[float],
    label_a: str,
    label_b: str,
) -> Dict:
    """Two-sided Mann-Whitney U test for reward distributions (pure numpy)."""
    a, b = np.array(rewards_a, dtype=float), np.array(rewards_b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"test": "Mann-Whitney U", "error": "need ≥2 samples per group"}
    # U statistic: count pairs where a > b
    u_a = float(np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :]))
    u_b = float(na * nb - u_a)
    statistic = min(u_a, u_b)
    # Normal approximation (valid for na,nb >= 8)
    mu_u  = na * nb / 2.0
    sig_u = np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z     = (statistic - mu_u) / sig_u if sig_u > 0 else 0.0
    # Two-tailed p via complementary error function (no scipy needed)
    p_value = float(np.exp(-0.5 * z * z) * np.sqrt(2 / np.pi) * 0.5 + 0.5 -
                    0.5 * np.sign(z) * (1 - np.exp(-0.5 * z * z)))
    # Accurate two-tailed p via erfc approximation
    az = abs(z) / np.sqrt(2)
    t  = 1.0 / (1.0 + 0.3275911 * az)
    erfc_approx = (t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                   t * (-1.453152027 + t * 1.061405429))))) * np.exp(-az * az)
    p_value = float(min(1.0, 2.0 * erfc_approx))
    significant = p_value < 0.05
    winner = (label_a if np.mean(a) > np.mean(b) else label_b) if significant else "no significant difference"
    return {
        "test": "Mann-Whitney U",
        "statistic": float(statistic),
        "p_value": p_value,
        "significant_at_0_05": significant,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def _make_plot(
    results: Dict[str, List[float]],
    stats: Dict[str, Dict],
    p_value: Optional[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[multi_algo] matplotlib not found — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d0d1a")
    colors = {"ppo": "#4499ff", "sac": "#ff9944", "ppo+transformer": "#44ff99"}
    algos = list(results.keys())

    # ── Violin plot ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#111122")
    data_for_violin = [results[a] for a in algos]
    parts = ax.violinplot(data_for_violin, positions=range(len(algos)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        algo = algos[i]
        pc.set_facecolor(colors.get(algo, "#aaaaaa"))
        pc.set_alpha(0.7)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[part_name].set_color("white")
        parts[part_name].set_linewidth(1.2)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos], color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_title("Reward Distribution (all seeds)", color="white", fontsize=11)
    ax.set_ylabel("Episode Reward", color="white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Significance annotation
    if p_value is not None and len(algos) == 2:
        y_max = max(max(v) for v in results.values())
        sig_text = f"p={p_value:.3f} {'*' if p_value < 0.05 else 'ns'}"
        ax.annotate(
            "", xy=(1, y_max * 0.98), xytext=(0, y_max * 0.98),
            arrowprops=dict(arrowstyle="<->", color="white", lw=1),
        )
        ax.text(0.5, y_max * 1.01, sig_text, ha="center", color="white", fontsize=9)

    # ── Bar chart: mean ± std ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#111122")
    means = [stats[a]["mean_reward"] for a in algos]
    stds = [stats[a]["std_reward"] for a in algos]
    bars = ax2.bar(
        range(len(algos)), means,
        yerr=stds, capsize=6,
        color=[colors.get(a, "#aaaaaa") for a in algos],
        alpha=0.85, edgecolor="white", linewidth=0.8,
    )
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels([a.upper() for a in algos], color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.set_title("Mean Reward ± Std (across seeds)", color="white", fontsize=11)
    ax2.set_ylabel("Mean Episode Reward", color="white")
    ax2.yaxis.label.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333355")

    # Annotate bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(
            i, mean + std + abs(mean) * 0.02,
            f"{mean:.0f}",
            ha="center", color="white", fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "PPO vs SAC — DroneEnvAdvanced Benchmark", color="white", fontsize=13, y=1.02
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"[multi_algo] Plot saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_comparison(
    algos: List[str] = DEFAULT_ALGOS,
    seeds: List[int] = DEFAULT_SEEDS,
    timesteps: int = DEFAULT_TIMESTEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    root_dir: Optional[Path] = None,
) -> Dict:
    """
    Train all (algo × seed) combinations and compare statistically.
    Returns the full results dict (also saved to JSON).
    """
    root = root_dir or Path(__file__).parent.parent
    log_dir = root / "logs"
    eval_dir = root / "evaluate"
    model_dir = root / "models" / "benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[BENCHMARK] algos={algos}  seeds={seeds}  "
        f"timesteps={timesteps}  eval_eps={eval_episodes}"
    )

    all_rewards: Dict[str, List[float]] = {a: [] for a in algos}
    per_seed: Dict[str, List[Dict]] = {a: [] for a in algos}

    for algo in algos:
        print(f"\n--- {algo.upper()} ---")
        for seed in seeds:
            mean_r, std_r, sr = _train_and_eval(
                algo, seed, timesteps, eval_episodes, model_dir
            )
            all_rewards[algo].append(mean_r)
            per_seed[algo].append({
                "seed": seed,
                "mean_reward": round(mean_r, 2),
                "std_reward": round(std_r, 2),
                "success_rate": round(sr, 4),
            })

    # Aggregate stats per algo
    stats: Dict[str, Dict] = {}
    for algo in algos:
        r = np.array(all_rewards[algo])
        stats[algo] = {
            "mean_reward": float(r.mean()),
            "std_reward": float(r.std()),
            "min_reward": float(r.min()),
            "max_reward": float(r.max()),
            "mean_success_rate": float(
                np.mean([s["success_rate"] for s in per_seed[algo]])
            ),
            "n_seeds": len(seeds),
        }

    print("\n=== RESULTS ===")
    for algo in algos:
        s = stats[algo]
        print(
            f"  {algo.upper():8s} | "
            f"mean={s['mean_reward']:8.2f} ± {s['std_reward']:6.2f} | "
            f"success={s['mean_success_rate']:.1%}"
        )

    # Statistical significance test (pairwise if 2 algos)
    sig_test: Optional[Dict] = None
    p_value: Optional[float] = None
    if len(algos) == 2:
        sig_test = _mann_whitney(
            all_rewards[algos[0]], all_rewards[algos[1]],
            algos[0], algos[1],
        )
        p_value = sig_test.get("p_value")
        print(f"\n  Statistical test: {sig_test}")

    # Save JSON
    output = {
        "algos": algos,
        "seeds": seeds,
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        "stats": stats,
        "per_seed": per_seed,
        "significance_test": sig_test,
    }
    out_json = log_dir / "multi_algo_results.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"\n[BENCHMARK] Results saved -> {out_json}")

    # Plot
    _make_plot(
        all_rewards, stats, p_value,
        out_path=eval_dir / "multi_algo_plot.png",
    )

    return output


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO vs SAC benchmark on DroneEnvAdvanced")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS, choices=["ppo", "sac"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    args, _ = parser.parse_known_args()
    run_comparison(
        algos=args.algos,
        seeds=args.seeds,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
    )
