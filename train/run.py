"""
train.run — Master CLI entrypoint for the Autonomous Drone RL project.

Single command to orchestrate all training, evaluation, and analysis modes.

Commands:
  --train [--algo ppo|sac] [--timesteps N] [--curriculum] [--transformer]
        Train on DroneEnvAdvanced (new environment with LIDAR + wind)

  --train-basic [--timesteps N]
        Train original PPO on the 10x10 grid env (coord observations)

  --train-lidar [--timesteps N]
        Train original PPO on DroneEnvLidar (8-beam, 10x10 grid)

  --hpo [--algo ppo|sac] [--trials N] [--timesteps N]
        Hyperparameter optimisation with Optuna

  --multi-algo [--algos ppo sac] [--seeds 42 123 456] [--timesteps N]
        PPO vs SAC statistical comparison + Mann-Whitney U test

  --plan "mission text" [--algo ppo|sac]
        Parse NL mission via Groq -> MissionSpec -> launch training

  --xai [--model-path path/to/model.zip]
        Run XAI analysis on a trained model (feature importance, entropy)

  --eval [--model-path path] [--episodes N]
        Evaluate a saved model deterministically

Examples:
    python train/run.py --train --algo ppo --timesteps 200000 --curriculum
    python train/run.py --train --algo sac --timesteps 150000 --transformer
    python train/run.py --hpo --algo ppo --trials 20
    python train/run.py --multi-algo --timesteps 100000 --seeds 42 123 456
    python train/run.py --plan "Storm run with 12 obstacles and 20% motor failure"
    python train/run.py --xai --model-path models/advanced/ppo/drone_ppo_advanced_final.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).parent.parent

# Ensure root is on sys.path so imports work whether run as
# `python train/run.py` or `python -m train.run`
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python train/run.py",
        description="Autonomous Drone RL - master training/evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true",
                      help="Train on DroneEnvAdvanced (new env)")
    mode.add_argument("--train-basic", action="store_true",
                      help="Train on original 10x10 grid env")
    mode.add_argument("--train-lidar", action="store_true",
                      help="Train on original 8-beam LIDAR env")
    mode.add_argument("--hpo", action="store_true",
                      help="Hyperparameter optimisation with Optuna")
    mode.add_argument("--multi-algo", action="store_true",
                      help="Multi-algorithm benchmark (PPO vs SAC)")
    mode.add_argument("--plan", type=str, metavar="MISSION",
                      help="NL mission -> training dispatch via Groq")
    mode.add_argument("--xai", action="store_true",
                      help="XAI analysis on a trained model")
    mode.add_argument("--eval", action="store_true",
                      help="Evaluate a saved model")

    # Shared options
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                   help="Algorithm (default: ppo)")
    p.add_argument("--timesteps", type=int, default=200_000,
                   help="Training timesteps (default: 200000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    # Advanced training options
    p.add_argument("--curriculum", action="store_true",
                   help="Enable 10-stage curriculum (advanced env)")
    p.add_argument("--transformer", action="store_true",
                   help="Use LidarTransformerExtractor policy")
    p.add_argument("--mlflow", action="store_true",
                   help="Log to MLflow experiment tracker")

    # Eval options
    p.add_argument("--model-path", type=str, default=None,
                   help="Path to saved model .zip for --eval / --xai")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of evaluation episodes (default: 50)")

    # HPO options
    p.add_argument("--trials", type=int, default=20,
                   help="Optuna trials (default: 20)")

    # Multi-algo options
    p.add_argument("--algos", nargs="+", default=["ppo", "sac"],
                   choices=["ppo", "sac"],
                   help="Algorithms for benchmark (default: ppo sac)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Seeds for multi-algo benchmark (default: 42 123 456)")

    # LIDAR training
    p.add_argument("--obstacles", type=int, default=3,
                   help="Number of obstacles for LIDAR env (default: 3)")

    return p


# ---------------------------------------------------------------------------
# MODE HANDLERS
# ---------------------------------------------------------------------------

def _run_train(args: argparse.Namespace) -> None:
    from train.train_advanced import main as train_advanced

    policy_kwargs = None
    if args.transformer:
        from drone.policies.transformer import make_transformer_kwargs
        policy_kwargs = make_transformer_kwargs()
        print(f"[run] Using LidarTransformerExtractor policy")

    saved = train_advanced(
        algo=args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        use_curriculum=args.curriculum,
        use_mlflow=args.mlflow,
        root_dir=ROOT_DIR,
        policy_kwargs_override=policy_kwargs,
    )
    print(f"[run] Model saved -> {saved}")


def _run_train_basic(args: argparse.Namespace) -> None:
    from train.train_ppo import main as train_ppo
    saved = train_ppo(
        timesteps=args.timesteps,
        seed=args.seed,
        use_curriculum=False,
        use_mlflow=args.mlflow,
        root_dir=ROOT_DIR,
    )
    print(f"[run] Model saved -> {saved}")


def _run_train_lidar(args: argparse.Namespace) -> None:
    from train.train_lidar import main as train_lidar
    saved = train_lidar(
        timesteps=args.timesteps,
        seed=args.seed,
        n_obstacles=args.obstacles,
        root_dir=ROOT_DIR,
    )
    print(f"[run] Model saved -> {saved}")


def _run_hpo(args: argparse.Namespace) -> None:
    from train.hpo import run_hpo
    result = run_hpo(
        algo=args.algo,
        n_trials=args.trials,
        timesteps_per_trial=min(args.timesteps, 50_000),
        eval_episodes=30,
        seed=args.seed,
        root_dir=ROOT_DIR,
    )
    print(f"\n[run] HPO complete. Best reward: {result['best_reward']:.2f}")
    print(f"      Best params: {result['best_params']}")


def _run_multi_algo(args: argparse.Namespace) -> None:
    from train.multi_algo import run_comparison
    result = run_comparison(
        algos=args.algos,
        seeds=args.seeds,
        timesteps=args.timesteps,
        eval_episodes=args.episodes,
        root_dir=ROOT_DIR,
    )
    print(f"\n[run] Benchmark complete.")
    for algo, stats in result["stats"].items():
        print(
            f"  {algo.upper():6s}: "
            f"mean={stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}  "
            f"success={stats['mean_success_rate']:.1%}"
        )
    if result.get("significance_test"):
        sig = result["significance_test"]
        print(f"  Mann-Whitney p={sig.get('p_value', '?'):.4f}  "
              f"winner={sig.get('winner', '?')}")


def _run_plan(args: argparse.Namespace) -> None:
    from drone.llm.planner import MissionPlanner, MissionSpec
    from train.train_advanced import main as train_advanced

    planner = MissionPlanner()
    print(repr(planner))
    spec = planner.plan(args.plan)
    print(repr(spec))
    print(f"Difficulty: {MissionSpec.difficulty_label(spec)}")
    print(f"Reasoning: {spec.reasoning}")

    domain_cfg = spec.to_domain_config()
    algo = spec.algo
    print(f"\n[run] Launching {algo.upper()} training with planned config...")
    print(f"      Config: {domain_cfg}")

    saved = train_advanced(
        algo=algo,
        timesteps=args.timesteps,
        seed=args.seed,
        use_curriculum=False,
        use_mlflow=args.mlflow,
        root_dir=ROOT_DIR,
        domain_config_override=domain_cfg,
    )
    print(f"[run] Model saved -> {saved}")


def _run_xai(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from drone.envs.advanced import DroneEnvAdvanced, DomainConfig, ObstacleConfig
    from drone.xai.explainer import XAIExplainer

    # Find model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Auto-find best advanced model
        candidates = list((ROOT_DIR / "models" / "advanced").rglob("*.zip"))
        if not candidates:
            print("[run] No model found. Run --train first.")
            sys.exit(1)
        model_path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        print(f"[run] Auto-selected model: {model_path}")

    # Load
    algo = args.algo
    AlgoClass = SAC if algo == "sac" else PPO
    model = AlgoClass.load(str(model_path))
    print(f"[run] Loaded {AlgoClass.__name__} from {model_path}")

    xai = XAIExplainer(model)
    print(repr(xai))

    print("\n--- Feature Importance (top 15) ---")
    imp = xai.feature_importance(n_samples=200, top_k=15)
    max_len = max(len(k) for k in imp)
    for name, score in imp.items():
        bar = "#" * int(score * 30)
        print(f"  {name:{max_len}s} {score:.3f} {bar}")

    print("\n--- Action Entropy (200 samples) ---")
    stats = xai.action_entropy_stats(n_samples=200)
    print(
        f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}"
    )

    print("\n--- What-If: zero out all LIDAR beams (fully blind) ---")
    env = DroneEnvAdvanced(
        config=DomainConfig(
            obstacles=ObstacleConfig(n_static=0, n_moving=0),
            wind_strength=0.0, motor_fail_prob=0.0,
        )
    )
    obs, _ = env.reset()
    modifications = {i: 1.0 for i in range(64)}  # all beams to max range (no obstacles)
    result = xai.whatif(obs, modifications)
    print(f"  Baseline action: {[round(a, 3) for a in result['baseline_action']]}")
    print(f"  Modified action: {[round(a, 3) for a in result['modified_action']]}")
    print(f"  Delta magnitude: {result['delta_magnitude']:.4f}")
    env.close()

    print("\n[run] XAI analysis complete.")


def _run_eval(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from drone.envs.advanced import DroneEnvAdvanced, DomainConfig, ObstacleConfig

    if not args.model_path:
        candidates = list((ROOT_DIR / "models" / "advanced").rglob("*.zip"))
        if not candidates:
            print("[run] No model found. Pass --model-path.")
            sys.exit(1)
        model_path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        print(f"[run] Auto-selected model: {model_path}")
    else:
        model_path = Path(args.model_path)

    AlgoClass = SAC if args.algo == "sac" else PPO
    model = AlgoClass.load(str(model_path))
    print(f"[run] Loaded {AlgoClass.__name__} from {model_path}")

    config = DomainConfig(
        obstacles=ObstacleConfig(n_static=8, n_moving=3),
        wind_strength=0.06, motor_fail_prob=0.0,
    )
    env = DummyVecEnv([lambda: Monitor(DroneEnvAdvanced(config=config))])
    mean_r, std_r = evaluate_policy(
        model, env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        warn=False,
    )
    print(
        f"\n[run] Evaluation ({args.episodes} episodes): "
        f"mean={mean_r:.2f} ± {std_r:.2f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    t0 = time.time()

    try:
        if args.train:
            _run_train(args)
        elif args.train_basic:
            _run_train_basic(args)
        elif args.train_lidar:
            _run_train_lidar(args)
        elif args.hpo:
            _run_hpo(args)
        elif args.multi_algo:
            _run_multi_algo(args)
        elif args.plan:
            _run_plan(args)
        elif args.xai:
            _run_xai(args)
        elif args.eval:
            _run_eval(args)
    except KeyboardInterrupt:
        print("\n[run] Interrupted by user.")
        sys.exit(0)

    elapsed = time.time() - t0
    print(f"\n[run] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
