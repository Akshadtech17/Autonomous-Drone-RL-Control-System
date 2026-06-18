#!/usr/bin/env bash
# Run 5 independent training jobs with different seeds.
# Each saves its model to models/seed_{seed}/ and metrics to logs/metrics_seed_{seed}.json
#
# Usage: bash train/run_seeds.sh [--timesteps 200000]

TIMESTEPS=${1:-200000}
SEEDS=(42 123 456 789 1337)

echo "========================================"
echo "Multi-seed training  timesteps=$TIMESTEPS"
echo "Seeds: ${SEEDS[*]}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED ---"
    python -m train.train_ppo --seed "$SEED" --timesteps "$TIMESTEPS"
    # rename metrics so plot_variance.py can find them
    if [ -f "logs/metrics.json" ]; then
        cp "logs/metrics.json" "logs/metrics_seed_${SEED}.json"
        echo "Saved metrics → logs/metrics_seed_${SEED}.json"
    fi
done

echo ""
echo "========================================"
echo "All seeds done. Plot variance:"
echo "  python -m evaluate.plot_variance"
echo "========================================"
