#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/synthetic.yaml}

for SEED in 13 21 42; do
  python scripts/run_experiment.py \
    --config "$CONFIG" \
    --set train.seed="$SEED" \
    --set train.output_dir="outputs/seed_${SEED}"
done
