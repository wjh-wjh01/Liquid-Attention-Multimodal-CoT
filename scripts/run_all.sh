#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/prepare_data.py --config configs/datasets/local_v1.yaml
python3 scripts/run_experiments.py --config configs/base.yaml --matrix configs/experiments/experiment_matrix.yaml
MPLCONFIGDIR=/tmp/mpl python3 scripts/collect_results.py --runs-dir outputs/runs --output-dir outputs/reports

echo "Pipeline completed. Reports in outputs/reports"
