#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Keep these values together so this file can be edited like the training
# launcher. Environment variables may still override them for batch jobs.
PYTHON="${PYTHON:-python}"
DATASET_YAML="${DATASET_YAML:-configs/data/robotwin.yaml}"
OUTPUT="${OUTPUT:-runs/robotwin_stats.json}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-30}"

"${PYTHON}" scripts/precompute_stats_optimize.py \
  --dataset-yaml "${DATASET_YAML}" \
  --output "${OUTPUT}" \
  --num-workers "${NUM_WORKERS}" \
  --skip-quantile \
  --profile \
  --profile-interval "${PROFILE_INTERVAL}" \
  "$@"
