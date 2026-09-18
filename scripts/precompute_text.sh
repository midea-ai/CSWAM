#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATASET_YAML="${DATASET_YAML:-configs/data/robotwin.yaml}"
CACHE_DIR="${CACHE_DIR:-/efs/share/1919650160032350208/projects/foundation_model/datasets/Fastwam_data/robotwin_clean/text_embeds_cache}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/precompute_text_embeds_direct.py \
  --dataset-yaml "${DATASET_YAML}" \
  --cache-dir "${CACHE_DIR}" \
  --context-len "${CONTEXT_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --skip-existing \
  "$@"
