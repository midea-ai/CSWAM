#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-${1:-$REPO_ROOT/step_028640.pt}}"
CONFIG="${CSWAM_CONFIG:-$REPO_ROOT/configs/train/cswam_robotwin_16d.yaml}"
DATASET_STATS="${CSWAM_DATASET_STATS:-}"
VJEPA_REPO="${CSWAM_VJEPA_REPO:-}"
VJEPA_CHECKPOINT="${CSWAM_VJEPA_CHECKPOINT:-}"
MODEL_CACHE="${DIFFSYNTH_MODEL_BASE_PATH:-}"
MODEL_PYTHON="${CSWAM_PYTHON:-python}"

HOST="${CSWAM_HOST:-127.0.0.1}"
PORT="${CSWAM_PORT:-8772}"
DEVICE="${CSWAM_DEVICE:-cuda:0}"
INFERENCE_STEPS="${CSWAM_INFERENCE_STEPS:-10}"
TEXT_ENCODER_DEVICE="${CSWAM_TEXT_ENCODER_DEVICE:-cpu}"
VAE_DEVICE_MODE="${CSWAM_VAE_DEVICE_MODE:-cpu}"
TEXT_CACHE_SIZE="${CSWAM_TEXT_CACHE_SIZE:-64}"

for name in DATASET_STATS VJEPA_REPO VJEPA_CHECKPOINT MODEL_CACHE; do
  if [[ -z "${!name}" ]]; then
    echo "$name must be set; see experiments/robotwin/README.md" >&2
    exit 2
  fi
done
if [[ "$MODEL_PYTHON" == */* ]]; then
  [[ -x "$MODEL_PYTHON" ]] || { echo "CSWAM Python not found: $MODEL_PYTHON" >&2; exit 2; }
else
  command -v "$MODEL_PYTHON" >/dev/null || { echo "CSWAM Python not found: $MODEL_PYTHON" >&2; exit 2; }
fi
for path in "$CHECKPOINT" "$CONFIG" "$DATASET_STATS" "$VJEPA_REPO" \
  "$VJEPA_CHECKPOINT" "$MODEL_CACHE"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_CACHE"
export MODELSCOPE_NO_DOWNLOAD="${MODELSCOPE_NO_DOWNLOAD:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Evaluator         : cswam"
echo "Checkpoint        : $CHECKPOINT"
echo "Config            : $CONFIG"
echo "Stats             : $DATASET_STATS"
echo "V-JEPA repository : $VJEPA_REPO"
echo "V-JEPA checkpoint : $VJEPA_CHECKPOINT"
echo "Wan model cache   : $MODEL_CACHE"
echo "Denoising steps   : $INFERENCE_STEPS"
echo "Server            : $HOST:$PORT"
echo "============================================================"

exec "$MODEL_PYTHON" -u "$SCRIPT_DIR/cswam_rpc_server.py" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --stats "$DATASET_STATS" \
  --vjepa-repo "$VJEPA_REPO" \
  --vjepa-checkpoint "$VJEPA_CHECKPOINT" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --mixed-precision bf16 \
  --num-inference-steps "$INFERENCE_STEPS" \
  --text-encoder-device "$TEXT_ENCODER_DEVICE" \
  --vae-device-mode "$VAE_DEVICE_MODE" \
  --text-cache-size "$TEXT_CACHE_SIZE"
