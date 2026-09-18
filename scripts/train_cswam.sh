#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# Network and multi-node settings
# =========================================================================
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,en0}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -r /etc/volcano/worker.host ]]; then
    MASTER_ADDR="$(awk 'NR == 1 {print $1}' /etc/volcano/worker.host)"
  else
    MASTER_ADDR="127.0.0.1"
  fi
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29604}"
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-${MACHINE_RANK:-0}}}"

# =========================================================================
# Training settings
# Defaults to 14D RoboTwin data with a 16D model/checkpoint interface.
# Change CONFIG to one of the four public CSWAM YAMLs for another mode.
# Every value can also be overridden as an environment variable.
# =========================================================================
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}"
CONFIG="${CONFIG:-configs/train/cswam_robotwin_16d.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"
TASK_NAME="${TASK_NAME:-cswam_robotwin_16d}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${TASK_NAME}/$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
SEED="${SEED:-42}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-10}"
RESUME="${RESUME:-}"

mkdir -p "${OUTPUT_DIR}"

OPTIONAL_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  OPTIONAL_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${HOST_MEMORY_TRIM_EVERY:-}" ]]; then
  OPTIONAL_ARGS+=(--host_memory_trim_every "${HOST_MEMORY_TRIM_EVERY}")
fi
if [[ -n "${RESUME}" ]]; then
  OPTIONAL_ARGS+=(--resume "${RESUME}")
fi

echo "------------------------------------------------"
echo "Master Node IP: ${MASTER_ADDR}"
echo "Master Port   : ${MASTER_PORT}"
echo "This Node Rank: ${NODE_RANK} (Total Nodes: ${NNODES})"
echo "Total GPUs    : ${TOTAL_GPUS}"
echo "Config        : ${CONFIG}"
echo "Output Dir    : ${OUTPUT_DIR}"
echo "Entry         : scripts/train.py"
echo "------------------------------------------------"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --seed "${SEED}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS}" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
