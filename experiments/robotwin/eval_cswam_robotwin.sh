#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
HOST="${CSWAM_HOST:-127.0.0.1}"
PORT="${CSWAM_PORT:-8772}"
MODE="${MODE:-clean}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SEED="${SEED:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
ROBOTWIN_GPU="${ROBOTWIN_GPU:-0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-36000}"
RUN_NAME="${RUN_NAME:-cswam_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/evaluate_results/cswam/$RUN_NAME}"

[[ -n "$ROBOTWIN_ROOT" ]] || { echo "ROBOTWIN_ROOT must be set." >&2; exit 2; }
if [[ "$ROBOTWIN_PYTHON" == */* ]]; then
  [[ -x "$ROBOTWIN_PYTHON" ]] || { echo "RoboTwin Python not found: $ROBOTWIN_PYTHON" >&2; exit 2; }
else
  command -v "$ROBOTWIN_PYTHON" >/dev/null || { echo "RoboTwin Python not found: $ROBOTWIN_PYTHON" >&2; exit 2; }
fi
[[ -f "$ROBOTWIN_ROOT/script/eval_policy.py" ]] || {
  echo "Invalid ROBOTWIN_ROOT: $ROBOTWIN_ROOT" >&2
  exit 2
}
case "$MODE" in
  clean|random|both) ;;
  *) echo "MODE must be clean, random, or both; got $MODE" >&2; exit 2 ;;
esac

ALL_TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
  handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
  move_playingcard_away move_stapler_pad open_laptop open_microwave
  pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
  place_bread_basket place_bread_skillet place_burger_fries place_can_basket
  place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
  place_fan place_mouse_pad place_object_basket place_object_scale
  place_object_stand place_phone_stand place_shoe press_stapler
  put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
  shake_bottle_horizontally stack_blocks_three stack_blocks_two
  stack_bowls_three stack_bowls_two stamp_seal turn_switch
)
if [[ -n "${TASKS:-}" ]]; then
  IFS=',' read -r -a SELECTED_TASKS <<< "$TASKS"
else
  SELECTED_TASKS=("${ALL_TASKS[@]}")
fi

export ROBOTWIN_SKIP_FRONT_CAMERA=1
export CUDA_VISIBLE_DEVICES="$ROBOTWIN_GPU"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT"
FAILURES=0

run_mode() {
  local evaluation_mode="$1"
  local mode_dir="$OUTPUT_ROOT/$evaluation_mode"
  mkdir -p "$mode_dir"
  for task in "${SELECTED_TASKS[@]}"; do
    local log_file="$mode_dir/${task}.log"
    echo "[$(date '+%F %T')] cswam start mode=$evaluation_mode task=$task"
    set +e
    timeout "$TASK_TIMEOUT" "$ROBOTWIN_PYTHON" -u \
      "$SCRIPT_DIR/eval_cswam_robotwin.py" \
      --robotwin-root "$ROBOTWIN_ROOT" \
      --task "$task" \
      --mode "$evaluation_mode" \
      --num-episodes "$NUM_EPISODES" \
      --seed "$SEED" \
      --host "$HOST" \
      --port "$PORT" \
      --replan-steps "$REPLAN_STEPS" \
      --run-name "$RUN_NAME" \
      > >(tee "$log_file") 2>&1
    local status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "[$(date '+%F %T')] cswam failed status=$status mode=$evaluation_mode task=$task" >&2
      FAILURES=$((FAILURES + 1))
    fi
  done
}

echo "CSWAM evaluation: mode=$MODE episodes=$NUM_EPISODES replan=$REPLAN_STEPS"
echo "RoboTwin: $ROBOTWIN_ROOT"
echo "Server: $HOST:$PORT"
echo "Logs: $OUTPUT_ROOT"
[[ "$MODE" == "clean" || "$MODE" == "both" ]] && run_mode clean
[[ "$MODE" == "random" || "$MODE" == "both" ]] && run_mode random
if (( FAILURES > 0 )); then
  echo "CSWAM evaluation finished with $FAILURES failed task runs." >&2
  exit 1
fi
echo "CSWAM evaluation finished successfully."
