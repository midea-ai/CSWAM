# CSWAM RoboTwin Evaluation

This directory evaluates CSWAM without vendoring RoboTwin, SAPIEN, Wan, or
V-JEPA assets. The model and simulator run in separate Python environments and
communicate through a localhost socket.

## Evaluation Contract

- The action horizon is read from the selected CSWAM YAML (`33 - 1 = 32`).
- `REPLAN_STEPS` defaults to 24 and must be divisible by the history stride 4.
- Eight mosaic history frames use offsets
  `[-28, -24, -20, -16, -12, -8, -4, 0]`.
- Only steps `0, 4, 8, ...` request a rendered observation.
- Each retained frame is one `384x320` uint8 head/wrist mosaic.
- Front view, depth, point cloud, segmentation, data collection, and video
  output are disabled.
- T5 and the first-frame VAE run on CPU by default; text contexts are cached.
- Both native 14D and padded 16D model checkpoints produce RoboTwin 14D actions.

The RPC protocol uses pickle and is intended only for a trusted host. Keep
`CSWAM_HOST=127.0.0.1` unless the network is otherwise secured.

## Check A Checkpoint

This validates dimensions and metadata without loading model components:

```bash
python experiments/robotwin/cswam_rpc_server.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/train/cswam_robotwin_16d.yaml \
  --stats /path/to/robotwin_14d_stats.json \
  --inspect-only
```

Use the 14D YAML for a native 14D checkpoint. A 16D checkpoint still requires
the raw 14D RoboTwin statistics used during training.

## Start The Model Server

```bash
CSWAM_PYTHON=/path/to/cswam/bin/python \
CSWAM_CONFIG=configs/train/cswam_robotwin_16d.yaml \
CSWAM_DATASET_STATS=/path/to/robotwin_14d_stats.json \
CSWAM_VJEPA_REPO=/path/to/vjepa2 \
CSWAM_VJEPA_CHECKPOINT=/path/to/vjepa2_checkpoint.pt \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan_model_cache \
  bash experiments/robotwin/run_cswam_server.sh /path/to/checkpoint.pt
```

Useful overrides are `CSWAM_DEVICE`, `CSWAM_PORT`,
`CSWAM_INFERENCE_STEPS`, `CSWAM_VAE_DEVICE_MODE`,
`CSWAM_TEXT_ENCODER_DEVICE`, and `CSWAM_TEXT_CACHE_SIZE`.

## Run RoboTwin

In a second terminal:

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/robotwin/bin/python \
MODE=clean NUM_EPISODES=1 TASKS=adjust_bottle REPLAN_STEPS=24 \
  bash experiments/robotwin/eval_cswam_robotwin.sh
```

`MODE` accepts `clean`, `random`, or `both`. `TASKS` accepts comma-separated
official task names and defaults to all 50 tasks. Logs are saved under
`evaluate_results/cswam/`.
