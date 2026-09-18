"""Run one CSWAM task through an external official RoboTwin checkout."""

from __future__ import annotations

import argparse
import builtins
import io
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


THIS_DIR = Path(__file__).resolve().parent


def _optimized_task_config(robotwin_root: Path, mode: str) -> dict[str, Any]:
    base_name = "demo_clean" if mode == "clean" else "demo_randomized"
    base_path = robotwin_root / "task_config" / f"{base_name}.yml"
    if not base_path.is_file():
        raise FileNotFoundError(f"RoboTwin task config not found: {base_path}")
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["render_freq"] = 0
    config["collect_data"] = False
    config["eval_video_log"] = False
    config["camera"]["collect_head_camera"] = True
    config["camera"]["collect_wrist_camera"] = True
    config["data_type"].update(
        {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": False,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        }
    )
    return config


@contextmanager
def _virtual_task_config(path: Path, config: dict[str, Any]) -> Iterator[None]:
    """Serve one generated YAML without writing into the RoboTwin checkout."""
    original_open = builtins.open
    target = path.resolve()
    serialized = yaml.safe_dump(config, sort_keys=False)

    def redirected_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        try:
            candidate = Path(file)
            candidate = candidate if candidate.is_absolute() else Path.cwd() / candidate
            matches = candidate.resolve() == target
        except (OSError, TypeError, ValueError):
            matches = False
        if matches and "r" in mode:
            return io.StringIO(serialized)
        return original_open(file, mode, *args, **kwargs)

    builtins.open = redirected_open
    try:
        yield
    finally:
        builtins.open = original_open


def _load_official_evaluator(robotwin_root: Path, num_episodes: int):
    script = robotwin_root / "script" / "eval_policy.py"
    if not script.is_file():
        raise FileNotFoundError(f"Official RoboTwin evaluator not found: {script}")

    search_paths = (
        THIS_DIR,
        robotwin_root,
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
    )
    for path in reversed(search_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    source = script.read_text(encoding="utf-8")
    marker = "test_num = 100"
    if source.count(marker) != 1:
        raise RuntimeError(
            "Unsupported RoboTwin eval_policy.py: expected exactly one "
            f"{marker!r} assignment."
        )
    source = source.replace(marker, f"test_num = {int(num_episodes)}", 1)
    module_name = "cswam_official_robotwin_eval"
    module = types.ModuleType(module_name)
    module.__file__ = str(script)
    sys.modules[module_name] = module
    exec(compile(source, str(script), "exec"), module.__dict__)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=("clean", "random"), default="clean")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--connect-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=1800)
    parser.add_argument("--run-name", default="cswam")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    robotwin_root = Path(args.robotwin_root).expanduser().resolve()
    config = _optimized_task_config(robotwin_root, args.mode)
    # RoboTwin imports several asset lists relative to the process CWD.
    os.environ.setdefault("ROBOTWIN_SKIP_FRONT_CAMERA", "1")
    os.chdir(robotwin_root)
    evaluator = _load_official_evaluator(robotwin_root, args.num_episodes)
    config_name = f"__cswam_{args.mode}"
    virtual_path = robotwin_root / "task_config" / f"{config_name}.yml"
    instruction_type = "seen" if args.mode == "clean" else "unseen"
    user_args = {
        "task_name": args.task,
        "task_config": config_name,
        "ckpt_setting": args.run_name,
        "policy_name": "cswam_policy",
        "instruction_type": instruction_type,
        "seed": args.seed,
        "model_host": args.host,
        "model_port": args.port,
        "replan_steps": args.replan_steps,
        "model_seed": args.model_seed,
        "connect_timeout_s": args.connect_timeout_s,
        "request_timeout_s": args.request_timeout_s,
    }

    print(
        "[cswam-eval] "
        f"task={args.task} mode={args.mode} episodes={args.num_episodes} "
        f"server={args.host}:{args.port} replan={args.replan_steps}",
        flush=True,
    )
    with _virtual_task_config(virtual_path, config):
        evaluator.main(user_args)


if __name__ == "__main__":
    main()
