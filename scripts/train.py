import sys
import argparse
from pathlib import Path

# Running `python scripts/train.py` puts `scripts/` rather than the repository
# root on sys.path. Prefer this checkout over a stale installed fastwam package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SRC = _REPO_ROOT / "src"
if _LOCAL_SRC.is_dir():
    local_src = str(_LOCAL_SRC)
    if local_src not in sys.path:
        sys.path.insert(0, local_src)

# Avoid retaining transformed Arrow tables while iterating very large datasets.
import datasets as _hf_datasets
_hf_datasets.disable_caching()

from omegaconf import OmegaConf

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _add_if_set(overrides: list[str], key: str, value):
    if value is not None:
        overrides.append(f"{key}={value}")


def _parse_direct_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Train CSWAM from one explicit YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a complete training YAML config.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--lr_scheduler_type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--host_memory_trim_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_num_inference_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_direct_args(argv)
    cfg = OmegaConf.load(args.config)

    overrides: list[str] = []
    _add_if_set(overrides, "output_dir", args.output_dir)
    _add_if_set(overrides, "batch_size", args.batch_size)
    _add_if_set(overrides, "num_workers", args.num_workers)
    _add_if_set(overrides, "learning_rate", args.learning_rate)
    _add_if_set(overrides, "weight_decay", args.weight_decay)
    _add_if_set(overrides, "num_epochs", args.num_epochs)
    _add_if_set(overrides, "max_steps", args.max_steps)
    _add_if_set(overrides, "gradient_accumulation_steps", args.gradient_accumulation_steps)
    _add_if_set(overrides, "mixed_precision", args.mixed_precision)
    _add_if_set(overrides, "lr_scheduler_type", args.lr_scheduler_type)
    _add_if_set(overrides, "seed", args.seed)
    _add_if_set(overrides, "max_grad_norm", args.max_grad_norm)
    _add_if_set(overrides, "log_every", args.log_every)
    _add_if_set(overrides, "host_memory_trim_every", args.host_memory_trim_every)
    _add_if_set(overrides, "save_every", args.save_every)
    _add_if_set(overrides, "eval_every", args.eval_every)
    _add_if_set(overrides, "eval_num_inference_steps", args.eval_num_inference_steps)
    _add_if_set(overrides, "resume", args.resume)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    run_training(cfg)


if __name__ == "__main__":
    main()
