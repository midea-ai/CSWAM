#!/usr/bin/env python
"""Precompute FastWAM text embeddings without Hydra.

The interface is intentionally narrow:

    python scripts/precompute_text_embeds_direct.py \
      --dataset-yaml configs/data/robotwin.yaml \
      --cache-dir /path/to/text_embeds_cache

The dataset YAML may be either:

    dataset_dirs: [
      /path/to/dataset_a,
      /path/to/dataset_b
    ]

or:

    dataset_dirs:
      - /path/to/dataset_a
      - /path/to/dataset_b

or a top-level list:

    - /path/to/dataset_a
    - /path/to/dataset_b

It writes the same cache filenames consumed by RobotVideoDataset:
`{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
ENC_ID = "wan22ti2v5b"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"

logger = logging.getLogger("precompute_text_embeds_direct")

torch = None
dist = None
tqdm = None
_load_registered_model = None
_resolve_configs = None
HuggingfaceTokenizer = None


def import_runtime_deps():
    """Import torch/model code only when encoding is actually needed."""
    global torch, dist, tqdm, _load_registered_model, _resolve_configs, HuggingfaceTokenizer
    if torch is not None:
        return

    import torch as torch_mod
    import torch.distributed as dist_mod
    from tqdm import tqdm as tqdm_mod

    from fastwam.models.wan22.helpers.loader import (
        _load_registered_model as load_registered_model,
        _resolve_configs as resolve_configs,
    )
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer as Tokenizer

    torch = torch_mod
    dist = dist_mod
    tqdm = tqdm_mod
    _load_registered_model = load_registered_model
    _resolve_configs = resolve_configs
    HuggingfaceTokenizer = Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute Wan2.2 T5 text embedding cache from explicit dataset/cache paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        nargs="+",
        default=[],
        help="LeRobot dataset dir(s), a root containing subdatasets, or a root with dataloader.json.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help=(
            "YAML containing a top-level dataset config or data.train. "
            "All dataset_dirs/dataset_w_subtask_dirs/dataset_wo_subtask_dirs are included."
        ),
    )
    parser.add_argument("--cache-dir", required=True, help="Output text embedding cache directory.")
    parser.add_argument(
        "--override-instruction",
        default=None,
        help="Encode exactly this one instruction instead of scanning dataset task metadata.",
    )
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Compatibility no-op. Existing cache files are overwritten by default.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prompts whose cache file already exists. By default this script mirrors the original script and overwrites.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show prompt counts without loading torch/model.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def unique_paths(paths: list[str]) -> list[Path]:
    result, seen = [], set()
    for raw in paths:
        path = Path(raw).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def load_dataset_paths_from_yaml(path: str) -> list[str]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading --dataset-yaml requires PyYAML (`pip install pyyaml`).") from exc

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []

    if isinstance(data, list):
        raw_paths = data
    elif isinstance(data, dict):
        dataset_cfg = data
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("train"), dict):
            dataset_cfg = data["data"]["train"]
        elif isinstance(data.get("train"), dict):
            dataset_cfg = data["train"]

        raw_paths = []
        for key in (
            "dataset_dirs",
            "dataset_w_subtask_dirs",
            "dataset_wo_subtask_dirs",
            "dataset_paths",
            "datasets",
        ):
            values = dataset_cfg.get(key)
            if values is not None:
                if not isinstance(values, list):
                    raise TypeError(f"`{key}` in {config_path} must be a list.")
                raw_paths.extend(values)
        if not raw_paths:
            raise KeyError(f"{config_path} does not contain any supported dataset path list.")
    else:
        raise TypeError(f"{config_path} must be a list or mapping, got {type(data).__name__}.")

    if not isinstance(raw_paths, list):
        raise TypeError(f"Dataset paths in {config_path} must be a list.")

    paths = []
    for idx, item in enumerate(raw_paths):
        if isinstance(item, str):
            raw_path = item
        elif isinstance(item, dict):
            raw_path = item.get("path") or item.get("dataset_dir") or item.get("root")
            if raw_path is None:
                raise KeyError(
                    f"Dataset item {idx} in {config_path} must contain path, dataset_dir, or root."
                )
        else:
            raise TypeError(
                f"Dataset item {idx} in {config_path} must be a string or mapping, got {type(item).__name__}."
            )

        ds_path = Path(str(raw_path)).expanduser()
        if not ds_path.is_absolute():
            ds_path = (config_path.parent / ds_path).resolve()
        paths.append(str(ds_path))

    return paths


def resolve_dataset_paths(args) -> list[Path]:
    raw_paths = []
    if args.dataset_yaml is not None:
        raw_paths.extend(load_dataset_paths_from_yaml(args.dataset_yaml))
    raw_paths.extend(args.dataset_dir)
    return unique_paths(raw_paths)


def expand_dataset_dirs(paths: list[Path]) -> list[Path]:
    """Match the repo's LeRobot root expansion: dataset, dataloader.json, or subdatasets."""
    expanded: list[Path] = []
    seen: set[str] = set()

    def add(path: Path):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            expanded.append(path)

    for path in paths:
        if not path.exists():
            logger.warning("Dataset path does not exist, skipping: %s", path)
            continue

        if (path / "meta" / "info.json").exists():
            add(path)
            continue

        dataloader_json = path / "dataloader.json"
        if dataloader_json.exists():
            with dataloader_json.open("r", encoding="utf-8") as f:
                config = json.load(f)
            for raw_child in config.get("dataset_paths", []):
                child = Path(raw_child).expanduser()
                child = child if child.is_absolute() else (path / child).resolve()
                if (child / "meta" / "info.json").exists():
                    add(child)
                else:
                    logger.warning("Invalid dataloader.json dataset path, skipping: %s", child)
            continue

        children = sorted(p for p in path.iterdir() if (p / "meta" / "info.json").exists())
        if children:
            logger.info("Expanded %s into %d subdatasets.", path, len(children))
            for child in children:
                add(child)
        else:
            logger.warning("No valid LeRobot dataset found under %s", path)

    return expanded


def read_tasks_parquet(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    df = pq.read_table(path).to_pandas()
    if df.index.name is None and "__index_level_0__" in df.columns:
        df = df.set_index("__index_level_0__")
    return [str(task_text) for task_text in df.index if str(task_text).strip()]


def read_tasks_jsonl(path: Path) -> list[str]:
    tasks: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_no}.")

            # Standard LeRobot v2.1 task row.
            if "task" in item:
                task = str(item["task"])
                if task.strip():
                    tasks.append(task)

            # Segment annotations used by dataset_w_subtask_dirs. One line may
            # contain one or several episode-indexed records.
            candidates = [item]
            candidates.extend(value for value in item.values() if isinstance(value, dict))
            for record in candidates:
                reasoning_list = record.get("reasoning_list")
                index_list = record.get("index_list")
                if not isinstance(reasoning_list, list) or not isinstance(index_list, list):
                    continue
                for task in reasoning_list:
                    task = str(task)
                    if task.strip():
                        tasks.append(task)
    return tasks


def build_prompt(task: str) -> str:
    """Mirror RobotVideoDataset exactly; prompt whitespace affects the cache hash."""
    return PROMPT_TEMPLATE.format(task=str(task))


def collect_prompts(dataset_dirs, override_instruction):
    if override_instruction and override_instruction.strip():
        prompt = build_prompt(override_instruction)
        return [prompt], 0, 1, [prompt]

    expanded_dirs = expand_dataset_dirs(dataset_dirs)

    prompts = []
    seen = set()
    first_prompts = []
    task_rows = 0

    for dataset_dir in expanded_dirs:
        task_files = [
            dataset_dir / "meta" / "tasks.parquet",
            dataset_dir / "meta" / "tasks.jsonl",
        ]
        existing_files = [path for path in task_files if path.exists()]
        if not existing_files:
            raise FileNotFoundError(
                f"Missing meta/tasks.parquet and meta/tasks.jsonl under {dataset_dir}"
            )

        raw_tasks: list[str] = []
        if task_files[0].exists():
            raw_tasks.extend(read_tasks_parquet(task_files[0]))
        if task_files[1].exists():
            raw_tasks.extend(read_tasks_jsonl(task_files[1]))

        for task in raw_tasks:
            task_rows += 1
            prompt = build_prompt(task)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
                if len(first_prompts) < 5:
                    first_prompts.append(prompt)

    return prompts, len(expanded_dirs), task_rows, first_prompts


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def current_device(is_distributed: bool, local_rank: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{local_rank}" if is_distributed else "cuda"


def cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.t5_len{context_len}.{ENC_ID}.pt"


def atomic_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def load_text_encoder(context_len: int, device: str):
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=MODEL_ID,
        tokenizer_model_id=TOKENIZER_MODEL_ID,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )
    return text_encoder, tokenizer


def encode_prompts(prompts: list[str], cache_dir: Path, args, rank: int, world_size: int, device: str):
    todo, skipped = [], 0
    for prompt in prompts:
        path = cache_path(cache_dir, prompt, args.context_len)
        if path.exists() and args.skip_existing:
            skipped += 1
        else:
            todo.append(prompt)

    logger.info("Rank %d/%d: skipped=%d encode=%d", rank, world_size, skipped, len(todo))
    if not todo:
        return 0, skipped, 0

    text_encoder, tokenizer = load_text_encoder(args.context_len, device)
    new_or_overwrite = 0
    over_length = 0

    with tqdm(
        total=len(todo),
        desc=f"Encoding prompts rank {rank}/{world_size}" if world_size > 1 else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=world_size > 1 and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start : start + args.batch_size]
                
                ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = text_encoder(ids, mask)
                over_length += int(mask.all(dim=1).sum().item())

                for i, prompt in enumerate(batch):
                    payload = {
                        "context": context[i].detach().cpu().to(torch.bfloat16).contiguous(),
                        "mask": mask[i].detach().cpu().bool().contiguous(),
                    }
                    atomic_save(payload, cache_path(cache_dir, prompt, args.context_len))
                    new_or_overwrite += 1
                pbar.update(len(batch))

    return new_or_overwrite, skipped, over_length


def reduce_counts(counts: tuple[int, int, int], device: str) -> tuple[int, int, int]:
    if not dist.is_initialized():
        return counts
    tensor = torch.tensor(counts, device=torch.device(device), dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tuple(int(x) for x in tensor.cpu().tolist())


def main():
    setup_logging()
    args = parse_args()
    if args.context_len <= 0 or args.batch_size <= 0:
        raise ValueError("--context-len and --batch-size must be positive.")

    dataset_dirs = resolve_dataset_paths(args)
    if not dataset_dirs and not (args.override_instruction and args.override_instruction.strip()):
        raise ValueError("Please pass --dataset-yaml or --dataset-dir.")

    cache_dir = Path(args.cache_dir).expanduser()
    prompts, expanded_count, task_rows, first_prompts = collect_prompts(dataset_dirs, args.override_instruction)
    if not prompts:
        logger.warning("No prompts found; nothing to do.")
        return 0

    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("Expanded datasets: %d", expanded_count)
        logger.info("Task rows read: %d", task_rows)
        logger.info("Unique prompts: %d", len(prompts))
        logger.info("Cache dir: %s", cache_dir)
        logger.info("Cache id: t5_len%d.%s", args.context_len, ENC_ID)
        for idx, prompt in enumerate(first_prompts, start=1):
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            logger.info("Prompt sample %d: %r", idx, prompt)
            logger.info("Prompt sample %d hash: %s", idx, digest)

    if args.dry_run:
        logger.info("Dry run: exiting before torch/model imports.")
        return 0

    import_runtime_deps()
    is_dist, rank, world_size, local_rank = init_distributed()
    local_prompts = prompts[rank::world_size] if is_dist else prompts
    device = current_device(is_dist, local_rank)
    counts = encode_prompts(local_prompts, cache_dir, args, rank, world_size, device)
    new_or_overwrite, skipped, over_length = reduce_counts(counts, device)

    if rank == 0:
        logger.info("Finished. wrote=%d skipped=%d over_length=%d", new_or_overwrite, skipped, over_length)

    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
