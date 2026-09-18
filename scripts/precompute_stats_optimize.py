#!/usr/bin/env python3
"""
Pre-compute dataset normalization stats and save to JSON.

Example:
  python scripts/precompute_stats_optimize.py \
    --dataset-yaml configs/data/robotwin.yaml \
    --output runs/robotwin_stats.json \
    --skip-quantile

目标：
- 尽量逐位对齐原版 BaseLerobotDataset.get_dataset_stats()
- 避免构造完整 HuggingFace Dataset，只读取 state/action parquet columns
- 避免将 episode 级统计写成大量 .npy 小文件
- 最终输出 JSON 格式与原版一致

实现方式：
- 每个 episode 先按原版算 ep_min / ep_max / ep_mean / ep_var / ep_q01 / ep_q99
- episode 级小 tensor 立即在线合并，不再落盘或保存所有 episode 中间结果
"""

from __future__ import annotations

import sys
import argparse
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from threading import Lock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OmegaConf = None
np = None
torch = None
tqdm = None


def import_config_deps():
    global OmegaConf
    if OmegaConf is not None:
        return

    from omegaconf import OmegaConf as OmegaConf_mod

    OmegaConf = OmegaConf_mod


def import_runtime_deps():
    global np, torch, tqdm
    if torch is not None:
        return

    import numpy as np_mod
    import torch as torch_mod
    from tqdm import tqdm as tqdm_mod

    np = np_mod
    torch = torch_mod
    tqdm = tqdm_mod


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pre-compute dataset normalization stats from one explicit dataset YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        required=True,
        default=None,
        help="Output stats JSON path.",
    )
    parser.add_argument("--tmp-dir", default=None, help="Deprecated no-op kept for CLI compatibility.")
    parser.add_argument("--keep-tmp", action="store_true", help="Deprecated no-op kept for CLI compatibility.")
    parser.add_argument(
        "--dataset-yaml",
        required=True,
        help="YAML file containing dataset paths and stats config fields.",
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dirs",
        dest="dataset_dirs",
        nargs="+",
        default=None,
        help="Override dataset_dirs from --dataset-yaml.",
    )
    parser.add_argument("--num_frames", type=int, default=None, help="Override num_frames from --dataset-yaml.")
    parser.add_argument(
        "--global_sample_stride",
        type=int,
        default=None,
        help="Override global_sample_stride from --dataset-yaml.",
    )
    parser.add_argument(
        "--processor_mode",
        choices=["eval", "train"],
        default="eval",
        help="Mode used for processor.action_state_transform().",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker threads for per-episode stat computation. 0 means serial.",
    )
    parser.add_argument(
        "--skip-quantile",
        action="store_true",
        help="Skip q01/q99 quantile computation and write q01=min, q99=max as fallback.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print timing breakdown for metadata loading, parquet I/O, tensor conversion, transforms, and reductions.",
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=30.0,
        help="When --profile is enabled, print a live timing summary every N seconds during stats computation. 0 disables live summaries.",
    )
    return parser.parse_args(argv)


class ProfileStats:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.lock = Lock()
        self.last_print_time = time.perf_counter()

    def add(self, name: str, seconds: float):
        if not self.enabled:
            return
        with self.lock:
            self.totals[name] += seconds
            self.counts[name] += 1

    def print_summary(self, wall_seconds: float, title: str = "Timing summary"):
        if not self.enabled:
            return
        print("------------------------------------------------")
        print(f"[PROFILE] {title}")
        print(f"[PROFILE] wall_total: {wall_seconds:.3f}s")
        print("[PROFILE] Note: some stages are nested, and totals can exceed wall time when --num-workers > 0.")
        print("[PROFILE] stage                         total_s    count      avg_ms   wall_%")
        for name, total in sorted(self.totals.items(), key=lambda item: item[1], reverse=True):
            count = self.counts[name]
            avg_ms = total / max(count, 1) * 1000.0
            pct = total / wall_seconds * 100.0 if wall_seconds > 0 else 0.0
            print(f"[PROFILE] {name:<28} {total:8.3f} {count:8d} {avg_ms:10.3f} {pct:7.1f}")
        print("------------------------------------------------")
        sys.stdout.flush()

    def maybe_print_summary(
        self,
        wall_seconds: float,
        interval_seconds: float,
        title: str,
        force: bool = False,
    ):
        if not self.enabled or interval_seconds <= 0:
            return
        now = time.perf_counter()
        if force or now - self.last_print_time >= interval_seconds:
            self.print_summary(wall_seconds, title=title)
            self.last_print_time = now


def unique_dataset_paths(paths: list[str]) -> list[str]:
    result, seen = [], set()
    for raw in paths:
        path = Path(str(raw)).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            result.append(str(path))
    return result


def load_dataset_yaml_config(path: str):
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    if cfg is None:
        return OmegaConf.create({})
    if OmegaConf.is_list(cfg):
        return OmegaConf.create({"dataset_dirs": cfg})
    if not OmegaConf.is_dict(cfg):
        raise TypeError(f"{config_path} must be a list or mapping, got {type(cfg).__name__}.")
    return cfg


def load_stats_config(args):
    cfg = load_dataset_yaml_config(args.dataset_yaml)
    OmegaConf.resolve(cfg)
    if "data" in cfg and cfg.data is not None and "train" in cfg.data:
        return cfg.data.train
    if "train" in cfg and cfg.train is not None:
        return cfg.train
    return cfg


def resolve_dataset_dirs(args, train_cfg) -> list[str]:
    raw_paths: list[str] = []
    if args.dataset_dirs is not None:
        raw_paths.extend(args.dataset_dirs)

    if raw_paths:
        return unique_dataset_paths(raw_paths)

    for key in (
        "dataset_dirs",
        "dataset_w_subtask_dirs",
        "dataset_wo_subtask_dirs",
        "dataset_paths",
        "datasets",
    ):
        values = train_cfg.get(key, None)
        if values is not None:
            raw_paths.extend(str(value) for value in values)

    if not raw_paths:
        raise ValueError(f"No dataset paths found in {args.dataset_yaml}.")
    return unique_dataset_paths(raw_paths)


def instantiate_from_config(cfg):
    """
    Small config object factory for this script.
    It supports the _target_ configs used by FastWAM processors and transforms.
    """
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if isinstance(cfg, list):
        return [instantiate_from_config(item) for item in cfg]

    if not isinstance(cfg, dict):
        return cfg

    if "_target_" not in cfg:
        return {key: instantiate_from_config(value) for key, value in cfg.items()}

    target = cfg["_target_"]
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = {
        key: instantiate_from_config(value)
        for key, value in cfg.items()
        if not key.startswith("_")
    }
    return cls(**kwargs)


def expand_dataset_dir(ds_dir: str) -> list[str]:
    ds_path = Path(ds_dir)
    if not ds_path.exists():
        print(f"[WARN] Dataset directory does not exist, skipping: {ds_dir}")
        return []

    if (ds_path / "meta" / "info.json").exists():
        return [str(ds_path)]

    json_config = ds_path / "dataloader.json"
    if json_config.exists():
        with json_config.open("r", encoding="utf-8") as f:
            config = json.load(f)
        resolved = []
        for raw_path in config.get("dataset_paths", []):
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (ds_path / path).resolve()
            if path.exists() and (path / "meta" / "info.json").exists():
                resolved.append(str(path))
            else:
                print(f"[WARN] Path from dataloader.json not valid, skipping: {path}")
        return sorted(resolved)

    subdatasets = sorted(
        str(subdir)
        for subdir in ds_path.iterdir()
        if subdir.is_dir() and (subdir / "meta" / "info.json").exists()
    )
    if subdatasets:
        return subdatasets

    print(f"[WARN] No valid dataset found under {ds_dir}")
    return []


def expand_dataset_dirs(dataset_dirs: list[str]) -> list[str]:
    expanded = []
    for ds_dir in dataset_dirs:
        expanded.extend(expand_dataset_dir(ds_dir))
    expanded = unique_dataset_paths(expanded)
    if not expanded:
        raise ValueError(f"No valid datasets found in: {dataset_dirs}")
    return expanded


def add_lerobot_keys(shape_meta: dict, fps: int, global_sample_stride: int):
    state_meta = shape_meta["state"]
    action_meta = shape_meta["action"]

    for meta in state_meta:
        key = meta["key"]
        meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"

    for meta in action_meta:
        key = meta["key"]
        meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"

    return state_meta, action_meta


def sliding_window_with_replication(x, window_size: int):
    assert x.dim() == 2
    assert window_size > 0
    n, _ = x.shape
    i_indices = torch.arange(n).unsqueeze(1)
    j_indices = torch.arange(window_size).unsqueeze(0)
    indices = torch.clamp(i_indices + j_indices, min=0, max=n - 1)
    return x[indices]


def arrow_array_to_numpy(array):
    try:
        return array.to_numpy(zero_copy_only=True)
    except Exception:
        return array.to_numpy(zero_copy_only=False)


def parquet_column_to_tensor(column):
    try:
        import pyarrow as pa

        array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
        array_type = array.type

        if pa.types.is_fixed_size_list(array_type):
            width = int(array_type.list_size)
            flat = arrow_array_to_numpy(array.values)
            np_arr = flat.reshape(len(array), width)
            return torch.from_numpy(np_arr)

        if pa.types.is_list(array_type) or pa.types.is_large_list(array_type):
            offsets = arrow_array_to_numpy(array.offsets)
            lengths = np.diff(offsets)
            if len(lengths) == 0:
                np_arr = np.empty((0, 0), dtype=np.float32)
                return torch.from_numpy(np_arr)
            if np.all(lengths == lengths[0]):
                width = int(lengths[0])
                flat = arrow_array_to_numpy(array.values)
                flat = flat[int(offsets[0]): int(offsets[-1])]
                np_arr = flat.reshape(len(array), width)
                return torch.from_numpy(np_arr)

        if pa.types.is_integer(array_type) or pa.types.is_floating(array_type) or pa.types.is_boolean(array_type):
            return torch.from_numpy(arrow_array_to_numpy(array))
    except Exception:
        pass

    try:
        np_arr = column.to_numpy(zero_copy_only=True)
    except Exception:
        raw = column.to_numpy()
        np_arr = np.stack(raw) if raw.dtype == object else raw

    if np_arr.dtype == object:
        np_arr = np.stack(np_arr)
    return torch.from_numpy(np_arr)


def variance_from_sum(sum_, sumsq, count: int):
    if count <= 1:
        return torch.full_like(sum_, float("nan"))
    var = (sumsq - sum_.square() / count) / (count - 1)
    return var.clamp_min(0)


def state_tensor_stats(state: torch.Tensor):
    if state.ndim == 1:
        state = state.unsqueeze(-1)
    count = state.shape[0]
    state_sum = state.sum(0)
    state_sumsq = state.square().sum(0)
    return {
        "min": state.amin(0).unsqueeze(0),
        "max": state.amax(0).unsqueeze(0),
        "mean": (state_sum / count).unsqueeze(0),
        "var": variance_from_sum(state_sum, state_sumsq, count).unsqueeze(0),
    }


def action_window_stats(action: torch.Tensor, window_size: int):
    if action.ndim == 1:
        action = action.unsqueeze(-1)
    count, dim = action.shape
    if count == 0:
        empty = torch.empty((window_size, dim), dtype=action.dtype, device=action.device)
        return {"min": empty, "max": empty, "mean": empty, "var": empty}

    action_sq = action.square()
    prefix_sum = torch.cumsum(action, dim=0)
    prefix_sumsq = torch.cumsum(action_sq, dim=0)
    total_sum = prefix_sum[-1]
    total_sumsq = prefix_sumsq[-1]

    rev_action = torch.flip(action, dims=[0])
    suffix_min = torch.flip(torch.cummin(rev_action, dim=0).values, dims=[0])
    suffix_max = torch.flip(torch.cummax(rev_action, dim=0).values, dims=[0])
    last = action[-1]
    last_sq = action_sq[-1]

    mins, maxs, means, vars_ = [], [], [], []
    for offset in range(window_size):
        if offset < count:
            if offset == 0:
                cur_sum = total_sum
                cur_sumsq = total_sumsq
            else:
                cur_sum = total_sum - prefix_sum[offset - 1]
                cur_sumsq = total_sumsq - prefix_sumsq[offset - 1]
            cur_sum = cur_sum + offset * last
            cur_sumsq = cur_sumsq + offset * last_sq
            cur_min = suffix_min[offset]
            cur_max = suffix_max[offset]
        else:
            cur_sum = count * last
            cur_sumsq = count * last_sq
            cur_min = last
            cur_max = last

        mins.append(cur_min)
        maxs.append(cur_max)
        means.append(cur_sum / count)
        vars_.append(variance_from_sum(cur_sum, cur_sumsq, count))

    return {
        "min": torch.stack(mins, dim=0),
        "max": torch.stack(maxs, dim=0),
        "mean": torch.stack(means, dim=0),
        "var": torch.stack(vars_, dim=0),
    }


def load_lerobot_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lerobot_episodes(root: Path) -> dict:
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes = {}
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                episodes[int(item["episode_index"])] = item
        return dict(sorted(episodes.items()))

    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.exists() and episodes_dir.is_dir():
        import pyarrow.parquet as pq

        wanted_columns = {
            "episode_index",
            "data_chunk_index",
            "data_file_index",
            "data/chunk_index",
            "data/file_index",
        }
        episodes = {}
        for parquet_file in sorted(episodes_dir.rglob("*.parquet")):
            schema = pq.read_schema(parquet_file)
            columns = [name for name in schema.names if name in wanted_columns]
            table = pq.read_table(parquet_file, columns=columns)
            arrays = {name: table[name] for name in table.column_names}
            episode_indices = arrays["episode_index"].to_pylist()
            for row_idx, episode_index in enumerate(episode_indices):
                item = {
                    name: arrays[name][row_idx].as_py()
                    for name in arrays
                }
                episodes[int(episode_index)] = item
        return dict(sorted(episodes.items()))

    raise FileNotFoundError(f"Episodes metadata not found under {root / 'meta'}")


class LightweightLeRobotMetadata:
    """Only load metadata needed for locating parquet files and counting episodes."""

    def __init__(self, repo_id: str, root: Path):
        import packaging.version

        self.repo_id = repo_id
        self.root = root
        self.info = load_lerobot_info(root)
        self.episodes = load_lerobot_episodes(root)
        self._version = packaging.version.parse(self.info["codebase_version"])
        self.is_v3 = self._version >= packaging.version.parse("v3.0")

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def chunks_size(self) -> int:
        return self.info["chunks_size"]

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    def get_data_file_path(self, ep_index: int) -> Path:
        if self.is_v3 and ep_index in self.episodes:
            ep_dict = self.episodes[ep_index]
            chunk_index = ep_dict.get("data_chunk_index", ep_dict.get("data/chunk_index", 0))
            file_index = ep_dict.get("data_file_index", ep_dict.get("data/file_index", 0))
            return Path(self.data_path.format(chunk_index=chunk_index, file_index=file_index))

        ep_chunk = self.get_episode_chunk(ep_index)
        return Path(self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index))


class StatsOnlyDataset:
    """
    Minimal stats-only dataset reader.

    It avoids constructing LeRobotDataset / HF Dataset and reads only the
    state/action parquet columns needed for normalization statistics.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: dict,
        action_size: int,
        global_sample_stride: int,
        profiler: ProfileStats | None = None,
    ):
        self.profiler = profiler
        self.dataset_dirs = expand_dataset_dirs(dataset_dirs)
        self.shape_meta = shape_meta
        self.action_size = action_size

        metas = []
        for ds_dir in self.dataset_dirs:
            ds_root = Path(ds_dir)
            metas.append(LightweightLeRobotMetadata(repo_id=ds_dir, root=ds_root))

        fps_list = [meta.fps for meta in metas]
        if len(set(fps_list)) != 1:
            raise AssertionError(f"All dataset_dirs must have the same fps, got {fps_list}")
        self.fps = fps_list[0]

        self.state_meta, self.action_meta = add_lerobot_keys(
            shape_meta,
            fps=self.fps,
            global_sample_stride=global_sample_stride,
        )

        self.episodes = []
        self.file_episodes = []
        total_frames = 0
        for ds_dir, meta in zip(self.dataset_dirs, metas, strict=True):
            ds_root = Path(ds_dir)
            total_frames += int(meta.total_frames)
            file_to_episodes = {}
            for episode_idx in range(meta.total_episodes):
                file_path = ds_root / meta.get_data_file_path(episode_idx)
                self.episodes.append((ds_root, meta, episode_idx))
                file_to_episodes.setdefault(file_path, []).append(episode_idx)
            self.file_episodes.extend(
                (ds_root, meta, file_path, episode_indices)
                for file_path, episode_indices in file_to_episodes.items()
            )

        self.multi_dataset = SimpleNamespace(
            num_episodes=len(self.episodes),
            num_frames=total_frames,
        )

    def _get_action(self, meta, parquet_sample):
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action = parquet_sample[lerobot_key]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, parquet_sample):
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state = parquet_sample[lerobot_key]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state

    def _read_episode_columns(self, episode_idx: int):
        ds_root, meta, local_episode_idx = self.episodes[episode_idx]
        file_path = ds_root / meta.get_data_file_path(local_episode_idx)
        columns = [meta["lerobot_key"] for meta in self.state_meta + self.action_meta]
        return self._read_file_columns(file_path, columns)

    def _read_file_columns(self, file_path: Path, columns: list[str]):
        import pyarrow.parquet as pq

        start = time.perf_counter()
        table = pq.read_table(str(file_path), columns=columns)
        if self.profiler is not None:
            self.profiler.add("parquet_read", time.perf_counter() - start)

        start = time.perf_counter()
        result = {
            col_name: parquet_column_to_tensor(table[col_name])
            for col_name in table.column_names
        }
        if self.profiler is not None:
            self.profiler.add("arrow_to_tensor", time.perf_counter() - start)
        return result

    def _get_episode_data(self, episode_idx: int):
        parquet_sample = self._read_episode_columns(episode_idx)
        start = time.perf_counter()
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, parquet_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, parquet_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        if self.profiler is not None:
            self.profiler.add("state_action_window", time.perf_counter() - start)
        return {"action": action, "state": state}


def optimized_get_dataset_stats(
    dataset,
    preprocessor,
    num_workers: int = 0,
    compute_quantile: bool = True,
    profiler: ProfileStats | None = None,
    profile_interval: float = 30.0,
    wall_start: float | None = None,
):
    """
    完整复现原版 episode-level 聚合语义，但不再将中间量落盘。
    """
    episodes_num = dataset.multi_dataset.num_episodes
    aggregates = {"state": defaultdict(dict), "action": defaultdict(dict)}
    fast_no_transform_stats = (
        not compute_quantile
        and getattr(preprocessor, "action_state_transforms", None) is None
    )

    def fast_stats_from_parquet_sample(parquet_sample: dict):
        result = {"state": {}, "action": {}}

        for meta in dataset.state_meta:
            state = dataset._get_state(meta, parquet_sample).float()
            result["state"][meta["key"]] = state_tensor_stats(state)

        for meta in dataset.action_meta:
            action = dataset._get_action(meta, parquet_sample).float()
            result["action"][meta["key"]] = action_window_stats(action, dataset.action_size)

        return result

    def process_episode_stats(episode_idx: int):
        if fast_no_transform_stats:
            return process_episode_stats_fast(episode_idx)

        start = time.perf_counter()
        batch = dataset._get_episode_data(episode_idx)  # only state/action, no images
        if profiler is not None:
            profiler.add("episode_get_data_total", time.perf_counter() - start)

        start = time.perf_counter()
        batch = preprocessor.action_state_transform(batch)
        if profiler is not None:
            profiler.add("processor_transform", time.perf_counter() - start)

        start = time.perf_counter()
        result = {"state": {}, "action": {}}

        for meta in dataset.state_meta:
            key = meta["key"]
            cur_state: torch.Tensor = batch["state"][key]  # (B, T, D)

            cur_stats = {
                "min": cur_state.amin(0),
                "max": cur_state.amax(0),
                "mean": cur_state.mean(0),
                "var": cur_state.var(0),
            }
            if compute_quantile:
                cur_stats["q01"] = torch.quantile(cur_state, 0.01, dim=0, keepdim=False)
                cur_stats["q99"] = torch.quantile(cur_state, 0.99, dim=0, keepdim=False)
            result["state"][key] = cur_stats

        for meta in dataset.action_meta:
            key = meta["key"]
            cur_action: torch.Tensor = batch["action"][key]  # (B, T, D)

            cur_stats = {
                "min": cur_action.amin(0),
                "max": cur_action.amax(0),
                "mean": cur_action.mean(0),
                "var": cur_action.var(0),
            }
            if compute_quantile:
                cur_stats["q01"] = torch.quantile(cur_action, 0.01, dim=0, keepdim=False)
                cur_stats["q99"] = torch.quantile(cur_action, 0.99, dim=0, keepdim=False)
            result["action"][key] = cur_stats

        if profiler is not None:
            profiler.add("stats_reduction", time.perf_counter() - start)
        return result

    def process_episode_stats_fast(episode_idx: int):
        start = time.perf_counter()
        parquet_sample = dataset._read_episode_columns(episode_idx)
        if profiler is not None:
            profiler.add("episode_get_data_total", time.perf_counter() - start)

        start = time.perf_counter()
        result = fast_stats_from_parquet_sample(parquet_sample)
        if profiler is not None:
            profiler.add("fast_stats_reduction", time.perf_counter() - start)
        return result

    def process_file_stats_fast(file_record):
        _, _, file_path, local_episode_indices = file_record
        data_columns = [meta["lerobot_key"] for meta in dataset.state_meta + dataset.action_meta]
        columns = ["episode_index", *data_columns]

        start = time.perf_counter()
        try:
            file_sample = dataset._read_file_columns(file_path, columns)
        except Exception:
            file_sample = dataset._read_file_columns(file_path, data_columns)
        if profiler is not None:
            profiler.add("file_get_data_total", time.perf_counter() - start)

        episode_index = file_sample.get("episode_index")
        if episode_index is not None:
            episode_index = episode_index.reshape(-1)

        results = []
        for local_episode_idx in local_episode_indices:
            if episode_index is None:
                parquet_sample = {key: file_sample[key] for key in data_columns}
            else:
                mask = episode_index == local_episode_idx
                if not bool(mask.any()):
                    continue
                parquet_sample = {key: file_sample[key][mask] for key in data_columns}

            start = time.perf_counter()
            results.append(fast_stats_from_parquet_sample(parquet_sample))
            if profiler is not None:
                profiler.add("fast_stats_reduction", time.perf_counter() - start)

        return results

    def update_aggregate(kind: str, key: str, cur_stats: dict):
        record = aggregates[kind][key]
        mean = cur_stats["mean"]
        var = cur_stats["var"]

        if not record:
            record["n"] = 0
            record["min"] = cur_stats["min"].clone()
            record["max"] = cur_stats["max"].clone()
            if compute_quantile:
                record["q01"] = cur_stats["q01"].clone()
                record["q99"] = cur_stats["q99"].clone()
            record["mean_sum"] = torch.zeros_like(mean)
            record["mean_sq_sum"] = torch.zeros_like(mean)
            record["var_sum"] = torch.zeros_like(var)
        else:
            record["min"] = torch.minimum(record["min"], cur_stats["min"])
            record["max"] = torch.maximum(record["max"], cur_stats["max"])
            if compute_quantile:
                record["q01"] = torch.minimum(record["q01"], cur_stats["q01"])
                record["q99"] = torch.maximum(record["q99"], cur_stats["q99"])

        record["mean_sum"] += mean
        record["mean_sq_sum"] += mean.square()
        record["var_sum"] += var
        record["n"] += 1

    def update_aggregates(cur_episode_stats: dict):
        for kind in ("state", "action"):
            for key, cur_stats in cur_episode_stats[kind].items():
                update_aggregate(kind, key, cur_stats)

    # -------- Pass 1: per-episode stats --------
    if fast_no_transform_stats and getattr(dataset, "file_episodes", None):
        file_records = dataset.file_episodes
        completed_episodes = 0
        if num_workers <= 0:
            iterator = (
                process_file_stats_fast(file_record)
                for file_record in tqdm(file_records, desc="Iterating parquet files to get normalization")
            )
            for file_stats in iterator:
                start = time.perf_counter()
                for cur_episode_stats in file_stats:
                    update_aggregates(cur_episode_stats)
                completed_episodes += len(file_stats)
                if profiler is not None:
                    profiler.add("aggregate_update", time.perf_counter() - start)
                    profiler.maybe_print_summary(
                        time.perf_counter() - (wall_start or start),
                        profile_interval,
                        title=(
                            f"Live timing summary "
                            f"({completed_episodes}/{episodes_num} episodes, {len(file_stats)} latest-file episodes)"
                        ),
                    )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_file_stats_fast, file_record): file_record
                    for file_record in file_records
                }
                completed_files = 0
                for future in tqdm(as_completed(futures), total=len(file_records), desc="Iterating parquet files to get normalization"):
                    file_stats = future.result()
                    start = time.perf_counter()
                    for cur_episode_stats in file_stats:
                        update_aggregates(cur_episode_stats)
                    completed_files += 1
                    completed_episodes += len(file_stats)
                    if profiler is not None:
                        profiler.add("aggregate_update", time.perf_counter() - start)
                        profiler.maybe_print_summary(
                            time.perf_counter() - (wall_start or start),
                            profile_interval,
                            title=(
                                f"Live timing summary "
                                f"({completed_episodes}/{episodes_num} episodes, {completed_files}/{len(file_records)} files)"
                            ),
                        )
    elif num_workers <= 0:
        for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
            cur_episode_stats = process_episode_stats(episode_idx)
            start = time.perf_counter()
            update_aggregates(cur_episode_stats)
            if profiler is not None:
                profiler.add("aggregate_update", time.perf_counter() - start)
                profiler.maybe_print_summary(
                    time.perf_counter() - (wall_start or start),
                    profile_interval,
                    title=f"Live timing summary ({episode_idx + 1}/{episodes_num} episodes)",
                )
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_episode_stats, episode_idx): episode_idx
                for episode_idx in range(episodes_num)
            }
            completed = 0
            for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                cur_episode_stats = future.result()
                start = time.perf_counter()
                update_aggregates(cur_episode_stats)
                completed += 1
                if profiler is not None:
                    profiler.add("aggregate_update", time.perf_counter() - start)
                    profiler.maybe_print_summary(
                        time.perf_counter() - (wall_start or start),
                        profile_interval,
                        title=f"Live timing summary ({completed}/{episodes_num} episodes)",
                    )

    # -------- Pass 2: aggregate exactly like original --------
    finalize_start = time.perf_counter()
    stats = {
        "state": defaultdict(dict),
        "action": defaultdict(dict),
        "num_episodes": episodes_num,
        "num_transition": dataset.multi_dataset.num_frames,
    }

    def finalize_record(record: dict):
        n = record["n"]
        mean_sum = record["mean_sum"]
        mean_sq_sum = record["mean_sq_sum"]
        var_sum = record["var_sum"]
        time_steps = mean_sum.shape[0]

        stepwise_mean = mean_sum / n
        stepwise_var = (var_sum + mean_sq_sum) / n - stepwise_mean.square()
        global_mean = mean_sum.sum(0) / (n * time_steps)
        global_var = (var_sum.sum(0) + mean_sq_sum.sum(0)) / (n * time_steps) - global_mean.square()

        stepwise_std = stepwise_var.clamp_min(0).sqrt()
        global_std = global_var.clamp_min(0).sqrt()

        stepwise_q01 = record["q01"] if compute_quantile else record["min"]
        stepwise_q99 = record["q99"] if compute_quantile else record["max"]

        return {
            "stepwise_min": record["min"],
            "stepwise_max": record["max"],
            "global_min": record["min"].amin(0),
            "global_max": record["max"].amax(0),
            "stepwise_q01": stepwise_q01,
            "stepwise_q99": stepwise_q99,
            "global_q01": stepwise_q01.amin(0),
            "global_q99": stepwise_q99.amax(0),
            "stepwise_mean": stepwise_mean,
            "stepwise_std": stepwise_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }

    for meta in dataset.state_meta:
        key = meta["key"]
        stats["state"][key].update(finalize_record(aggregates["state"][key]))

    for meta in dataset.action_meta:
        key = meta["key"]
        stats["action"][key].update(finalize_record(aggregates["action"][key]))

    if profiler is not None:
        profiler.add("finalize_stats", time.perf_counter() - finalize_start)

    return stats


def resolve_output_path(args) -> str:
    return str(args.output)


def main(argv=None):
    wall_start = time.perf_counter()
    args = parse_args(argv)
    profiler = ProfileStats(enabled=args.profile)
    active_profiler = profiler if args.profile else None

    start = time.perf_counter()
    import_config_deps()
    import_runtime_deps()
    profiler.add("imports", time.perf_counter() - start)

    start = time.perf_counter()
    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    train_cfg = load_stats_config(args)
    profiler.add("load_config", time.perf_counter() - start)

    start = time.perf_counter()
    dataset_dirs = resolve_dataset_dirs(args, train_cfg)
    shape_meta = OmegaConf.to_container(train_cfg.shape_meta, resolve=True)
    num_frames = int(args.num_frames if args.num_frames is not None else train_cfg.get("num_frames", 2))
    obs_size = int(train_cfg.get("obs_size", num_frames))
    action_size = int(train_cfg.get("action_size", obs_size - 1))
    global_sample_stride = int(
        args.global_sample_stride
        if args.global_sample_stride is not None
        else train_cfg.get("global_sample_stride", 1)
    )
    output_path = resolve_output_path(args)
    profiler.add("resolve_args", time.perf_counter() - start)

    print(f"[INFO] dataset_yaml : {args.dataset_yaml}")
    print(f"[INFO] dataset_dirs : {dataset_dirs}")
    print(f"[INFO] output       : {output_path}")
    print(f"[INFO] obs/action   : {obs_size}/{action_size}")
    print(f"[INFO] stride       : {global_sample_stride}")
    print(f"[INFO] num_workers  : {args.num_workers}")
    print(f"[INFO] q01/q99      : {not args.skip_quantile}")

    if args.tmp_dir is not None or args.keep_tmp:
        print("[INFO] tmp_dir/keep_tmp are ignored by the streaming stats path.")
    if args.profile:
        print(f"[INFO] profile      : enabled, live interval={args.profile_interval}s")

    # Build stats-only dataset reader.
    start = time.perf_counter()
    dataset = StatsOnlyDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        action_size=action_size,
        global_sample_stride=global_sample_stride,
        profiler=active_profiler,
    )
    profiler.add("metadata_init", time.perf_counter() - start)
    print(f"[INFO] parquet files : {len(dataset.file_episodes)}")

    # Build processor
    start = time.perf_counter()
    processor_cfg = train_cfg.get("processor", None)
    if processor_cfg is not None:
        processor = instantiate_from_config(processor_cfg)

        if args.processor_mode == "eval" and hasattr(processor, "eval"):
            processor.eval()
        elif args.processor_mode == "train" and hasattr(processor, "train"):
            processor.train()

        print(f"[INFO] Processor set to {args.processor_mode}().")
    else:
        raise RuntimeError("No processor config found.")
    profiler.add("processor_init", time.perf_counter() - start)
    fast_no_transform_stats = args.skip_quantile and getattr(processor, "action_state_transforms", None) is None
    print(f"[INFO] fast_stats   : {fast_no_transform_stats}")
    if fast_no_transform_stats:
        print("[INFO] fast_stats   : reading each parquet file once and grouping rows by episode_index")

    episodes_num = dataset.multi_dataset.num_episodes
    print(f"[INFO] Computing stats over {episodes_num} episodes...")

    try:
        start = time.perf_counter()
        stats = optimized_get_dataset_stats(
            dataset=dataset,
            preprocessor=processor,
            num_workers=args.num_workers,
            compute_quantile=not args.skip_quantile,
            profiler=active_profiler,
            profile_interval=args.profile_interval,
            wall_start=wall_start,
        )
        profiler.add("compute_stats_total", time.perf_counter() - start)

        start = time.perf_counter()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_stats_to_json(stats, str(out))
        profiler.add("save_json", time.perf_counter() - start)

        print(f"[DONE] Stats saved to: {out}")
        print("       Set data.train.pretrained_norm_stats and data.val.pretrained_norm_stats in YAML if you use this stats file.")
    finally:
        profiler.print_summary(time.perf_counter() - wall_start, title="Final timing summary")


if __name__ == "__main__":
    main()
