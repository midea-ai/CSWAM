import ctypes
import gc
import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, default_collate

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.timing import CudaEventTrace
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


_TRAINING_TIMING_KEYS = (
    "step_wall",
    "data_wait",
    "data_wait_step_max",
    "data_worker_batch",
    "data_worker_sample_max",
    "data_lerobot_get",
    "data_video_process",
    "data_jepa_history",
    "data_subtask_padding",
    "data_text_cache",
    "data_finalize",
    "data_collate",
    "data_worker_trim",
    "host_trim",
    "forward_wall",
    "backward_wall",
    "grad_clip_wall",
    "optimizer_wall",
    "zero_grad_wall",
    "forward_cuda",
    "backward_cuda",
    "grad_clip_cuda",
    "optimizer_cuda",
    "zero_grad_cuda",
    "input_h2d",
    "vae_encode",
    "input_finalize",
    "noise_setup",
    "vjepa_history",
    "vjepa_future",
    "jepa_prepare",
    "pre_dit",
    "attention_mask",
    "mot",
    "post_and_loss",
)

_TRAINING_TIMING_GROUPS = (
    ("pipeline", ("step_wall", "data_wait", "data_wait_step_max", "host_trim")),
    (
        "data_worker",
        (
            "data_worker_batch",
            "data_worker_sample_max",
            "data_lerobot_get",
            "data_video_process",
            "data_jepa_history",
            "data_subtask_padding",
            "data_text_cache",
            "data_finalize",
            "data_collate",
            "data_worker_trim",
        ),
    ),
    (
        "phase_wall",
        (
            "forward_wall",
            "backward_wall",
            "grad_clip_wall",
            "optimizer_wall",
            "zero_grad_wall",
        ),
    ),
    (
        "phase_cuda",
        (
            "forward_cuda",
            "backward_cuda",
            "grad_clip_cuda",
            "optimizer_cuda",
            "zero_grad_cuda",
        ),
    ),
    (
        "forward_gpu",
        (
            "input_h2d",
            "vae_encode",
            "input_finalize",
            "noise_setup",
            "vjepa_history",
            "vjepa_future",
            "jepa_prepare",
            "pre_dit",
            "attention_mask",
            "mot",
            "post_and_loss",
        ),
    ),
)


_LIBC = None
_MALLOC_TRIM_UNAVAILABLE = False


class _DataTimingBatch:
    """CPU-only worker timing payload that Accelerate leaves off device."""

    def __init__(self, values: dict[str, object]):
        self.values = values


def _trim_host_memory() -> tuple[int, bool]:
    """Collect Python garbage and return free glibc heap pages to the OS."""
    global _LIBC, _MALLOC_TRIM_UNAVAILABLE

    collected = gc.collect()
    if _MALLOC_TRIM_UNAVAILABLE:
        return collected, False

    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
            _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
            _LIBC.malloc_trim.restype = ctypes.c_int
        except (AttributeError, OSError):
            _MALLOC_TRIM_UNAVAILABLE = True
            return collected, False

    return collected, bool(_LIBC.malloc_trim(0))


class _PeriodicHostMemoryTrimCollator:
    """Collate a batch while optionally profiling and trimming the worker heap."""

    def __init__(self, every: int, *, profile_data_timing: bool = False):
        self.every = int(every)
        self.profile_data_timing = bool(profile_data_timing)
        self.batch_count = 0

    def __call__(self, batch):
        timing_rows = []
        if self.profile_data_timing:
            for sample in batch:
                if isinstance(sample, dict):
                    timing_rows.append(sample.pop("_data_timing", {}))
                else:
                    timing_rows.append({})
        collate_start = time.perf_counter() if self.profile_data_timing else 0.0
        collated = default_collate(batch)
        collate_ms = (
            (time.perf_counter() - collate_start) * 1000.0
            if self.profile_data_timing
            else 0.0
        )
        self.batch_count += 1
        trim_ms = 0.0
        if self.every > 0 and self.batch_count % self.every == 0:
            trim_start = time.perf_counter() if self.profile_data_timing else 0.0
            del batch
            _trim_host_memory()
            if self.profile_data_timing:
                trim_ms = (time.perf_counter() - trim_start) * 1000.0
        if self.profile_data_timing and isinstance(collated, dict):
            timing: dict[str, object] = {}
            timing_keys = set().union(
                *(row.keys() for row in timing_rows if isinstance(row, dict))
            )
            for key in timing_keys:
                timing[key] = [float(row[key]) for row in timing_rows if key in row]
            timing["collate"] = float(collate_ms)
            timing["worker_trim"] = float(trim_ms)
            collated["_data_timing"] = _DataTimingBatch(timing)
        return collated

class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.host_memory_trim_every = int(getattr(cfg, "host_memory_trim_every", 1))
        if self.host_memory_trim_every < 0:
            raise ValueError("`host_memory_trim_every` must be non-negative.")
        self.timing_enabled = bool(getattr(cfg, "timing_enabled", False))
        default_timing_log_every = self.log_every if self.log_every > 0 else 10
        self.timing_log_every = int(
            getattr(cfg, "timing_log_every", default_timing_log_every)
        )
        self.timing_warmup_steps = int(getattr(cfg, "timing_warmup_steps", 5))
        if self.timing_log_every <= 0:
            raise ValueError("`timing_log_every` must be positive.")
        if self.timing_warmup_steps < 0:
            raise ValueError("`timing_warmup_steps` must be non-negative.")
        self._timing_cpu_sums: dict[str, float] = {}
        self._timing_cpu_counts: dict[str, int] = {}
        self._timing_cpu_maxes: dict[str, float] = {}
        self._timing_cuda_traces: list[CudaEventTrace] = []
        self._timing_window_steps = 0
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_num_samples = int(getattr(cfg, "eval_num_samples", 32))
        self.eval_seed = int(getattr(cfg, "eval_seed", 42))
        if self.eval_num_samples <= 0:
            raise ValueError(f"eval_num_samples must be positive, got {self.eval_num_samples}")
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.host_memory_trim_every > 0 and self.accelerator.is_main_process:
            logger.info(
                "Host memory trimming enabled every %d optimizer steps and worker batches.",
                self.host_memory_trim_every,
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        self._preloaded_weight_checkpoint = False
        self._load_weight_checkpoint_before_prepare()

        timing_setter = getattr(self.model, "set_training_timing_enabled", None)
        if callable(timing_setter):
            timing_setter(self.timing_enabled)
        if self.accelerator.is_main_process:
            logger.info(
                "Training timing: enabled=%s log_every=%d warmup_steps=%d; values are rank max/mean milliseconds.",
                self.timing_enabled,
                self.timing_log_every,
                self.timing_warmup_steps,
            )

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        #Timwmask
        warmup_steps = int(total_train_steps * 0.05)
        # warmup_steps = int(total_train_steps * 0.1)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        collate_fn = None
        profile_data_timing = bool(getattr(dataset, "profile_data_timing", False))
        if self.host_memory_trim_every > 0 or profile_data_timing:
            collate_fn = _PeriodicHostMemoryTrimCollator(
                self.host_memory_trim_every,
                profile_data_timing=profile_data_timing,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _measure_log_window_speed(self):
        now = time.perf_counter()
        elapsed = max(now - self.log_window_start_time, 1e-6)
        done_steps = max(self.global_step - self.log_window_start_step, 1)
        steps_per_sec = done_steps / elapsed
        self.log_window_start_time = now
        self.log_window_start_step = self.global_step
        return steps_per_sec

    def _timing_add_cpu(self, name: str, elapsed_ms: float) -> None:
        if not self.timing_enabled:
            return
        self._timing_cpu_sums[name] = self._timing_cpu_sums.get(name, 0.0) + float(elapsed_ms)
        self._timing_cpu_counts[name] = self._timing_cpu_counts.get(name, 0) + 1

    def _timing_set_cpu_max(self, name: str, elapsed_ms: float) -> None:
        if not self.timing_enabled:
            return
        self._timing_cpu_maxes[name] = max(
            self._timing_cpu_maxes.get(name, float("-inf")),
            float(elapsed_ms),
        )

    def _record_data_timing(self, timing) -> None:
        if isinstance(timing, _DataTimingBatch):
            timing = timing.values
        if not self.timing_enabled or not isinstance(timing, dict):
            return

        def values_as_floats(value) -> list[float]:
            if isinstance(value, (list, tuple)):
                return [float(item) for item in value]
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
            if isinstance(value, (int, float)):
                return [float(value)]
            return []

        phase_keys = {
            "lerobot_get": "data_lerobot_get",
            "video_process": "data_video_process",
            "jepa_history": "data_jepa_history",
            "subtask_padding": "data_subtask_padding",
            "text_cache": "data_text_cache",
            "finalize": "data_finalize",
        }
        for source_key, timing_key in phase_keys.items():
            values = values_as_floats(timing.get(source_key))
            if values:
                self._timing_add_cpu(timing_key, sum(values))

        sample_total = values_as_floats(timing.get("sample_total"))
        if sample_total:
            self._timing_add_cpu("data_worker_batch", sum(sample_total))
            self._timing_set_cpu_max("data_worker_sample_max", max(sample_total))

        for source_key, timing_key in (
            ("collate", "data_collate"),
            ("worker_trim", "data_worker_trim"),
        ):
            values = values_as_floats(timing.get(source_key))
            if len(values) == 1:
                self._timing_add_cpu(timing_key, values[0])

    def _timing_add_cuda_trace(self, trace: CudaEventTrace) -> None:
        if self.timing_enabled and trace.has_intervals:
            self._timing_cuda_traces.append(trace)

    def _reset_timing_window(self, model) -> None:
        self._timing_cpu_sums.clear()
        self._timing_cpu_counts.clear()
        self._timing_cpu_maxes.clear()
        self._timing_cuda_traces.clear()
        self._timing_window_steps = 0
        pop_traces = getattr(model, "pop_training_timing_traces", None)
        if callable(pop_traces):
            pop_traces()

    def _log_timing_window(self, model) -> None:
        if not self.timing_enabled or self._timing_window_steps <= 0:
            return

        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)

        sums = dict(self._timing_cpu_sums)
        counts = dict(self._timing_cpu_counts)
        for key, value in self._timing_cpu_maxes.items():
            sums[key] = value
            counts[key] = 1
        traces = self._timing_cuda_traces
        pop_traces = getattr(model, "pop_training_timing_traces", None)
        if callable(pop_traces):
            traces = [*traces, *pop_traces()]
        for trace in traces:
            for name, elapsed_ms in trace.elapsed_ms().items():
                sums[name] = sums.get(name, 0.0) + elapsed_ms
                counts[name] = counts.get(name, 0) + 1

        local_values = [
            sums[key] / counts[key] if counts.get(key, 0) > 0 else float("nan")
            for key in _TRAINING_TIMING_KEYS
        ]
        local_tensor = torch.tensor(
            local_values,
            device=self.accelerator.device,
            dtype=torch.float64,
        ).reshape(1, -1)
        gathered = self.accelerator.gather(local_tensor).reshape(
            self.accelerator.num_processes, -1
        )
        valid = ~torch.isnan(gathered)
        valid_count = valid.sum(dim=0).clamp(min=1)
        rank_mean = torch.where(valid, gathered, torch.zeros_like(gathered)).sum(dim=0) / valid_count
        rank_max = torch.where(
            valid,
            gathered,
            torch.full_like(gathered, -torch.inf),
        ).max(dim=0).values

        if self.accelerator.is_main_process:
            stats = {
                key: (float(rank_max[index].item()), float(rank_mean[index].item()))
                for index, key in enumerate(_TRAINING_TIMING_KEYS)
                if bool(valid[:, index].any().item())
            }
            logger.info(
                "[timing] step=%d window_steps=%d unit=ms values=rank_max/rank_mean",
                self.global_step,
                self._timing_window_steps,
            )
            for group_name, group_keys in _TRAINING_TIMING_GROUPS:
                fields = [
                    f"{key}={stats[key][0]:.1f}/{stats[key][1]:.1f}"
                    for key in group_keys
                    if key in stats
                ]
                if fields:
                    logger.info("[timing] %s %s", group_name, " ".join(fields))

        self._reset_timing_window(model)

    def _load_weight_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

        logger.info("Loading weight checkpoint before DeepSpeed/Accelerate prepare: %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._preloaded_weight_checkpoint = True
        logger.warning(
            "Loaded .pt weights before DeepSpeed/Accelerate prepare; "
            "optimizer/scheduler/step were not restored."
        )

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if getattr(self, "_preloaded_weight_checkpoint", False):
            logger.info("Weight checkpoint already loaded before prepare; skipping post-prepare weight load.")
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        jepa_history_video = sample.get("jepa_history_video", None)
        image_is_pad = sample.get("image_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        proprio_is_pad = sample.get("proprio_is_pad", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        if jepa_history_video is not None:
            if not isinstance(jepa_history_video, torch.Tensor):
                raise TypeError("`sample['jepa_history_video']` must be a torch.Tensor.")
            if jepa_history_video.ndim == 4:
                jepa_history_video = jepa_history_video.unsqueeze(0)
            if jepa_history_video.ndim != 5:
                raise ValueError(
                    "`sample['jepa_history_video']` must be [3,T,H,W] or [B,3,T,H,W], "
                    f"got {tuple(jepa_history_video.shape)}"
                )
            if jepa_history_video.shape[0] != video.shape[0]:
                raise ValueError(
                    "JEPA history/video batch mismatch: "
                    f"history={jepa_history_video.shape[0]} vs video={video.shape[0]}"
                )

        pad_tensors = {
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        for name, pad in pad_tensors.items():
            if pad is None:
                continue
            if not isinstance(pad, torch.Tensor):
                raise TypeError(f"`sample[{name!r}]` must be a torch.Tensor.")
            if pad.ndim == 1:
                pad = pad.unsqueeze(0)
            if pad.ndim != 2 or pad.shape[0] != video.shape[0]:
                raise ValueError(
                    f"`sample[{name!r}]` must be [T] or [B,T], got {tuple(pad.shape)}"
                )
            pad_tensors[name] = pad

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "jepa_history_video": jepa_history_video,
            "image_is_pad": pad_tensors["image_is_pad"],
            "action_is_pad": pad_tensors["action_is_pad"],
            "action_dim_is_pad": pad_tensors["action_dim_is_pad"],
            "proprio_is_pad": pad_tensors["proprio_is_pad"],
            "action_horizon": action_horizon,
        }

    def _compute_eval_action_metric_dict(self, sample, pred_action, gt_action):
        """Return padding-aware denormalized action errors and per-dimension summaries."""
        if gt_action is None or pred_action is None:
            return None
        if sample["proprio"] is None:
            raise ValueError("Eval sample must contain `proprio` for action denormalization.")

        proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
        processor = self.val_dataset.lerobot_dataset.processor
        denorm_actions = {}
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        for action_name, raw_action in (("pred", pred_action), ("gt", gt_action)):
            if not isinstance(raw_action, torch.Tensor):
                raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
            if raw_action.ndim == 2:
                action_btd = raw_action.unsqueeze(0)
            elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                action_btd = raw_action
            else:
                raise ValueError(
                    f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                )
            action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
            batch = {"action": action_btd, "state": proprio}
            batch = processor.action_state_merger.backward(batch)
            batch = processor.normalizer.backward(batch)
            merged_batch = {
                "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
            }
            merged_batch = processor.action_state_merger.forward(merged_batch)
            denorm_action = merged_batch["action"].unsqueeze(0)
            if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                raise ValueError(
                    f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                )
            denorm_actions[action_name] = denorm_action

        if denorm_actions["pred"].shape != denorm_actions["gt"].shape:
            raise ValueError(
                "Predicted action/GT action shape mismatch after denormalization: "
                f"pred={tuple(denorm_actions['pred'].shape)} vs gt={tuple(denorm_actions['gt'].shape)}"
            )
        action_diff = denorm_actions["pred"] - denorm_actions["gt"]
        valid_steps = torch.ones(
            action_diff.shape[:2], dtype=torch.bool, device=action_diff.device
        )
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.detach().to(device=action_diff.device, dtype=torch.bool)
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            if action_is_pad.shape != valid_steps.shape:
                raise ValueError(
                    "action_is_pad shape mismatch for eval metrics: "
                    f"mask={tuple(action_is_pad.shape)} actions={tuple(action_diff.shape)}"
                )
            valid_steps = ~action_is_pad

        valid_dims = torch.ones(
            (action_diff.shape[0], action_diff.shape[2]),
            dtype=torch.bool,
            device=action_diff.device,
        )
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.detach().to(
                device=action_diff.device, dtype=torch.bool
            )
            if action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0)
            if action_dim_is_pad.shape != valid_dims.shape:
                raise ValueError(
                    "action_dim_is_pad shape mismatch for eval metrics: "
                    f"mask={tuple(action_dim_is_pad.shape)} actions={tuple(action_diff.shape)}"
                )
            valid_dims = ~action_dim_is_pad

        valid = valid_steps.unsqueeze(-1) & valid_dims.unsqueeze(1)
        valid_float = valid.to(dtype=action_diff.dtype)
        abs_diff = action_diff.abs()
        sq_diff = action_diff.pow(2)
        per_dim_count = valid_float.sum(dim=(0, 1)).clamp(min=1.0)
        per_dim_l1 = (abs_diff * valid_float).sum(dim=(0, 1)) / per_dim_count
        per_dim_l2 = (sq_diff * valid_float).sum(dim=(0, 1)) / per_dim_count
        valid_count = valid_float.sum().clamp(min=1.0)
        action_l1 = (abs_diff * valid_float).sum() / valid_count
        action_l2 = (sq_diff * valid_float).sum() / valid_count
        return {
            "action_l1": float(action_l1.item()),
            "action_l2": float(action_l2.item()),
            "action_rmse": float(action_l2.sqrt().item()),
            "action_l1_per_dim": per_dim_l1,
            "action_l2_per_dim": per_dim_l2,
        }

    def _compute_eval_action_metrics(self, sample, pred_action, gt_action):
        metrics = self._compute_eval_action_metric_dict(sample, pred_action, gt_action)
        if metrics is None:
            return None, None
        return metrics["action_l1"], metrics["action_l2"]

    @staticmethod
    def _get_unweighted_eval_losses(model, loss_details):
        """Return comparable, unweighted validation losses for each expert.

        ``training_loss`` reports branch losses after applying their objective
        weights so that the values can be logged during training.  Validation
        reports the raw branch objectives as ``val_loss_*`` and keeps the
        weighted sum as ``val_loss``.
        """
        if not isinstance(loss_details, dict):
            return {}

        losses = {}
        for branch, weight_attr in (
            ("video", "loss_lambda_video"),
            ("jepa", "loss_lambda_jepa"),
            ("action", "loss_lambda_action"),
        ):
            key = f"loss_{branch}"
            if key not in loss_details:
                continue
            value = float(loss_details[key])
            weight = float(getattr(model, weight_attr, 1.0))
            if abs(weight) > 1e-12:
                value /= weight
            losses[f"val_loss_{branch}"] = value
        return losses

    @torch.no_grad()
    def _evaluate_action_only_fixed(self, model, was_dit_training: bool):
        """Evaluate CSWAM on a fixed validation subset distributed across ranks."""
        num_samples = min(self.eval_num_samples, len(self.val_dataset))
        index_generator = torch.Generator(device="cpu").manual_seed(self.eval_seed)
        eval_indices = torch.randperm(
            len(self.val_dataset), generator=index_generator
        )[:num_samples].tolist()
        local_indices = eval_indices[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]

        action_dim = int(model.action_expert.action_dim)
        # count, four losses, L2, L1, RMSE, per-dim L1, per-dim L2
        metric_size = 8 + 2 * action_dim
        local_sums = torch.zeros(
            metric_size, device=self.accelerator.device, dtype=torch.float64
        )

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for eval_index in local_indices:
                sample_seed = self.eval_seed + int(eval_index)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

                sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
                with self.accelerator.autocast():
                    val_loss, loss_details = model.training_loss(sample)
                branch_losses = self._get_unweighted_eval_losses(model, loss_details)

                prompt = sample["prompt"][0]
                video0 = sample["video"][0]
                action = sample["action"][0] if sample.get("action") is not None else None
                proprio = (
                    sample["proprio"][0, 0]
                    if sample.get("proprio") is not None
                    else None
                )
                infer_kwargs = {
                    "input_image": video0[:, 0].unsqueeze(0),
                    "num_frames": int(video0.shape[1]),
                    "action": action,
                    "action_horizon": sample["action_horizon"],
                    "proprio": proprio,
                    "text_cfg_scale": 1.0,
                    "action_cfg_scale": 1.0,
                    "num_inference_steps": self.eval_num_inference_steps,
                    "seed": sample_seed,
                    "tiled": False,
                }
                if sample["context"] is not None:
                    infer_kwargs["prompt"] = None
                    infer_kwargs["context"] = sample["context"][0]
                    infer_kwargs["context_mask"] = sample["context_mask"][0]
                else:
                    infer_kwargs["prompt"] = prompt
                if sample.get("jepa_history_video") is not None:
                    infer_kwargs["jepa_history_video"] = sample["jepa_history_video"][0]

                pred_action = model.infer(**infer_kwargs).get("action")
                action_metrics = self._compute_eval_action_metric_dict(
                    sample, pred_action, action
                )
                if action_metrics is None:
                    raise RuntimeError("Action-only validation did not produce action metrics.")

                local_sums[0] += 1.0
                local_sums[1] += float(val_loss.float().item())
                local_sums[2] += float(branch_losses.get("val_loss_video", 0.0))
                local_sums[3] += float(branch_losses["val_loss_jepa"])
                local_sums[4] += float(branch_losses["val_loss_action"])
                local_sums[5] += float(action_metrics["action_l2"])
                local_sums[6] += float(action_metrics["action_l1"])
                local_sums[7] += float(action_metrics["action_rmse"])
                local_sums[8 : 8 + action_dim] += action_metrics[
                    "action_l1_per_dim"
                ].to(device=local_sums.device, dtype=local_sums.dtype)
                local_sums[8 + action_dim :] += action_metrics[
                    "action_l2_per_dim"
                ].to(device=local_sums.device, dtype=local_sums.dtype)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            if was_dit_training:
                self._set_dit_only_train_mode()

        global_sums = self.accelerator.reduce(local_sums, reduction="sum")
        count = global_sums[0].clamp(min=1.0)
        means = global_sums[1:] / count
        per_dim_l1 = means[7 : 7 + action_dim]
        per_dim_l2 = means[7 + action_dim :]
        max_l1_dim = int(per_dim_l1.argmax().item())
        max_l2_dim = int(per_dim_l2.argmax().item())

        return {
            "val_loss": float(means[0].item()),
            "val_loss_video": float(means[1].item()),
            "val_loss_jepa": float(means[2].item()),
            "val_loss_action": float(means[3].item()),
            "action_l2": float(means[4].item()),
            "action_l1": float(means[5].item()),
            "action_rmse": float(means[6].item()),
            "action_l1_max": float(per_dim_l1[max_l1_dim].item()),
            "action_l1_max_dim": max_l1_dim,
            "action_l2_max": float(per_dim_l2[max_l2_dim].item()),
            "action_l2_max_dim": max_l2_dim,
            "eval_num_samples": int(global_sums[0].item()),
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        if bool(getattr(model, "action_only_inference", False)):
            return self._evaluate_action_only_fixed(model, was_dit_training)

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, loss_details = model.training_loss(sample)
            val_loss = val_loss.float().item()
        branch_losses = self._get_unweighted_eval_losses(model, loss_details)
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        if sample.get("jepa_history_video") is not None:
            infer_kwargs["jepa_history_video"] = sample["jepa_history_video"][0]

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_action = pred.get("action", None)

        pred_video = pred["video"]

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1, action_l2 = self._compute_eval_action_metrics(sample, pred_action, action)

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(branch_losses.get("val_loss_video", -1.0)),
                float(branch_losses.get("val_loss_jepa", -1.0)),
                float(branch_losses.get("val_loss_action", -1.0)),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics.mean(dim=0)
        action_l2_mean = gathered_metrics[:, 10].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 11].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[4].item()),
            "ssim_rg": float(mean_metrics[5].item()),
            "psnr_rd": float(mean_metrics[6].item()),
            "ssim_rd": float(mean_metrics[7].item()),
            "psnr_dg": float(mean_metrics[8].item()),
            "ssim_dg": float(mean_metrics[9].item()),
            "video_path": video_path,
        }
        for index, name in enumerate(("video", "jepa", "action"), start=1):
            key = f"val_loss_{name}"
            if key in branch_losses:
                result[key] = float(mean_metrics[index].item())
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _cleanup_old_checkpoints(self, max_keep: int = 5):
        """删除最旧的存档，保持 weights 和 state 各不超过 max_keep 个。"""
        # 清理 weights：按文件名排序（step_XXXXXX.pt），删除最旧的
        weight_files = sorted(
            [f for f in os.listdir(self.weights_dir) if f.endswith(".pt")],
        )
        while len(weight_files) > max_keep:
            oldest = os.path.join(self.weights_dir, weight_files.pop(0))
            try:
                os.remove(oldest)
                logger.info("[ckpt] 删除旧权重存档: %s", oldest)
            except OSError as e:
                logger.warning("[ckpt] 删除旧权重存档失败 %s: %s", oldest, e)

        # 清理 state：按文件夹名排序（step_XXXXXX），删除最旧的
        state_dirs = sorted(
            [d for d in os.listdir(self.state_dir)
             if os.path.isdir(os.path.join(self.state_dir, d))],
        )
        while len(state_dirs) > max_keep:
            oldest = os.path.join(self.state_dir, state_dirs.pop(0))
            try:
                import shutil
                shutil.rmtree(oldest)
                logger.info("[ckpt] 删除旧状态存档: %s", oldest)
            except OSError as e:
                logger.warning("[ckpt] 删除旧状态存档失败 %s: %s", oldest, e)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()

        # 先清理旧存档，释放空间，再写新存档，避免存储爆满
        if self.accelerator.is_main_process:
            self._cleanup_old_checkpoints(max_keep=4)
        self.accelerator.wait_for_everyone()

        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        self.log_window_start_step = self.global_step
        self.log_window_start_time = self.run_start_time

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            should_trim_host_memory = False
            should_log_timing = False
            reached_max_steps = False
            iteration_wall_start = time.perf_counter() if self.timing_enabled else 0.0
            data_wait_start = time.perf_counter() if self.timing_enabled else 0.0
            try:
                sample = next(data_iter)
                if self.timing_enabled:
                    data_wait_ms = (time.perf_counter() - data_wait_start) * 1000.0
                    self._timing_add_cpu(
                        "data_wait", data_wait_ms
                    )
                    self._timing_set_cpu_max("data_wait_step_max", data_wait_ms)
                self._record_data_timing(sample.pop("_data_timing", None))
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            training_trace = CudaEventTrace(enabled=self.timing_enabled)
            training_trace_added = False
            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                phase_wall_start = time.perf_counter() if self.timing_enabled else 0.0
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                if self.timing_enabled:
                    self._timing_add_cpu(
                        "forward_wall", (time.perf_counter() - phase_wall_start) * 1000.0
                    )
                training_trace.mark("forward_cuda")
                phase_wall_start = time.perf_counter() if self.timing_enabled else 0.0
                self.accelerator.backward(loss)
                if self.timing_enabled:
                    self._timing_add_cpu(
                        "backward_wall", (time.perf_counter() - phase_wall_start) * 1000.0
                    )
                training_trace.mark("backward_cuda")

                if self.accelerator.sync_gradients:
                    phase_wall_start = time.perf_counter() if self.timing_enabled else 0.0
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.timing_enabled:
                        self._timing_add_cpu(
                            "grad_clip_wall",
                            (time.perf_counter() - phase_wall_start) * 1000.0,
                        )
                    training_trace.mark("grad_clip_cuda")
                    phase_wall_start = time.perf_counter() if self.timing_enabled else 0.0
                    self.optimizer.step()
                    if self.timing_enabled:
                        self._timing_add_cpu(
                            "optimizer_wall",
                            (time.perf_counter() - phase_wall_start) * 1000.0,
                        )
                    training_trace.mark("optimizer_cuda")
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    phase_wall_start = time.perf_counter() if self.timing_enabled else 0.0
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.timing_enabled:
                        self._timing_add_cpu(
                            "zero_grad_wall",
                            (time.perf_counter() - phase_wall_start) * 1000.0,
                        )
                    training_trace.mark("zero_grad_cuda")
                    self._timing_add_cuda_trace(training_trace)
                    training_trace_added = True
                    self.global_step += 1
                    if self.timing_enabled:
                        self._timing_add_cpu(
                            "step_wall",
                            (time.perf_counter() - iteration_wall_start) * 1000.0,
                        )
                        self._timing_window_steps += 1
                        if self.global_step <= self.timing_warmup_steps:
                            self._reset_timing_window(unwrapped_model)
                        else:
                            should_log_timing = (
                                self.global_step % self.timing_log_every == 0
                            )
                    should_trim_host_memory = (
                        self.host_memory_trim_every > 0
                        and self.global_step % self.host_memory_trim_every == 0
                    )
                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    if should_log:
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            if isinstance(value, torch.Tensor):
                                metric_tensor = value.detach().to(device=loss.device, dtype=torch.float32)
                                if metric_tensor.numel() != 1:
                                    raise ValueError(
                                        f"`loss_dict[{key}]` must be a scalar for logging, "
                                        f"got shape {tuple(metric_tensor.shape)}"
                                    )
                                metric_tensor = metric_tensor.reshape(1)
                            else:
                                metric_tensor = torch.tensor(
                                    float(value),
                                    device=loss.device,
                                    dtype=torch.float32,
                                ).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )

                        current_lr = float(self.optimizer.param_groups[0]["lr"])

                        if self.accelerator.is_main_process:
                            eta_str, _ = self._estimate_eta()
                            steps_per_sec = self._measure_log_window_speed()
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                                description += detail_str + " "
                            description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec * self.batch_size * self.accelerator.num_processes,
                                eta_str,
                            )
                            logger.info(description)

                        if should_log_timing:
                            self._log_timing_window(unwrapped_model)
                            should_log_timing = False

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        timing_setter = getattr(
                            unwrapped_model, "set_training_timing_enabled", None
                        )
                        if self.timing_enabled and callable(timing_setter):
                            pop_traces = getattr(
                                unwrapped_model, "pop_training_timing_traces", None
                            )
                            if callable(pop_traces):
                                for trace in pop_traces():
                                    self._timing_add_cuda_trace(trace)
                            timing_setter(False)
                        try:
                            metrics = self.evaluate()
                        finally:
                            if self.timing_enabled and callable(timing_setter):
                                timing_setter(True)
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step, metrics["val_loss"]
                            )
                            for key in (
                                "val_loss_video",
                                "val_loss_jepa",
                                "val_loss_action",
                            ):
                                if key in metrics:
                                    description += " %s=%.4f" % (key, metrics[key])
                            if "psnr_rd" in metrics and "ssim_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"], metrics["ssim_rd"]
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            if "action_rmse" in metrics:
                                description += " action_rmse=%.4f" % metrics["action_rmse"]
                            if "action_l1_max" in metrics:
                                description += " action_l1_max=%.4f(dim=%d)" % (
                                    metrics["action_l1_max"],
                                    metrics["action_l1_max_dim"],
                                )
                            if "action_l2_max" in metrics:
                                description += " action_l2_max=%.4f(dim=%d)" % (
                                    metrics["action_l2_max"],
                                    metrics["action_l2_max_dim"],
                                )
                            if "eval_num_samples" in metrics:
                                description += " eval_samples=%d" % metrics["eval_num_samples"]
                            logger.info(description)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        reached_max_steps = True
                        should_log_timing = self.timing_enabled and self._timing_window_steps > 0

            if not training_trace_added:
                self._timing_add_cuda_trace(training_trace)

            del sample, loss, loss_dict
            if should_trim_host_memory:
                trim_start = time.perf_counter() if self.timing_enabled else 0.0
                _trim_host_memory()
                if self.timing_enabled and self.global_step > self.timing_warmup_steps:
                    self._timing_add_cpu(
                        "host_trim", (time.perf_counter() - trim_start) * 1000.0
                    )
            if should_log_timing:
                self._log_timing_window(unwrapped_model)
            if reached_max_steps:
                return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
