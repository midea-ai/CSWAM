import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
logger = get_logger(__name__)


# DEFAULT_PROMPT = "A video recorded from a robot which {task}"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class InvalidSegmentPromptError(RuntimeError):
    pass


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs=None,
        shape_meta=None,
        dataset_w_subtask_dirs=None,
        dataset_wo_subtask_dirs=None,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        norm_stats_source: str = "pretrained",
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        video_tolerance_s: float = 1e-4,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        exclude_segment_prompts: Optional[list[str]] = None,
        load_jepa_video: bool = False,
        jepa_history_offsets: Optional[list[int]] = None,
        jepa_future_offsets: Optional[list[int]] = None,
        allow_repeated_jepa_history_offsets: bool = False,
        jepa_video_size: Optional[list[int]] = None,
        profile_data_timing: bool = False,
    ):
        if shape_meta is None:
            raise ValueError("`shape_meta` must be provided.")
        dataset_dirs = list(dataset_dirs or [])
        dataset_w_subtask_dirs = list(dataset_w_subtask_dirs or [])
        dataset_wo_subtask_dirs = list(dataset_wo_subtask_dirs or [])
        if not dataset_dirs and not dataset_w_subtask_dirs and not dataset_wo_subtask_dirs:
            raise ValueError(
                "At least one of `dataset_dirs`, `dataset_w_subtask_dirs`, "
                "or `dataset_wo_subtask_dirs` must be provided."
            )
        all_dataset_dirs = dataset_dirs + dataset_w_subtask_dirs + dataset_wo_subtask_dirs
        # Backward compatibility: legacy `dataset_dirs` keeps the previous behavior
        # and is treated as segment/subtask annotated data.
        self._raw_subtask_dataset_dirs = dataset_dirs + dataset_w_subtask_dirs
        self._raw_no_subtask_dataset_dirs = dataset_wo_subtask_dirs
        self.dataset_dirs = all_dataset_dirs
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.global_sample_stride = int(global_sample_stride)

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))
        self.load_jepa_video = bool(load_jepa_video)
        self.profile_data_timing = bool(profile_data_timing)
        self.jepa_history_offsets = [
            int(v) for v in (jepa_history_offsets or [-28, -24, -20, -16, -12, -8, -4, 0])
        ]
        self.allow_repeated_jepa_history_offsets = bool(allow_repeated_jepa_history_offsets)
        if not self.jepa_history_offsets:
            raise ValueError("JEPA history offsets must be non-empty.")
        if (
            self.jepa_history_offsets != sorted(self.jepa_history_offsets)
            or (
                not self.allow_repeated_jepa_history_offsets
                and len(self.jepa_history_offsets) != len(set(self.jepa_history_offsets))
            )
            or self.jepa_history_offsets[-1] != 0
            or any(offset > 0 for offset in self.jepa_history_offsets)
        ):
            raise ValueError(
                "JEPA history offsets must be increasing, non-positive, and end at 0; "
                "duplicates additionally require allow_repeated_jepa_history_offsets=true; "
                f"got {self.jepa_history_offsets}."
            )
        # JEPA and Wan share the exact future RGB frames. The dataset only
        # returns history; the model derives future from its Wan video tensor.
        self.jepa_future_offsets = list(self.video_sample_indices[1:])
        if jepa_future_offsets is not None:
            configured_future_offsets = [int(v) for v in jepa_future_offsets]
            if configured_future_offsets != self.jepa_future_offsets:
                raise ValueError(
                    "`jepa_future_offsets` must match Wan future frame offsets exactly: "
                    f"expected {self.jepa_future_offsets}, got {configured_future_offsets}. "
                    "Omit the option to derive it automatically."
                )
        self.jepa_video_size = tuple(int(v) for v in (jepa_video_size or video_size))
        if len(self.jepa_video_size) != 2 or any(v <= 0 for v in self.jepa_video_size):
            raise ValueError(f"`jepa_video_size` must be [H,W], got {self.jepa_video_size}")

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=all_dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_obs_indices=self.video_sample_indices,
            extra_image_obs_indices=(
                self.jepa_history_offsets if self.load_jepa_video else None
            ),
            allow_repeated_extra_image_obs_indices=(
                self.load_jepa_video and self.allow_repeated_jepa_history_offsets
            ),
            video_tolerance_s=video_tolerance_s,
            pretrained_norm_stats=pretrained_norm_stats,
            norm_stats_source=norm_stats_source,
        )
        self.dataset_dirs = list(self.lerobot_dataset.dataset_dirs)
        self.dataset_has_subtask = [
            self._dataset_path_matches(path, self._raw_subtask_dataset_dirs)
            and not self._dataset_path_matches(path, self._raw_no_subtask_dataset_dirs)
            for path in self.dataset_dirs
        ]
        if not self._raw_subtask_dataset_dirs and self._raw_no_subtask_dataset_dirs:
            self.dataset_has_subtask = [False for _ in self.dataset_dirs]

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.exclude_segment_prompts = set(
            self._normalize_prompt(prompt)
            for prompt in (exclude_segment_prompts or [])
            if str(prompt).strip()
        )
        self.valid_indices = None
        self.segment_debug = os.environ.get("FASTWAM_SEGMENT_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.subtask_pad_debug = os.environ.get("FASTWAM_SUBTASK_PAD_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if hasattr(processor, "image_obs_steps"):
                processor.image_obs_steps = len(self.video_sample_indices)
            self.lerobot_dataset.set_processor(processor)
        
        # Load segment prompts from meta/tasks.jsonl
        self._load_segment_prompts()
        self._build_valid_indices()

    @staticmethod
    def _dataset_path_matches(path, roots) -> bool:
        from pathlib import Path

        if not roots:
            return False
        candidate = Path(path).expanduser().resolve()
        for root in roots:
            root_path = Path(root).expanduser().resolve()
            if candidate == root_path or root_path in candidate.parents:
                return True
        return False

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return " ".join(str(prompt).strip().lower().split())

    def _is_excluded_segment_prompt(self, prompt: str) -> bool:
        return self._normalize_prompt(prompt) in self.exclude_segment_prompts

    def _get_segment_prompt_from_ids(self, dataset_idx: int, ep_idx: int, frame_idx: int) -> str:
        seg_config = self.segment_prompts_by_dataset.get((int(dataset_idx), int(ep_idx)))
        if seg_config is None:
            raise InvalidSegmentPromptError(
                f"Missing segment prompt annotation: "
                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}"
            )

        index_list = [int(v) for v in seg_config.get("index_list", [])]
        reasoning_list = list(seg_config.get("reasoning_list", []))
        if not index_list or not reasoning_list:
            raise InvalidSegmentPromptError(
                f"Empty segment prompt annotation: "
                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}, "
                f"index_len={len(index_list)}, reasoning_len={len(reasoning_list)}"
            )

        for i, boundary in enumerate(index_list):
            if int(frame_idx) < int(boundary):
                return reasoning_list[i] if i < len(reasoning_list) else reasoning_list[-1]
        return reasoning_list[-1]

    @staticmethod
    def _as_int(value) -> int:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    def _build_valid_indices(self):
        if not self.exclude_segment_prompts:
            return

        valid_indices = []
        excluded_count = 0
        missing_prompt_count = 0
        global_start_idx = 0

        for dataset_idx, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            episode_indices = dataset.hf_dataset["episode_index"]
            frame_indices = dataset.hf_dataset["frame_index"]

            for local_idx, (ep_idx_raw, frame_idx_raw) in enumerate(zip(episode_indices, frame_indices)):
                ep_idx = self._as_int(ep_idx_raw)
                frame_idx = self._as_int(frame_idx_raw)
                global_idx = global_start_idx + local_idx

                try:
                    if self.dataset_has_subtask[dataset_idx]:
                        prompt = self._get_segment_prompt_from_ids(dataset_idx, ep_idx, frame_idx)
                    else:
                        row = dataset.hf_dataset[local_idx]
                        prompt = row.get("task") or row.get("instruction")
                        if prompt is None:
                            raise InvalidSegmentPromptError(
                                f"Legacy dataset sample has no task/instruction fallback: "
                                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}"
                            )
                except InvalidSegmentPromptError:
                    missing_prompt_count += 1
                    valid_indices.append(global_idx)
                    continue

                if self._is_excluded_segment_prompt(prompt):
                    excluded_count += 1
                    continue
                valid_indices.append(global_idx)

            global_start_idx += len(dataset)

        if not valid_indices:
            raise ValueError(
                "exclude_segment_prompts filtered out every sample. "
                f"Excluded prompts={sorted(self.exclude_segment_prompts)}"
            )

        self.valid_indices = valid_indices
        logger.info(
            "[RobotVideoDataset] exclude_segment_prompts active: kept %d/%d samples, "
            "excluded %d, missing prompt annotations kept %d",
            len(self.valid_indices),
            len(self.lerobot_dataset),
            excluded_count,
            missing_prompt_count,
        )

    def _resolve_sample_index(self, idx: int) -> int:
        idx = self._as_int(idx)
        if self.valid_indices is None:
            return idx
        if idx < 0 or idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of bounds {len(self.valid_indices)}.")
        return int(self.valid_indices[idx])

    def _random_sample_index(self) -> int:
        if self.valid_indices is not None:
            return int(self.valid_indices[np.random.randint(len(self.valid_indices))])
        return int(np.random.randint(len(self.lerobot_dataset)))
    
    def _load_segment_prompts(self):
        """
        Load segment prompts from meta/tasks.jsonl in each dataset.
        Format (top-level key is episode_index as string): 
        {
          "0": {
            "index_list": [250, 400, 500, 650, 964],
            "reasoning_list": ["prompt_A", "prompt_A", "prompt_B", ...]
          }
        }
        """
        import json
        from pathlib import Path
        
        self.segment_prompts = {}  # ep_idx -> segment config
        self.segment_prompts_by_dataset = {}  # (dataset_index, ep_idx) -> segment config
        total_eps_with_prompts = 0
        
        for dataset_index, ds_dir in enumerate(self.dataset_dirs):
            if not self.dataset_has_subtask[dataset_index]:
                logger.info(
                    "[RobotVideoDataset] Dataset %s uses legacy task prompts; skipping segment prompt load.",
                    ds_dir,
                )
                continue
            tasks_file = Path(ds_dir) / "meta" / "tasks.jsonl"
            logger.info(f"[RobotVideoDataset] Checking tasks file: {tasks_file}")
            if tasks_file.exists():
                logger.info(f"[RobotVideoDataset] Found tasks file: {tasks_file}")
                with open(tasks_file, 'r') as f:
                    for line_no, line in enumerate(f):
                        if line.strip():
                            try:
                                outer_record = json.loads(line)
                                # Format: {"0": {"index_list": [...], "reasoning_list": [...]}, "1": {...}}
                                # or {"0": {"index_list": [...], "reasoning_list": [...]}} (one episode per line)
                                for ep_idx_str, record in outer_record.items():
                                    if isinstance(record, dict) and "index_list" in record and "reasoning_list" in record:
                                        ep_idx = int(ep_idx_str)
                                        self.segment_prompts[ep_idx] = record
                                        self.segment_prompts_by_dataset[(dataset_index, ep_idx)] = record
                                        total_eps_with_prompts += 1
                                        logger.debug(f"[RobotVideoDataset] Loaded segment prompts for episode {ep_idx}: {len(record['reasoning_list'])} segments")
                            except (json.JSONDecodeError, ValueError, KeyError) as e:
                                logger.warning(f"[RobotVideoDataset] Failed to parse line {line_no}: {e}")
            else:
                logger.warning(f"[RobotVideoDataset] Tasks file not found: {tasks_file}")
        
        logger.info(f"[RobotVideoDataset] Loaded segment prompts for {total_eps_with_prompts} episodes from {len(self.dataset_dirs)} dataset(s)")

    def _dataset_root_for_debug(self, dataset_idx: int):
        try:
            return str(self.lerobot_dataset.dataset_dirs[int(dataset_idx)])
        except Exception:
            pass
        try:
            dataset = self.lerobot_dataset.multi_dataset._datasets[int(dataset_idx)]
            return str(getattr(dataset, "root", "unknown"))
        except Exception:
            return "unknown"

    def _debug_segment_fallback(self, reason: str, dataset_idx, ep_idx, frame_idx, sample):
        if not self.segment_debug:
            return

        from pathlib import Path

        dataset_root = self._dataset_root_for_debug(dataset_idx)
        tasks_file = Path(dataset_root) / "meta" / "tasks.jsonl" if dataset_root != "unknown" else None
        by_dataset_has_key = (dataset_idx, ep_idx) in self.segment_prompts_by_dataset
        ep_only_has_key = ep_idx in self.segment_prompts if ep_idx is not None else False
        instruction = sample.get("instruction", "")
        print(
            "[segment-debug] skip_invalid_segment_sample "
            f"reason={reason} dataset_idx={dataset_idx} ep_idx={ep_idx} frame_idx={frame_idx}\n"
            f"[segment-debug] dataset_root={dataset_root}\n"
            f"[segment-debug] tasks_file={tasks_file} exists={tasks_file.exists() if tasks_file is not None else False}\n"
            f"[segment-debug] has_by_dataset_key={by_dataset_has_key} "
            f"has_ep_only_key={ep_only_has_key} "
            f"loaded_dataset_keys_for_ep={[key for key in self.segment_prompts_by_dataset.keys() if key[1] == ep_idx][:10]}\n"
            f"[segment-debug] fallback_instruction={instruction!r}",
            flush=True,
        )

    def _sample_ids(self, sample):
        ep_idx = sample.get("episode_index")
        frame_idx = sample.get("frame_index")
        dataset_idx = sample.get("dataset_index", 0)

        if hasattr(ep_idx, "item"):
            ep_idx = ep_idx.item()
        if hasattr(frame_idx, "item"):
            frame_idx = frame_idx.item()
        if hasattr(dataset_idx, "item"):
            dataset_idx = dataset_idx.item()

        ep_idx = None if ep_idx is None else int(ep_idx)
        frame_idx = None if frame_idx is None else int(frame_idx)
        dataset_idx = int(dataset_idx)
        return dataset_idx, ep_idx, frame_idx

    def _get_episode_length(self, dataset_idx: int, ep_idx: int) -> int | None:
        try:
            dataset = self.lerobot_dataset.multi_dataset._datasets[dataset_idx]
            if ep_idx in dataset.meta.episodes:
                length = dataset.meta.episodes[ep_idx].get("length")
                if length is not None:
                    return int(length)
            if hasattr(dataset, "episode_data_index"):
                local_ep_idx = self._get_local_episode_idx(dataset, ep_idx)
                start = int(dataset.episode_data_index["from"][local_ep_idx])
                end = int(dataset.episode_data_index["to"][local_ep_idx])
                return max(end - start, 0)
        except Exception as err:
            logger.warning(f"[_get_episode_length] Failed for dataset={dataset_idx}, episode={ep_idx}: {err}")
        return None

    @staticmethod
    def _get_local_episode_idx(dataset, ep_idx: int) -> int:
        if getattr(dataset, "_ep_idx_to_local", None) is not None:
            return int(dataset._ep_idx_to_local[int(ep_idx)])
        return int(ep_idx)

    def _get_segment_info(self, sample):
        """
        Return segment info for the current sample:
          segment_idx, segment_start, segment_end, segment_prompt.

        Segment boundaries come from meta/tasks.jsonl index_list. The first
        segment starts at frame 0. The final segment ends at episode length when
        possible; otherwise it falls back to the last boundary.
        """
        dataset_idx, ep_idx, frame_idx = self._sample_ids(sample)

        if ep_idx is None or frame_idx is None:
            self._debug_segment_fallback("missing_ep_or_frame", dataset_idx, ep_idx, frame_idx, sample)
            raise InvalidSegmentPromptError(
                f"Missing episode/frame index for segment prompt: "
                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}"
            )

        seg_config = self.segment_prompts_by_dataset.get((dataset_idx, ep_idx))
        if seg_config is None:
            if not self.dataset_has_subtask[dataset_idx]:
                legacy_prompt = sample.get("task") or sample.get("instruction")
                if legacy_prompt is None:
                    self._debug_segment_fallback("legacy_task_missing", dataset_idx, ep_idx, frame_idx, sample)
                    raise InvalidSegmentPromptError(
                        f"Legacy dataset sample has no task/instruction fallback: "
                        f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}"
                    )
                episode_len = self._get_episode_length(dataset_idx, ep_idx)
                segment_end = int(episode_len) if episode_len is not None else int(frame_idx) + 1
                return {
                    "segment_idx": 0,
                    "segment_start": 0,
                    "segment_end": max(segment_end, 1),
                    "segment_prompt": str(legacy_prompt),
                    "dataset_idx": dataset_idx,
                    "episode_idx": ep_idx,
                    "frame_idx": frame_idx,
                    "legacy_task_fallback": True,
                }
            self._debug_segment_fallback("seg_config_none", dataset_idx, ep_idx, frame_idx, sample)
            raise InvalidSegmentPromptError(
                f"Missing segment prompt annotation: "
                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}"
            )

        index_list = [int(v) for v in seg_config.get("index_list", [])]
        reasoning_list = list(seg_config.get("reasoning_list", []))
        if not index_list or not reasoning_list:
            self._debug_segment_fallback("empty_index_or_reasoning_list", dataset_idx, ep_idx, frame_idx, sample)
            raise InvalidSegmentPromptError(
                f"Empty segment prompt annotation: "
                f"dataset_idx={dataset_idx}, ep_idx={ep_idx}, frame_idx={frame_idx}, "
                f"index_len={len(index_list)}, reasoning_len={len(reasoning_list)}"
            )

        for i, boundary in enumerate(index_list):
            if frame_idx < int(boundary):
                segment_start = 0 if i == 0 else int(index_list[i - 1])
                segment_end = int(boundary)
                segment_prompt = reasoning_list[i] if i < len(reasoning_list) else reasoning_list[-1]
                return {
                    "segment_idx": i,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "segment_prompt": segment_prompt,
                    "dataset_idx": dataset_idx,
                    "episode_idx": ep_idx,
                    "frame_idx": frame_idx,
                }

        segment_idx = max(len(reasoning_list) - 1, 0)
        segment_start = 0 if segment_idx == 0 else int(index_list[min(segment_idx - 1, len(index_list) - 1)])
        segment_end = int(index_list[min(segment_idx, len(index_list) - 1)])
        if segment_end <= segment_start:
            segment_end = segment_start + 1
        return {
            "segment_idx": segment_idx,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "segment_prompt": reasoning_list[-1],
            "dataset_idx": dataset_idx,
            "episode_idx": ep_idx,
            "frame_idx": frame_idx,
        }
        
    def _get_segment_prompt(self, sample):
        """
        根据当前 sample 的最后一帧 frame_index 找到对应的 prompt。
        """
        return self._get_segment_info(sample)["segment_prompt"]

    @staticmethod
    def _clamp_padded_steps(x: torch.Tensor, pad_mask: torch.Tensor, time_dim: int) -> torch.Tensor:
        if not bool(pad_mask.any().item()):
            return x

        valid_indices = torch.nonzero(~pad_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            fill_idx = 0
        else:
            fill_idx = int(valid_indices[-1].item())

        x = x.clone()
        fill = x.select(time_dim, fill_idx).clone()
        padded_indices = torch.nonzero(pad_mask, as_tuple=False).flatten().tolist()
        for idx in padded_indices:
            x.select(time_dim, int(idx)).copy_(fill)
        return x

    def _subtask_step_pad_mask(self, segment_info: dict, num_steps: int, offsets: list[int]) -> torch.Tensor:
        frame_idx = int(segment_info["frame_idx"])
        segment_end = int(segment_info["segment_end"])
        step_frames = [frame_idx + int(offset) * self.global_sample_stride for offset in offsets[:num_steps]]
        return torch.tensor(
            [frame >= segment_end for frame in step_frames],
            dtype=torch.bool,
        )

    def _apply_subtask_padding(
        self,
        video: torch.Tensor,
        action: torch.Tensor,
        proprio: torch.Tensor,
        image_is_pad: torch.Tensor,
        action_is_pad: torch.Tensor,
        proprio_is_pad: torch.Tensor,
        segment_info: dict,
    ):
        if segment_info.get("legacy_task_fallback", False):
            return video, action, proprio, image_is_pad, action_is_pad, proprio_is_pad

        image_is_pad = image_is_pad.bool().clone()
        action_is_pad = action_is_pad.bool().clone()
        proprio_is_pad = proprio_is_pad.bool().clone()

        video_mask = self._subtask_step_pad_mask(
            segment_info,
            num_steps=video.shape[1],
            offsets=self.video_sample_indices,
        ).to(device=image_is_pad.device)
        action_offsets = list(range(action.shape[0]))
        action_mask = self._subtask_step_pad_mask(
            segment_info,
            num_steps=action.shape[0],
            offsets=action_offsets,
        ).to(device=action_is_pad.device)
        proprio_mask = self._subtask_step_pad_mask(
            segment_info,
            num_steps=proprio.shape[0],
            offsets=action_offsets,
        ).to(device=proprio_is_pad.device)

        if image_is_pad.shape[0] != video_mask.shape[0]:
            raise ValueError(
                f"image_is_pad length mismatch: got {image_is_pad.shape[0]}, "
                f"expected {video_mask.shape[0]}"
            )
        if action_is_pad.shape[0] != action_mask.shape[0]:
            raise ValueError(
                f"action_is_pad length mismatch: got {action_is_pad.shape[0]}, "
                f"expected {action_mask.shape[0]}"
            )
        if proprio_is_pad.shape[0] != proprio_mask.shape[0]:
            proprio_is_pad = proprio_is_pad[:proprio.shape[0]].clone()
            if proprio_is_pad.shape[0] != proprio_mask.shape[0]:
                raise ValueError(
                    f"proprio_is_pad length mismatch: got {proprio_is_pad.shape[0]}, "
                    f"expected {proprio_mask.shape[0]}"
                )

        image_is_pad = image_is_pad | video_mask
        action_is_pad = action_is_pad | action_mask
        proprio_is_pad = proprio_is_pad | proprio_mask

        video = self._clamp_padded_steps(video, video_mask.to(device=video.device), time_dim=1)
        action = self._clamp_padded_steps(action, action_mask.to(device=action.device), time_dim=0)
        proprio = self._clamp_padded_steps(proprio, proprio_mask.to(device=proprio.device), time_dim=0)

        if self.subtask_pad_debug and (
            bool(video_mask.any().item())
            or bool(action_mask.any().item())
            or bool(proprio_mask.any().item())
        ):
            print(
                "[subtask-pad-debug] "
                f"dataset_idx={segment_info['dataset_idx']} "
                f"episode_idx={segment_info['episode_idx']} "
                f"frame_idx={segment_info['frame_idx']} "
                f"segment_start={segment_info['segment_start']} "
                f"segment_end={segment_info['segment_end']} "
                f"video_pad_count={int(video_mask.sum().item())} "
                f"action_pad_count={int(action_mask.sum().item())} "
                f"proprio_pad_count={int(proprio_mask.sum().item())} "
                f"task={segment_info['segment_prompt']!r}",
                flush=True,
            )

        return video, action, proprio, image_is_pad, action_is_pad, proprio_is_pad

    def _prepare_jepa_camera_frames(self, frames: torch.Tensor, meta: dict) -> torch.Tensor:
        """Convert queried camera frames to the per-camera training resolution."""
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(f"Expected camera frames [T,C,H,W], got {tuple(frames.shape)}")
        frames = frames.float()
        if frames.max() > 2.0:
            frames = frames / 255.0
        target_shape = meta.get("shape")
        if target_shape is not None and len(target_shape) == 3:
            frames = transforms_F.resize(
                frames,
                size=[int(target_shape[-2]), int(target_shape[-1])],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        return frames.clamp(0.0, 1.0)

    def _compose_jepa_history(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the JEPA history mosaic from frames decoded with the Wan query."""
        image_metas = list(self.lerobot_dataset.image_meta)
        camera_frames = []
        for meta in image_metas:
            key = str(meta["key"])
            if key not in images:
                raise KeyError(f"Missing decoded camera key {key!r} for JEPA history.")
            camera_frames.append(self._prepare_jepa_camera_frames(images[key], meta))

        if len(camera_frames) != 3:
            raise ValueError(
                "CSWAM currently expects exactly three cameras for the RobotWin mosaic, "
                f"got {len(camera_frames)}"
            )
        cam_top, cam_left, cam_right = camera_frames
        cam_top = transforms_F.resize(
            cam_top,
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_left = transforms_F.resize(
            cam_left,
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_right = transforms_F.resize(
            cam_right,
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        clip = torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
        clip = self.resize_transform(clip)
        clip = self.crop_transform(clip).clamp(0.0, 1.0)
        if tuple(clip.shape[-2:]) != self.jepa_video_size:
            clip = transforms_F.resize(
                clip,
                size=list(self.jepa_video_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        return clip.permute(1, 0, 2, 3).contiguous()

    def __len__(self):
        if self.valid_indices is not None:
            return len(self.valid_indices)
        return len(self.lerobot_dataset)

    def _get(self, idx):
        profile_timing = self.profile_data_timing
        sample_timing_start = time.perf_counter() if profile_timing else 0.0
        lerobot_get_ms = 0.0
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            phase_start = time.perf_counter() if profile_timing else 0.0
            sample = self.lerobot_dataset[sample_idx]
            if profile_timing:
                lerobot_get_ms += (time.perf_counter() - phase_start) * 1000.0
            
            # Debug: log sample keys
            if attempt == 0:
                logger.debug(f"[RobotVideoDataset._get] Sample keys: {list(sample.keys())}")
                logger.debug(f"[RobotVideoDataset._get] episode_index: {sample.get('episode_index')}, frame_index: {sample.get('frame_index')}")

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = self._random_sample_index()

        data_timing = {}
        if profile_timing:
            data_timing["lerobot_get"] = lerobot_get_ms
        phase_start = time.perf_counter() if profile_timing else 0.0
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            T_video, C, H, W = video.shape

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        # 替换这一段：
        # task = sample["instruction"]
        # if self.override_instruction is not None:
        #     task = self.override_instruction
        # instruction = DEFAULT_PROMPT.format(task=task)

        # 改成：
        segment_info = self._get_segment_info(sample)
        task = segment_info["segment_prompt"]
        if self._is_excluded_segment_prompt(task):
            raise InvalidSegmentPromptError(
                f"Excluded segment prompt reached _get: "
                f"dataset_idx={segment_info['dataset_idx']}, "
                f"episode_idx={segment_info['episode_idx']}, "
                f"frame_idx={segment_info['frame_idx']}, "
                f"task={task!r}"
            )

        if profile_timing:
            data_timing["video_process"] = (time.perf_counter() - phase_start) * 1000.0
        phase_start = time.perf_counter() if profile_timing else 0.0

        jepa_history_video = None
        jepa_history_is_pad = None
        if self.load_jepa_video:
            history_images = sample.pop("_extra_images", None)
            history_is_pad = sample.pop("_extra_image_is_pad", None)
            if history_images is None or history_is_pad is None:
                raise KeyError(
                    "Merged JEPA history frames are missing from BaseLerobotDataset."
                )
            jepa_history_video = self._compose_jepa_history(history_images)
            jepa_history_is_pad = history_is_pad.to(dtype=torch.bool)
            if jepa_history_is_pad.shape != (len(self.jepa_history_offsets),):
                raise ValueError(
                    "JEPA history padding shape mismatch: expected "
                    f"{(len(self.jepa_history_offsets),)}, got {tuple(jepa_history_is_pad.shape)}"
                )
        if profile_timing:
            data_timing["jepa_history"] = (time.perf_counter() - phase_start) * 1000.0
        phase_start = time.perf_counter() if profile_timing else 0.0
        video, action, proprio, image_is_pad, action_is_pad, proprio_is_pad = self._apply_subtask_padding(
            video=video,
            action=action,
            proprio=proprio,
            image_is_pad=image_is_pad,
            action_is_pad=sample["action_is_pad"],
            proprio_is_pad=sample["proprio_is_pad"],
            segment_info=segment_info,
        )
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        if profile_timing:
            data_timing["subtask_padding"] = (time.perf_counter() - phase_start) * 1000.0
        phase_start = time.perf_counter() if profile_timing else 0.0

        try:
            context, context_mask = self._get_cached_text_context(instruction)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"{err}\n"
                f"[prompt-debug] dataset_idx={segment_info['dataset_idx']} "
                f"episode_idx={segment_info['episode_idx']} frame_idx={segment_info['frame_idx']} "
                f"segment_idx={segment_info['segment_idx']} "
                f"segment_start={segment_info['segment_start']} segment_end={segment_info['segment_end']}\n"
                f"[prompt-debug] task={task!r}\n"
                f"[prompt-debug] full_instruction={instruction!r}"
            ) from err
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        if profile_timing:
            data_timing["text_cache"] = (time.perf_counter() - phase_start) * 1000.0
        phase_start = time.perf_counter() if profile_timing else 0.0

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "segment_idx": torch.tensor(int(segment_info["segment_idx"]), dtype=torch.long),
            "segment_start": torch.tensor(int(segment_info["segment_start"]), dtype=torch.long),
            "segment_end": torch.tensor(int(segment_info["segment_end"]), dtype=torch.long),
            "dataset_index": torch.tensor(int(segment_info["dataset_idx"]), dtype=torch.long),
            "episode_index": torch.tensor(int(segment_info["episode_idx"]), dtype=torch.long),
            "frame_index": torch.tensor(int(segment_info["frame_idx"]), dtype=torch.long),
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        if self.load_jepa_video:
            data.update(
                {
                    "jepa_history_video": jepa_history_video,
                    "jepa_history_is_pad": jepa_history_is_pad,
                }
            )
        if profile_timing:
            data_timing["finalize"] = (time.perf_counter() - phase_start) * 1000.0
            data_timing["sample_total"] = (
                time.perf_counter() - sample_timing_start
            ) * 1000.0
            data["_data_timing"] = data_timing
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}.\n"
                f"[prompt-cache] hash={hashed}\n"
                f"[prompt-cache] context_len={self.context_len}\n"
                f"[prompt-cache] cache_dir={cache_dir}\n"
                f"[prompt-cache] prompt={prompt!r}\n"
                "Run scripts/precompute_text.sh first for this exact prompt/context_len."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        sample_idx = self._resolve_sample_index(idx)
        max_segment_retry = max(self.max_padding_retry, 10)
        for attempt in range(max_segment_retry + 1):
            try:
                return self._get(sample_idx)
            except InvalidSegmentPromptError as e:
                if self.segment_debug:
                    print(
                        f"[segment-debug] skip sample idx {sample_idx}: {e}",
                        flush=True,
                    )
                if attempt >= max_segment_retry:
                    raise
                sample_idx = self._random_sample_index()
            except Exception as e:
                print(f"Error processing sample idx {sample_idx}: {e}. Returning a random sample instead.")
                # trace back
                print(traceback.format_exc())
                if attempt >= max_segment_retry:
                    raise
                sample_idx = self._random_sample_index()
