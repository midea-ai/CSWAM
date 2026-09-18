"""CSWAM model adapter for split-environment RoboTwin evaluation."""

from __future__ import annotations

import inspect
import logging
import types
from collections import OrderedDict, deque
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


logger = logging.getLogger(__name__)


def _restore_official_conv3d_forward(module: torch.nn.Module) -> None:
    """Undo torch.compile wrappers before moving inference-only Conv3d modules."""
    for child in module.children():
        if isinstance(child, torch.nn.Conv3d):
            child._conv_forward = torch.nn.Conv3d._conv_forward.__get__(
                child, torch.nn.Conv3d
            )
        _restore_official_conv3d_forward(child)


def _patch_cpu_vae(model: torch.nn.Module, vae_device_mode: str) -> None:
    if vae_device_mode == "gpu":
        return
    if vae_device_mode != "cpu":
        raise ValueError(
            f"Unsupported vae_device_mode={vae_device_mode!r}; expected cpu or gpu."
        )

    try:
        output_param = next(model.parameters())
        output_device = output_param.device
        output_dtype = output_param.dtype
    except StopIteration:
        output_device = torch.device(model.device)
        output_dtype = model.torch_dtype

    model.vae = model.vae.to(device="cpu", dtype=torch.float32).eval()
    if output_device.type == "cuda":
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_input_image_latents_cpu(
        self,
        input_image: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                "input_image must be [B,3,H,W], "
                f"got {tuple(input_image.shape)}."
            )

        latents = []
        for image in input_image:
            video = image.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous().unsqueeze(1)
            latent = self.vae.encode(
                [video],
                device="cpu",
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if not isinstance(latent, torch.Tensor):
                raise TypeError(f"VAE encode returned {type(latent)!r}.")
            latents.append(latent.to(device=output_device, dtype=output_dtype))
        return latents[0] if len(latents) == 1 else torch.cat(latents, dim=0)

    model._encode_input_image_latents_tensor = types.MethodType(
        encode_input_image_latents_cpu, model
    )
    logger.info(
        "CSWAM VAE remains on CPU; encoded latents move to %s as %s.",
        output_device,
        output_dtype,
    )


def _is_none_like(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _optional_int(value: Any) -> int | None:
    return None if _is_none_like(value) else int(value)


def _optional_float(value: Any) -> float | None:
    return None if _is_none_like(value) else float(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def _model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed_precision={mixed_precision!r}; expected no, fp16, or bf16."
    )


def _load_training_config(config_path: str) -> DictConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSWAM training config not found: {path}")
    cfg = OmegaConf.load(path)
    if "model" not in cfg or "data" not in cfg:
        raise ValueError(f"Expected model/data sections in CSWAM config: {path}")
    return cfg


def _data_config(cfg: DictConfig) -> DictConfig:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, DictConfig):
        raise ValueError("CSWAM config must contain a data mapping.")
    train_cfg = data_cfg.get("train")
    resolved = train_cfg if isinstance(train_cfg, DictConfig) else data_cfg
    missing = [key for key in ("num_frames", "processor") if key not in resolved]
    if missing:
        raise ValueError(
            f"CSWAM data config is missing {missing}; available={list(resolved.keys())}."
        )
    return resolved


def _evaluation_config(cfg: DictConfig) -> DictConfig:
    value = cfg.get("EVALUATION")
    return value if isinstance(value, DictConfig) else OmegaConf.create({})


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(
        pil_image.resize(size_wh, resample=resampling.BILINEAR), dtype=np.uint8
    )


class CSWAMRpcAdapter:
    """Run CSWAM one control step at a time while preserving temporal state."""

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: str,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int | None,
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        num_video_frames: int,
        text_encoder_device: str = "cpu",
        vae_device_mode: str = "cpu",
        text_cache_size: int = 64,
    ) -> None:
        model_cfg_copy = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
        )
        model_cfg_copy.load_text_encoder = True

        # The full training checkpoint is loaded below. Initialization-only
        # expert checkpoints must not be required on an evaluation machine.
        if "skip_dit_load_from_pretrain" in model_cfg_copy:
            model_cfg_copy.skip_dit_load_from_pretrain = True
        if "action_dit_pretrained_path" in model_cfg_copy:
            model_cfg_copy.action_dit_pretrained_path = None
        if "jepa_dit_pretrained_path" in model_cfg_copy:
            model_cfg_copy.jepa_dit_pretrained_path = None

        self.text_encoder_device = torch.device(text_encoder_device)
        self.text_cache_size = int(text_cache_size)
        if self.text_cache_size <= 0:
            raise ValueError(
                f"text_cache_size must be positive, got {self.text_cache_size}."
            )

        # Instantiate and deserialize on CPU so T5 never contributes to peak GPU
        # memory. The checkpoint does not contain T5 or frozen V-JEPA weights.
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device="cpu")
        self.model.load_checkpoint(checkpoint_path)

        cpu_text_encoder = None
        if self.text_encoder_device.type == "cpu":
            if self.model.text_encoder is None:
                raise ValueError("CSWAM text encoder was not loaded.")
            cpu_text_encoder = self.model.text_encoder
            self.model.text_encoder = None

        self.model = self.model.to(device).eval()
        self.model.device = torch.device(device)
        if cpu_text_encoder is not None:
            self.model.text_encoder = cpu_text_encoder.to(device="cpu").eval()
        try:
            self.model.torch_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model.torch_dtype = model_dtype
        _restore_official_conv3d_forward(self.model)
        _patch_cpu_vae(self.model, vae_device_mode=vae_device_mode)

        if "jepa_history_video" not in inspect.signature(self.model.infer_action).parameters:
            raise TypeError(
                "The configured model does not expose infer_action(..., jepa_history_video=...). "
                "Use the ordinary kit adapter for non-JEPA models."
            )

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        stats = load_dataset_stats_from_json(dataset_stats_path)
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.num_video_frames = int(num_video_frames)
        self._text_context_cache: OrderedDict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()

        offsets = tuple(int(value) for value in self.model.jepa_history_offsets)
        expected_frames = int(self.model.vjepa_history_num_frames)
        allow_repeats = bool(
            getattr(self.model, "allow_repeated_jepa_history_offsets", False)
        )
        if len(offsets) != expected_frames:
            raise ValueError(
                "JEPA history requires one offset per V-JEPA input frame: "
                f"offsets={offsets}, expected_frames={expected_frames}."
            )
        if (
            not offsets
            or tuple(sorted(offsets)) != offsets
            or (not allow_repeats and len(offsets) != len(set(offsets)))
            or offsets[-1] != 0
        ):
            raise ValueError(
                "JEPA history offsets must be increasing and end at zero; duplicates "
                "require allow_repeated_jepa_history_offsets=true; "
                f"got {offsets}."
            )
        if any(offset > 0 for offset in offsets):
            raise ValueError(f"JEPA history offsets must be non-positive, got {offsets}.")

        self.history_offsets = offsets
        self.observation_on_replan_only = all(offset == 0 for offset in offsets)
        nonzero_offsets = [abs(offset) for offset in offsets if offset != 0]
        self.history_stride = gcd(*nonzero_offsets) if nonzero_offsets else 1
        self.history_sample_offsets = tuple(
            offset // self.history_stride for offset in offsets
        )
        self.history_frames: deque[np.ndarray] = deque(
            maxlen=1 - self.history_sample_offsets[0]
        )
        self.pending_actions: deque[np.ndarray] = deque()
        self.control_step = 0

        logger.info(
            "Loaded CSWAM checkpoint=%s stats=%s horizon=%d "
            "history_offsets=%s observation_stride=%d observation_on_replan_only=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.history_offsets,
            self.history_stride,
            self.observation_on_replan_only,
        )

    def rpc_capabilities(self) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "action_mode": "single_step",
            "observation_every_step": False,
            "observation_stride": self.history_stride,
            "observation_on_replan_only": self.observation_on_replan_only,
            "model_family": "cswam",
            "history_offsets": list(self.history_offsets),
            "history_num_frames": len(self.history_offsets),
            "action_horizon": self.action_horizon,
        }

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state entry in processor.shape_meta.")
        state_key = state_meta[0]["key"]
        batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        proprio = batch["state"]
        if proprio.shape[-1] != self.processor.proprio_output_dim:
            raise ValueError(
                "Processed proprio dimension mismatch: "
                f"expected {self.processor.proprio_output_dim}, got {proprio.shape[-1]}."
            )
        return proprio

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one merged action entry in processor.shape_meta.")
        action_key = action_meta[0]["key"]
        if action.shape[-1] != self.processor.action_output_dim:
            raise ValueError(
                "Model action dimension mismatch: "
                f"expected {self.processor.action_output_dim}, got {action.shape[-1]}."
            )
        batch = {"action": action.float().cpu()}
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        return batch["action"][action_key].numpy()

    def _get_text_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        cached = self._text_context_cache.get(prompt)
        if cached is None:
            with torch.no_grad():
                if self.text_encoder_device.type == "cpu":
                    ids, context_mask = self.model.tokenizer(
                        prompt,
                        return_mask=True,
                        add_special_tokens=True,
                    )
                    ids = ids.to(device="cpu")
                    context_mask = context_mask.to(device="cpu", dtype=torch.bool)
                    context = self.model.text_encoder(ids, context_mask)
                    seq_lens = context_mask.gt(0).sum(dim=1).tolist()
                    for index, seq_len in enumerate(seq_lens):
                        context[index, int(seq_len) :] = 0
                    # Keep the original FastWAM text-mask behavior for checkpoint
                    # compatibility; the padded embeddings themselves are zero.
                    context_mask = torch.ones_like(context_mask)
                else:
                    context, context_mask = self.model.encode_prompt(prompt)
            cached = (
                context.detach().to(device="cpu").contiguous(),
                context_mask.detach().to(device="cpu", dtype=torch.bool).contiguous(),
            )
            self._text_context_cache[prompt] = cached
            while len(self._text_context_cache) > self.text_cache_size:
                self._text_context_cache.popitem(last=False)
        else:
            self._text_context_cache.move_to_end(prompt)
        return cached

    @staticmethod
    def _build_mosaic(observation: dict[str, Any]) -> np.ndarray:
        obs = observation["observation"]
        head = _resize_rgb(obs["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        return np.ascontiguousarray(np.concatenate([head, bottom], axis=0))

    def _image_tensor(self, mosaic: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(mosaic).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.model.device, dtype=self.model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _history_tensor(
        self, history_video: np.ndarray | None = None
    ) -> torch.Tensor:
        if history_video is not None:
            clip = np.asarray(history_video, dtype=np.uint8)
            expected_shape = (
                len(self.history_offsets),
                int(self.model.vjepa_encoder.input_size[0]),
                int(self.model.vjepa_encoder.input_size[1]),
                3,
            )
            if clip.shape != expected_shape:
                raise ValueError(
                    "jepa_history_video shape mismatch: "
                    f"expected {expected_shape}, got {clip.shape}."
                )
            clip = np.ascontiguousarray(clip)
        else:
            if not self.history_frames:
                raise RuntimeError("JEPA history is empty.")
            frames = list(self.history_frames)
            latest = len(frames) - 1
            selected = [
                frames[max(0, latest + offset)]
                for offset in self.history_sample_offsets
            ]
            clip = np.stack(selected, axis=0)
        return (
            torch.from_numpy(clip)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .div_(255.0)
        )

    def _predict_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str,
        mosaic: np.ndarray,
        history_video: np.ndarray | None = None,
    ) -> np.ndarray:
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        context, context_mask = self._get_text_context(instruction)
        infer_kwargs: dict[str, Any] = {
            "prompt": None,
            "context": context,
            "context_mask": context_mask,
            "input_image": self._image_tensor(mosaic),
            "action_horizon": self.action_horizon,
            "proprio": self._normalize_state(state),
            "jepa_history_video": self._history_tensor(history_video),
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self.num_video_frames
        with torch.no_grad():
            prediction = self.model.infer_action(**infer_kwargs)
        actions = self._denormalize_action(prediction["action"])[0]
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(f"CSWAM returned invalid actions with shape {actions.shape}")
        return actions

    def predict_action_chunk(
        self, observation: dict[str, Any], instruction: str
    ) -> np.ndarray:
        """Predict one chunk from the compact public RoboTwin RPC payload."""
        if "mosaic" in observation:
            mosaic = np.asarray(observation["mosaic"], dtype=np.uint8)
        else:
            mosaic = self._build_mosaic(observation)
        expected_size = tuple(int(value) for value in self.model.vjepa_encoder.input_size)
        expected_shape = (*expected_size, 3)
        if mosaic.shape != expected_shape:
            raise ValueError(
                f"CSWAM mosaic must be {expected_shape}, got {mosaic.shape}."
            )
        history_video = observation.get("jepa_history_video")
        return self._predict_action_chunk(
            observation=observation,
            instruction=instruction,
            mosaic=np.ascontiguousarray(mosaic),
            history_video=history_video,
        )

    def rpc_step(
        self,
        observation: dict[str, Any] | None,
        instruction: str,
        replan_steps: int,
    ) -> np.ndarray:
        replan_steps = max(1, int(replan_steps))
        if replan_steps % self.history_stride != 0:
            raise ValueError(
                f"replan_steps={replan_steps} must be divisible by the JEPA "
                f"observation stride {self.history_stride}."
            )

        if self.observation_on_replan_only:
            sample_step = not self.pending_actions
        else:
            sample_step = self.control_step % self.history_stride == 0
        if sample_step and observation is None:
            raise ValueError(
                f"CSWAM requires an observation at control step {self.control_step}; "
                f"its observation stride is {self.history_stride}."
            )

        mosaic = None
        if sample_step:
            mosaic = self._build_mosaic(observation)
            self.history_frames.append(mosaic)

        if not self.pending_actions:
            if observation is None or mosaic is None:
                raise ValueError(
                    "CSWAM requires a sampled observation when replanning."
                )
            action_chunk = self._predict_action_chunk(observation, instruction, mosaic)
            execute = min(replan_steps, len(action_chunk))
            self.pending_actions.extend(
                np.asarray(action_chunk[index], dtype=np.float32)
                for index in range(execute)
            )
        if not self.pending_actions:
            raise RuntimeError("CSWAM produced an empty action queue.")
        action = self.pending_actions.popleft()
        self.control_step += 1
        return action

    def rpc_reset(self) -> None:
        self.pending_actions.clear()
        self.history_frames.clear()
        self.control_step = 0


def get_model(args: dict[str, Any]) -> CSWAMRpcAdapter:
    config_path = args.get("sim_cfg_path")
    if _is_none_like(config_path):
        raise ValueError("sim_cfg_path must point to the checkpoint's CSWAM YAML.")
    cfg = _load_training_config(str(config_path))
    data_cfg = _data_config(cfg)
    eval_cfg = _evaluation_config(cfg)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    vjepa_repo = args.get("vjepa_repo")
    vjepa_ckpt = args.get("vjepa_ckpt")
    if not _is_none_like(vjepa_repo):
        model_cfg.vjepa_repo = str(Path(str(vjepa_repo)).expanduser().resolve())
    if not _is_none_like(vjepa_ckpt):
        model_cfg.vjepa_ckpt = str(Path(str(vjepa_ckpt)).expanduser().resolve())

    checkpoint = Path(str(args.get("ckpt_setting"))).expanduser().resolve()
    stats_path = Path(str(args.get("dataset_stats_path"))).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CSWAM checkpoint not found: {checkpoint}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")

    device = str(args.get("device") or eval_cfg.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    action_horizon = _optional_int(args.get("action_horizon"))
    if action_horizon is None:
        action_horizon = _optional_int(eval_cfg.get("action_horizon"))
    if action_horizon is None:
        action_horizon = int(data_cfg.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    inference_steps = _optional_int(args.get("num_inference_steps"))
    if inference_steps is None:
        inference_steps = int(
            eval_cfg.get("num_inference_steps", cfg.get("eval_num_inference_steps", 20))
        )
    sigma_shift = _optional_float(args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _optional_float(eval_cfg.get("sigma_shift"))

    ratio = int(data_cfg.get("action_video_freq_ratio", 4))
    if ratio <= 0:
        raise ValueError(f"action_video_freq_ratio must be positive, got {ratio}")

    return CSWAMRpcAdapter(
        model_cfg=model_cfg,
        processor_cfg=data_cfg.processor,
        checkpoint_path=str(checkpoint),
        dataset_stats_path=str(stats_path),
        device=device,
        model_dtype=_model_dtype(
            str(args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
        ),
        action_horizon=action_horizon,
        num_inference_steps=inference_steps,
        sigma_shift=sigma_shift,
        seed=_optional_int(args.get("seed")),
        text_cfg_scale=float(
            args.get("text_cfg_scale", eval_cfg.get("text_cfg_scale", 1.0))
        ),
        negative_prompt=str(
            args.get("negative_prompt", eval_cfg.get("negative_prompt", ""))
        ),
        rand_device=str(args.get("rand_device", eval_cfg.get("rand_device", "cpu"))),
        tiled=_parse_bool(args.get("tiled", eval_cfg.get("tiled", False))),
        num_video_frames=(int(data_cfg.num_frames) - 1) // ratio + 1,
        text_encoder_device=str(args.get("text_encoder_device", "cpu")),
        vae_device_mode=str(args.get("vae_device_mode", "cpu")),
        text_cache_size=int(args.get("text_cache_size", 64)),
    )
