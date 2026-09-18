"""CSWAM: FastWAM with a frozen V-JEPA history/future expert."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger
from fastwam.utils.timing import CudaEventTrace

from .action_dit import ActionDiT
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .jepa_encoder import VJEPA2Encoder
from .jepa_inputs import derive_jepa_future_from_video
from .jepa_video_dit import JEPAVideoDiT
from .mot import MoT

logger = get_logger(__name__)


class CSWAM(FastWAM):
    """Three-expert Wan + JEPA latent DiT + ActionDiT model."""

    CHECKPOINT_FORMAT = "cswam-v1"
    LEGACY_CHECKPOINT_FORMATS = frozenset({"jepawam01-v2"})

    def __init__(
        self,
        *,
        video_expert,
        jepa_expert: JEPAVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: int,
        proprio_dim: Optional[int],
        vjepa_repo: str,
        vjepa_ckpt: str,
        vjepa_history_num_frames: int = 8,
        vjepa_future_num_frames: int = 8,
        jepa_history_offsets: Optional[list[int] | tuple[int, ...]] = None,
        jepa_future_offsets: Optional[list[int] | tuple[int, ...]] = None,
        allow_repeated_jepa_history_offsets: bool = False,
        vjepa_input_size: tuple[int, int] = (384, 320),
        vjepa_patch_size: int = 16,
        vjepa_tubelet_size: int = 2,
        vjepa_embed_dim: int = 768,
        vjepa_strict_checkpoint: bool = False,
        compile_vjepa: bool = False,
        vjepa_compile_mode: str = "default",
        compile_vae: bool = False,
        vae_compile_mode: str = "default",
        vae_encode_micro_batch_size: int = 1,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_jepa: float = 0.02,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        self.jepa_expert = jepa_expert
        self.train_jepa_scheduler = self.train_video_scheduler
        self.infer_jepa_scheduler = self.infer_video_scheduler
        self.loss_lambda_jepa = float(loss_lambda_jepa)
        self.vjepa_repo = str(vjepa_repo)
        self.vjepa_ckpt = str(vjepa_ckpt)
        self.vjepa_history_num_frames = int(vjepa_history_num_frames)
        self.vjepa_future_num_frames = int(vjepa_future_num_frames)
        self.vjepa_tubelet_size = int(vjepa_tubelet_size)
        self.vjepa_embed_dim = int(vjepa_embed_dim)
        self.compile_vjepa = bool(compile_vjepa)
        self.vjepa_compile_mode = str(vjepa_compile_mode)
        self.compile_vae = bool(compile_vae)
        self.vae_compile_mode = str(vae_compile_mode)
        self.vae_encode_micro_batch_size = int(vae_encode_micro_batch_size)
        if self.vae_encode_micro_batch_size <= 0:
            raise ValueError(
                "`vae_encode_micro_batch_size` must be positive, got "
                f"{self.vae_encode_micro_batch_size}."
            )
        self.vae.configure_encode_optimization(
            compile_encoder=self.compile_vae,
            compile_mode=self.vae_compile_mode,
        )
        if jepa_history_offsets is None:
            jepa_history_offsets = [
                -4 * index
                for index in range(self.vjepa_history_num_frames - 1, -1, -1)
            ]
        if jepa_future_offsets is None:
            jepa_future_offsets = [
                4 * (index + 1) for index in range(self.vjepa_future_num_frames)
            ]
        self.jepa_history_offsets = tuple(int(value) for value in jepa_history_offsets)
        self.jepa_future_offsets = tuple(int(value) for value in jepa_future_offsets)
        self.allow_repeated_jepa_history_offsets = bool(allow_repeated_jepa_history_offsets)
        for name, num_frames in (
            ("history", self.vjepa_history_num_frames),
            ("future", self.vjepa_future_num_frames),
        ):
            if num_frames <= 0 or num_frames % self.vjepa_tubelet_size != 0:
                raise ValueError(
                    f"V-JEPA {name} frames must be positive and divisible by "
                    f"tubelet_size={self.vjepa_tubelet_size}, got {num_frames}"
                )
        self._validate_frame_offsets(
            name="history",
            offsets=self.jepa_history_offsets,
            expected_length=self.vjepa_history_num_frames,
            allow_zero=True,
            allow_repeats=self.allow_repeated_jepa_history_offsets,
        )
        self._validate_frame_offsets(
            name="future",
            offsets=self.jepa_future_offsets,
            expected_length=self.vjepa_future_num_frames,
            allow_zero=False,
            allow_repeats=False,
        )
        vjepa_encoder = VJEPA2Encoder(
            repo=vjepa_repo,
            checkpoint=vjepa_ckpt,
            num_frames=max(
                self.vjepa_history_num_frames,
                self.vjepa_future_num_frames,
            ),
            input_size=tuple(vjepa_input_size),
            patch_size=vjepa_patch_size,
            tubelet_size=vjepa_tubelet_size,
            embed_dim=vjepa_embed_dim,
            device=device,
            strict_checkpoint=vjepa_strict_checkpoint,
            compile_encoder=self.compile_vjepa,
            compile_mode=self.vjepa_compile_mode,
        )
        # The encoder is a fixed external feature extractor loaded independently
        # on each rank.  Keep it outside `_modules` so Accelerate/DeepSpeed state
        # checkpoints do not duplicate its weights; `.to()` below manages its
        # device explicitly.
        object.__setattr__(self, "vjepa_encoder", vjepa_encoder)
        self.vjepa_encoder.eval()
        self.vjepa_encoder.requires_grad_(False)
        self.model_paths = {}
        # CSWAM uses Wan and JEPA only to build the static visual context
        # for action denoising; it does not sample a future video at inference.
        self.action_only_inference = True
        self._training_timing_enabled = False
        self._active_training_timing_trace: Optional[CudaEventTrace] = None
        self._training_timing_traces: list[CudaEventTrace] = []
        logger.info(
            "CSWAM encoder optimization: V-JEPA compile=%s mode=%s; "
            "VAE compile=%s mode=%s micro_batch=%d",
            self.compile_vjepa,
            self.vjepa_compile_mode,
            self.compile_vae,
            self.vae_compile_mode,
            self.vae_encode_micro_batch_size,
        )

    def set_training_timing_enabled(self, enabled: bool) -> None:
        self._training_timing_enabled = bool(enabled)
        if not self._training_timing_enabled:
            self._active_training_timing_trace = None
            self._training_timing_traces.clear()

    def _start_training_timing_trace(self) -> None:
        if not self._training_timing_enabled:
            return
        self._active_training_timing_trace = CudaEventTrace(enabled=True)

    def _training_timing_mark(self, name: str) -> None:
        trace = self._active_training_timing_trace
        if trace is not None:
            trace.mark(name)

    def _finish_training_timing_trace(self) -> None:
        trace = self._active_training_timing_trace
        self._active_training_timing_trace = None
        if trace is not None and trace.has_intervals:
            self._training_timing_traces.append(trace)

    def pop_training_timing_traces(self) -> list[CudaEventTrace]:
        traces = self._training_timing_traces
        self._training_timing_traces = []
        return traces

    def to(self, *args, **kwargs):
        # FastWAM moves all registered children together.  Keep the frozen
        # V-JEPA target encoder in fp32 even when trainable experts use bf16.
        result = super().to(*args, **kwargs)
        if hasattr(self, "vjepa_encoder"):
            self.vjepa_encoder.to(*args, **kwargs)
            self.vjepa_encoder.to(dtype=torch.float32)
            self.vjepa_encoder.eval()
            self.vjepa_encoder.requires_grad_(False)
        return result

    @staticmethod
    def _validate_frame_offsets(
        *,
        name: str,
        offsets: tuple[int, ...],
        expected_length: int,
        allow_zero: bool,
        allow_repeats: bool = False,
    ) -> None:
        if len(offsets) != int(expected_length):
            raise ValueError(
                f"JEPA {name} offsets must contain {expected_length} entries, got {offsets}."
            )
        if tuple(sorted(offsets)) != offsets or (
            not allow_repeats and len(offsets) != len(set(offsets))
        ):
            raise ValueError(
                f"JEPA {name} offsets must be increasing"
                f"{' with optional repeats' if allow_repeats else ' and unique'}, got {offsets}."
            )
        if allow_zero:
            if offsets[-1] != 0 or any(value > 0 for value in offsets):
                raise ValueError(
                    "JEPA history offsets must be non-positive and end at the current frame 0, "
                    f"got {offsets}."
                )
        elif any(value <= 0 for value in offsets):
            raise ValueError(f"JEPA future offsets must be positive, got {offsets}.")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "vjepa_repo": self.vjepa_repo,
            "vjepa_ckpt": self.vjepa_ckpt,
            "history_num_frames": self.vjepa_history_num_frames,
            "future_num_frames": self.vjepa_future_num_frames,
            "history_offsets": list(self.jepa_history_offsets),
            "future_offsets": list(self.jepa_future_offsets),
            "input_size": list(self.vjepa_encoder.input_size),
            "patch_size": int(self.vjepa_encoder.patch_size),
            "tubelet_size": self.vjepa_tubelet_size,
            "embed_dim": self.vjepa_embed_dim,
            "jepa_latent_patch_size": list(self.jepa_expert.latent_patch_size),
            "jepa_output_patch_space": self.jepa_expert.output_patch_space,
            "loss_lambda_video": self.loss_lambda_video,
            "loss_lambda_jepa": self.loss_lambda_jepa,
            "loss_lambda_action": self.loss_lambda_action,
        }

    @classmethod
    def _metadata_from_checkpoint(cls, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        metadata = payload.get("cswam")
        if isinstance(metadata, dict):
            return metadata, False
        metadata = payload.get("jepawam01")
        if isinstance(metadata, dict):
            return metadata, True
        return None, False

    def save_checkpoint(self, path, optimizer=None, step=None):
        # Save the three trainable experts, but not the separately loaded frozen
        # V-JEPA encoder weights.
        payload = {
            "mot": self.mot.state_dict(),
            "cswam": self._checkpoint_metadata(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location=self.device)
        metadata, legacy_metadata = self._metadata_from_checkpoint(payload)
        strict_mot = isinstance(metadata, dict)
        if strict_mot:
            expected = self._checkpoint_metadata()
            saved_format = metadata.get("format")
            supported_formats = {self.CHECKPOINT_FORMAT, *self.LEGACY_CHECKPOINT_FORMATS}
            if saved_format not in supported_formats:
                raise ValueError(
                    f"Unsupported CSWAM checkpoint format for {path}: {saved_format!r}. "
                    f"Expected one of {sorted(supported_formats)}."
                )
            if legacy_metadata:
                logger.info("Loading legacy checkpoint metadata from %s.", path)
            critical_keys = (
                "history_num_frames",
                "future_num_frames",
                "history_offsets",
                "future_offsets",
                "input_size",
                "patch_size",
                "tubelet_size",
                "embed_dim",
            )
            if not legacy_metadata:
                critical_keys += (
                    "jepa_latent_patch_size",
                    "jepa_output_patch_space",
                )
            for path_key in ("vjepa_repo", "vjepa_ckpt"):
                saved_path = metadata.get(path_key)
                runtime_path = expected[path_key]
                if saved_path and saved_path != runtime_path:
                    logger.warning(
                        "CSWAM %s path differs from checkpoint metadata; using runtime path. "
                        "checkpoint=%s runtime=%s",
                        path_key,
                        saved_path,
                        runtime_path,
                    )
            mismatches = {
                key: (metadata.get(key), expected[key])
                for key in critical_keys
                if metadata.get(key) != expected[key]
            }
            if mismatches:
                raise ValueError(
                    f"CSWAM checkpoint/model contract mismatch for {path}: {mismatches}"
                )
        else:
            logger.warning(
                "Checkpoint %s has no CSWAM metadata; loading MoT non-strictly as a legacy checkpoint.",
                path,
            )

        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=strict_mot)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into the Wan video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping the current initialization."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but this model has proprio_dim=None; ignoring."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        jepa_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        jepa_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        vjepa_repo: str = "/data/share/1919650160032350208/tim/sub_fastwam/subtask_wam/JEPA_WAM/vjepa2",
        vjepa_ckpt: str = "/data/share/1919650160032350208/tim/sub_fastwam/subtask_wam/JEPA_WAM/wam-posttrain_dev/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
        vjepa_history_num_frames: int = 8,
        vjepa_future_num_frames: int = 8,
        jepa_history_offsets: Optional[list[int] | tuple[int, ...]] = None,
        jepa_future_offsets: Optional[list[int] | tuple[int, ...]] = None,
        allow_repeated_jepa_history_offsets: bool = False,
        vjepa_input_size: tuple[int, int] = (384, 320),
        vjepa_patch_size: int = 16,
        vjepa_tubelet_size: int = 2,
        vjepa_embed_dim: int = 768,
        vjepa_strict_checkpoint: bool = False,
        compile_vjepa: bool = False,
        vjepa_compile_mode: str = "default",
        compile_vae: bool = False,
        vae_compile_mode: str = "default",
        vae_encode_micro_batch_size: int = 1,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_jepa: float = 0.02,
        loss_lambda_action: float = 1.0,
    ):
        if cls is not CSWAM:
            logger.warning(
                "CSWAM factory was bound to %s; constructing CSWAM explicitly.",
                cls.__qualname__,
            )
        if video_dit_config is None or jepa_dit_config is None or action_dit_config is None:
            raise ValueError("video_dit_config, jepa_dit_config, and action_dit_config are required.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        jepa_expert = JEPAVideoDiT(**jepa_dit_config).to(device=device, dtype=torch_dtype)
        init_report = jepa_expert.init_from_action_dit(action_expert)
        logger.info(
            "JEPA-DiT ActionDiT initialization copied=%d skipped=%d unexpected_source=%d",
            len(init_report["copied"]),
            len(init_report["skipped"]),
            len(init_report["unexpected_source"]),
        )

        for name, expert in (("video", video_expert), ("jepa", jepa_expert), ("action", action_expert)):
            if int(expert.num_heads) != int(action_expert.num_heads):
                raise ValueError(f"{name} expert num_heads must match ActionDiT for MoT.")
            if int(expert.attn_head_dim) != int(action_expert.attn_head_dim):
                raise ValueError(f"{name} expert attn_head_dim must match ActionDiT for MoT.")
            if int(len(expert.blocks)) != int(len(action_expert.blocks)):
                raise ValueError(f"{name} expert num_layers must match ActionDiT for MoT.")

        if jepa_dit_pretrained_path:
            CSWAM._load_jepa_dit_checkpoint(jepa_expert, jepa_dit_pretrained_path)

        mot = MoT(
            mixtures={"video": video_expert, "jepa": jepa_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = CSWAM(
            video_expert=video_expert,
            jepa_expert=jepa_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            vjepa_repo=vjepa_repo,
            vjepa_ckpt=vjepa_ckpt,
            vjepa_history_num_frames=vjepa_history_num_frames,
            vjepa_future_num_frames=vjepa_future_num_frames,
            jepa_history_offsets=jepa_history_offsets,
            jepa_future_offsets=jepa_future_offsets,
            allow_repeated_jepa_history_offsets=allow_repeated_jepa_history_offsets,
            vjepa_input_size=vjepa_input_size,
            vjepa_patch_size=vjepa_patch_size,
            vjepa_tubelet_size=vjepa_tubelet_size,
            vjepa_embed_dim=vjepa_embed_dim,
            vjepa_strict_checkpoint=vjepa_strict_checkpoint,
            compile_vjepa=compile_vjepa,
            vjepa_compile_mode=vjepa_compile_mode,
            compile_vae=compile_vae,
            vae_compile_mode=vae_compile_mode,
            vae_encode_micro_batch_size=vae_encode_micro_batch_size,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_jepa=loss_lambda_jepa,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "vjepa_repo": vjepa_repo,
            "vjepa_ckpt": vjepa_ckpt,
            "jepa_dit_pretrained": jepa_dit_pretrained_path,
        }
        return model

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        return self.vae.encode_batched(
            video_tensor,
            device=self.device,
            micro_batch_size=self.vae_encode_micro_batch_size,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @staticmethod
    def _load_jepa_dit_checkpoint(model: JEPAVideoDiT, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"JEPA-DiT checkpoint not found: {checkpoint_path}")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("jepa_dit", "state_dict", "model", "mot"):
                if isinstance(payload.get(key), dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported JEPA-DiT checkpoint payload: {type(payload)}")
        missing, unexpected = model.load_state_dict(payload, strict=False)
        if missing or unexpected:
            logger.warning(
                "Loaded JEPA-DiT checkpoint non-strictly: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        required = ("jepa_history_video",)
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(
                "CSWAM requires dataset fields "
                f"{required}; missing={missing}. Set `load_jepa_video: true`."
            )
        history = sample["jepa_history_video"]
        if history.ndim != 5 or history.shape[1] != 3:
            raise ValueError(
                f"`jepa_history_video` must be [B,3,T,H,W], got {tuple(history.shape)}"
            )
        if history.shape[2] != self.vjepa_history_num_frames:
            raise ValueError(
                "`jepa_history_video` must contain "
                f"{self.vjepa_history_num_frames} frames, got {history.shape[2]}"
            )
        if tuple(history.shape[-2:]) != tuple(self.vjepa_encoder.input_size):
            raise ValueError(
                f"`jepa_history_video` spatial size must be {self.vjepa_encoder.input_size}, "
                f"got {tuple(history.shape[-2:])}"
            )
        future, future_is_pad = derive_jepa_future_from_video(
            sample,
            expected_num_frames=self.vjepa_future_num_frames,
            expected_size=tuple(self.vjepa_encoder.input_size),
            device=self.device,
        )
        if history.shape[0] != future.shape[0]:
            raise ValueError("JEPA history and Wan video batch sizes must match.")
        inputs.update(
            {
                "jepa_history_video": history.to(self.device, dtype=torch.float32, non_blocking=True),
                "jepa_future_video": future,
                "jepa_future_is_pad": future_is_pad,
            }
        )
        return inputs

    @staticmethod
    def _build_clean_prefix_mask(
        seq_len: int,
        tokens_per_frame: int,
        clean_prefix_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        clean_len = int(tokens_per_frame) * int(clean_prefix_frames)
        if clean_len < 0 or clean_len > seq_len:
            raise ValueError(
                f"Invalid clean prefix: frames={clean_prefix_frames}, tokens_per_frame={tokens_per_frame}, "
                f"seq_len={seq_len}"
            )
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        mask[:clean_len, :clean_len] = True
        mask[clean_len:, :] = True
        return mask

    @torch.no_grad()
    def _build_cswam_attention_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        jepa_seq_len: int,
        jepa_tokens_per_frame: int,
        jepa_clean_frames: int,
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = video_seq_len + jepa_seq_len + action_seq_len
        video_start, video_end = 0, video_seq_len
        jepa_start, jepa_end = video_end, video_end + jepa_seq_len
        action_start, action_end = jepa_end, total
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        wan_self = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_start:video_end, video_start:video_end] = wan_self
        jepa_self = self._build_clean_prefix_mask(
            seq_len=jepa_seq_len,
            tokens_per_frame=jepa_tokens_per_frame,
            clean_prefix_frames=jepa_clean_frames,
            device=device,
        )
        mask[jepa_start:jepa_end, jepa_start:jepa_end] = jepa_self
        mask[action_start:action_end, action_start:action_end] = True

        video_clean = min(int(video_tokens_per_frame), video_seq_len)
        jepa_clean = int(jepa_tokens_per_frame) * int(jepa_clean_frames)

        # Wc -> Wc, Jh; Wf -> Wc, Wf, Jh, Jf.
        mask[video_start:video_start + video_clean, video_start:video_start + video_clean] = True
        mask[video_start:video_start + video_clean, jepa_start:jepa_start + jepa_clean] = True
        if video_clean < video_seq_len:
            mask[video_start + video_clean:video_end, video_start:video_end] = True
            mask[video_start + video_clean:video_end, jepa_start:jepa_end] = True

        # Jh -> Wc, Jh; Jf -> Wc, Wf, Jh, Jf.
        mask[jepa_start:jepa_start + jepa_clean, video_start:video_start + video_clean] = True
        if jepa_clean < jepa_seq_len:
            mask[jepa_start + jepa_clean:jepa_end, video_start:video_end] = True
            mask[jepa_start + jepa_clean:jepa_end, jepa_start:jepa_end] = True

        # Action reads only current Wan tokens, clean JEPA history, and itself.
        mask[action_start:action_end, video_start:video_start + video_clean] = True
        mask[action_start:action_end, jepa_start:jepa_start + jepa_clean] = True
        return mask

    @staticmethod
    def _compute_jepa_loss_per_sample(
        pred_future: torch.Tensor,
        target_future: torch.Tensor,
        future_is_pad: torch.Tensor,
        tubelet_size: int,
    ) -> torch.Tensor:
        if pred_future.shape != target_future.shape or pred_future.ndim != 5:
            raise ValueError(
                f"JEPA prediction/target must match as [B,D,T,H,W], got "
                f"{tuple(pred_future.shape)} and {tuple(target_future.shape)}"
            )
        batch_size, _, latent_steps, _, _ = pred_future.shape
        if future_is_pad.shape[1] != latent_steps * tubelet_size:
            raise ValueError(
                f"Future pad length={future_is_pad.shape[1]} must equal "
                f"latent_steps({latent_steps})*tubelet_size({tubelet_size})."
            )
        pad_latent = future_is_pad.view(batch_size, latent_steps, tubelet_size).any(dim=2)
        loss_steps = F.mse_loss(
            pred_future.float(), target_future.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        valid = (~pad_latent).to(device=loss_steps.device, dtype=loss_steps.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (loss_steps * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        self._start_training_timing_trace()
        inputs = self.build_inputs(sample, tiled=tiled)
        self._training_timing_mark("input_finalize")
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_video[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        self._training_timing_mark("noise_setup")

        # `no_grad` keeps the target encoder frozen without creating inference
        # tensors, which must subsequently enter the trainable JEPA-DiT.
        with torch.no_grad():
            z_history = self.vjepa_encoder(inputs["jepa_history_video"])
            self._training_timing_mark("vjepa_history")
            z_future = self.vjepa_encoder(inputs["jepa_future_video"])
            self._training_timing_mark("vjepa_future")
        z_history = z_history.to(device=self.device, dtype=self.torch_dtype)
        z_future = z_future.to(device=self.device, dtype=self.torch_dtype)
        noise_jepa = torch.randn_like(z_future)
        noisy_future = self.train_jepa_scheduler.add_noise(z_future, noise_jepa, timestep_video)
        target_jepa = self.train_jepa_scheduler.training_target(z_future, noise_jepa, timestep_video)
        # Merged JEPA output predicts one vector per latent patch group, so its
        # fixed flow target must use the same compact spatial grid.
        target_jepa = self.jepa_expert.target_to_output_space(target_jepa)
        jepa_latents = torch.cat([z_history, noisy_future], dim=2)
        self._training_timing_mark("jepa_prepare")

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        jepa_pre = self.jepa_expert.pre_dit(
            x=jepa_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            clean_prefix_frames=z_history.shape[2],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        self._training_timing_mark("pre_dit")

        attention_mask = self._build_cswam_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            jepa_seq_len=jepa_pre["tokens"].shape[1],
            jepa_tokens_per_frame=int(jepa_pre["meta"]["tokens_per_frame"]),
            jepa_clean_frames=int(jepa_pre["meta"]["clean_prefix_frames"]),
            action_seq_len=action_pre["tokens"].shape[1],
            device=input_latents.device,
        )
        self._training_timing_mark("attention_mask")
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "jepa": jepa_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "jepa": jepa_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                name: {"context": state["context"], "mask": state["context_mask"]}
                for name, state in (("video", video_pre), ("jepa", jepa_pre), ("action", action_pre))
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "jepa": jepa_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        self._training_timing_mark("mot")

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_jepa = self.jepa_expert.post_dit(tokens_out["jepa"], jepa_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        history_steps = int(z_history.shape[2])
        pred_jepa_future = pred_jepa[:, :, history_steps:]
        loss_jepa_per_sample = self._compute_jepa_loss_per_sample(
            pred_future=pred_jepa_future,
            target_future=target_jepa,
            future_is_pad=inputs["jepa_future_is_pad"],
            tubelet_size=self.vjepa_tubelet_size,
        )
        loss_jepa = (
            loss_jepa_per_sample
            * self.train_jepa_scheduler.training_weight(timestep_video).to(
                loss_jepa_per_sample.device, dtype=loss_jepa_per_sample.dtype
            )
        ).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs["action_dim_is_pad"],
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_jepa * loss_jepa
            + self.loss_lambda_action * loss_action
        )
        self._training_timing_mark("post_and_loss")
        self._finish_training_timing_trace()
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_jepa": self.loss_lambda_jepa * float(loss_jepa.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        jepa_history_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Predict actions using clean Wan and JEPA history features.

        ``input_image`` follows the existing FastWAM convention and is a
        single Wan mosaic in ``[-1, 1]``.  ``jepa_history_video`` is optional;
        when supplied it must be a ``[B, 3, T, H, W]`` clip in ``[0, 1]`` with
        the same mosaic layout used by the training dataset.  If omitted, the
        current image is repeated as a conservative single-frame fallback.
        The future JEPA frames are intentionally absent at inference time.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

        if jepa_history_video is None:
            # The real control loop should provide a history queue.  Repeating
            # the current frame keeps the public API usable for smoke tests and
            # mirrors the dataset's zero-history fallback.
            jepa_history_video = ((input_image.float() + 1.0) * 0.5).unsqueeze(2)
            jepa_history_video = jepa_history_video.expand(
                -1, -1, self.vjepa_history_num_frames, -1, -1
            ).contiguous()
            if tuple(jepa_history_video.shape[-2:]) != tuple(self.vjepa_encoder.input_size):
                batch_size, channels, num_frames, height, width = jepa_history_video.shape
                resized = F.interpolate(
                    jepa_history_video.permute(0, 2, 1, 3, 4).reshape(
                        batch_size * num_frames, channels, height, width
                    ),
                    size=tuple(self.vjepa_encoder.input_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                jepa_history_video = resized.reshape(
                    batch_size, num_frames, channels, *self.vjepa_encoder.input_size
                ).permute(0, 2, 1, 3, 4).contiguous()
        else:
            if jepa_history_video.ndim == 4:
                jepa_history_video = jepa_history_video.unsqueeze(0)
            if jepa_history_video.ndim != 5:
                raise ValueError(
                    "`jepa_history_video` must be [3,T,H,W] or [B,3,T,H,W], "
                    f"got {tuple(jepa_history_video.shape)}"
                )
            if jepa_history_video.shape[0] != 1 or jepa_history_video.shape[1] != 3:
                raise ValueError(
                    "`jepa_history_video` must have batch/channel shape [1,3,...], "
                    f"got {tuple(jepa_history_video.shape)}"
                )
            if jepa_history_video.shape[2] != self.vjepa_history_num_frames:
                raise ValueError(
                    f"`jepa_history_video` must contain {self.vjepa_history_num_frames} frames, "
                    f"got {jepa_history_video.shape[2]}"
                )
            if tuple(jepa_history_video.shape[-2:]) != tuple(self.vjepa_encoder.input_size):
                raise ValueError(
                    "`jepa_history_video` spatial size must match the V-JEPA encoder, "
                    f"expected {self.vjepa_encoder.input_size}, got {tuple(jepa_history_video.shape[-2:])}"
                )
            jepa_history_video = jepa_history_video.float().clamp(0.0, 1.0)

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # Wan sees the current observation; JEPA sees the clean history clip.
        first_frame_t = torch.zeros(
            (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=first_frame_t,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        jepa_history_video = jepa_history_video.to(device=self.device, dtype=torch.float32, non_blocking=True)
        z_history = self.vjepa_encoder(jepa_history_video).to(dtype=self.torch_dtype)
        jepa_pre = self.jepa_expert.pre_dit(
            x=z_history,
            timestep=torch.zeros_like(first_frame_t),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            clean_prefix_frames=int(z_history.shape[2]),
        )

        static_seq_len = int(video_pre["tokens"].shape[1] + jepa_pre["tokens"].shape[1])
        full_attention_mask = self._build_cswam_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            jepa_seq_len=int(jepa_pre["tokens"].shape[1]),
            jepa_tokens_per_frame=int(jepa_pre["meta"]["tokens_per_frame"]),
            jepa_clean_frames=int(jepa_pre["meta"]["clean_prefix_frames"]),
            action_seq_len=int(action_horizon),
            device=video_pre["tokens"].device,
        )
        static_cache = self.mot.prefill_static_expert_cache(
            expert_inputs={
                "video": {
                    "tokens": video_pre["tokens"],
                    "freqs": video_pre["freqs"],
                    "t_mod": video_pre["t_mod"],
                    "context": video_pre["context"],
                    "context_mask": video_pre["context_mask"],
                },
                "jepa": {
                    "tokens": jepa_pre["tokens"],
                    "freqs": jepa_pre["freqs"],
                    "t_mod": jepa_pre["t_mod"],
                    "context": jepa_pre["context"],
                    "context_mask": jepa_pre["context_mask"],
                },
            },
            attention_mask=full_attention_mask[:static_seq_len, :static_seq_len],
            static_expert_order=["video", "jepa"],
        )

        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_static_cache(
                action_inputs={
                    "tokens": action_pre["tokens"],
                    "freqs": action_pre["freqs"],
                    "t_mod": action_pre["t_mod"],
                    "context": action_pre["context"],
                    "context_mask": action_pre["context_mask"],
                },
                static_cache=static_cache,
                attention_mask=full_attention_mask,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: Optional[int] = None,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        jepa_history_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper for the existing FastWAM inference API.

        ``num_frames``, ``action``, and ``action_cfg_scale`` are accepted so
        the shared trainer/evaluation code can call this model.  CSWAM
        performs action-only inference, so the action horizon is either read
        from ``action_horizon`` or inferred from the supplied reference action.
        """
        del num_frames, action_cfg_scale
        if action_horizon is None:
            if action is None:
                raise ValueError(
                    "CSWAM requires `action_horizon` for inference when no reference `action` is supplied."
                )
            if action.ndim not in (2, 3):
                raise ValueError(
                    f"Reference `action` must be [T,D] or [B,T,D], got {tuple(action.shape)}"
                )
            action_horizon = int(action.shape[-2])
        return self.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=int(action_horizon),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            jepa_history_video=jepa_history_video,
        )
