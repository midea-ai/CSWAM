"""Frozen V-JEPA 2.1 encoder wrapper for CSWAM.

The wrapper deliberately keeps representation extraction outside the trainable
MoT.  It accepts clips in ``[0, 1]`` and returns dense patch features in
``[B, D, T_latent, H_patch, W_patch]`` format, which is the native layout used
by the JEPA latent DiT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _unwrap_checkpoint(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"V-JEPA checkpoint must contain a dict, got {type(payload)}")
    # The V-JEPA 2.1 ViT-B checkpoint used by CSWAM is evaluated from
    # ``ema_encoder`` in the official 2.1 configuration.  Keep older
    # checkpoints supported as fallbacks, but prefer the 2.1 inference branch
    # whenever both encoder entries are present.
    for key in ("ema_encoder", "target_encoder", "encoder", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


class VJEPA2Encoder(nn.Module):
    """Load and freeze V-JEPA 2.1 ViT-B/16 while exposing dense tokens."""

    def __init__(
        self,
        repo: str,
        checkpoint: str,
        *,
        num_frames: int = 8,
        input_size: tuple[int, int] = (384, 320),
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.float32,
        strict_checkpoint: bool = False,
        compile_encoder: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()
        self.repo = str(repo)
        self.checkpoint = str(checkpoint)
        # ``num_frames`` configures the hub model at construction time.  The
        # frozen ViT can encode shorter compatible clips as well; ``forward``
        # derives the temporal grid from the actual input length.
        self.num_frames = int(num_frames)
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.embed_dim = int(embed_dim)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.compile_encoder = bool(compile_encoder)
        self.compile_mode = str(compile_mode)
        object.__setattr__(self, "_compiled_encoder", None)

        if len(self.input_size) != 2 or any(v <= 0 for v in self.input_size):
            raise ValueError(f"`input_size` must be a positive (H, W) pair, got {input_size}")
        if self.num_frames <= 0 or self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames must be positive and divisible by tubelet_size={self.tubelet_size}, "
                f"got {self.num_frames}"
            )
        if any(v % self.patch_size != 0 for v in self.input_size):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )
        if self.patch_size != 16 or self.tubelet_size != 2:
            raise ValueError(
                "The configured V-JEPA 2.1 ViT-B/16 checkpoint uses fixed "
                "patch_size=16 and tubelet_size=2."
            )

        model_obj = torch.hub.load(
            self.repo,
            "vjepa2_1_vit_base_384",
            source="local",
            pretrained=False,
            num_frames=self.num_frames,
        )
        self.encoder = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
        self._load_checkpoint()
        self.encoder.to(device=device, dtype=torch.float32)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.to(device=device)
        self.requires_grad_(False)
        if self.compile_encoder:
            self._enable_compile()

        logger.info(
            "Loaded frozen V-JEPA 2.1: repo=%s checkpoint=%s clip=%d input=%s "
            "dense_grid=%s compile=%s mode=%s",
            self.repo,
            self.checkpoint,
            self.num_frames,
            self.input_size,
            (
                self.num_frames // self.tubelet_size,
                self.input_size[0] // self.patch_size,
                self.input_size[1] // self.patch_size,
            ),
            self.compile_encoder,
            self.compile_mode,
        )

    def _enable_compile(self) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("V-JEPA compile requires a PyTorch build with torch.compile().")
        compiled_encoder = torch.compile(
            self.encoder,
            mode=self.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        # Keep the OptimizedModule outside nn.Module registration. The frozen
        # encoder remains the checkpoint owner, so compile never changes keys.
        object.__setattr__(self, "_compiled_encoder", compiled_encoder)

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V-JEPA checkpoint not found: {checkpoint_path}. "
                "Set `model.vjepa_ckpt` to the server-local checkpoint path."
            )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        state = _clean_state_dict(_unwrap_checkpoint(payload))
        encoder_state = self.encoder.state_dict()
        matched_numel = sum(
            int(target.numel())
            for key, target in encoder_state.items()
            if key in state
            and isinstance(state[key], torch.Tensor)
            and tuple(state[key].shape) == tuple(target.shape)
        )
        total_numel = sum(int(target.numel()) for target in encoder_state.values())
        loaded_fraction = matched_numel / max(total_numel, 1)
        if loaded_fraction < 0.95:
            raise RuntimeError(
                "V-JEPA checkpoint coverage is too low for a frozen target encoder: "
                f"loaded {loaded_fraction:.2%} of encoder parameters from {checkpoint_path}."
            )
        missing, unexpected = self.encoder.load_state_dict(state, strict=self.strict_checkpoint)
        if missing or unexpected:
            logger.warning(
                "V-JEPA checkpoint loaded with missing=%d unexpected=%d (strict=%s).",
                len(missing),
                len(unexpected),
                self.strict_checkpoint,
            )
            if self.strict_checkpoint:
                raise RuntimeError(
                    f"Strict V-JEPA checkpoint load failed: missing={missing[:8]}, "
                    f"unexpected={unexpected[:8]}"
                )
        logger.info("V-JEPA checkpoint parameter coverage: %.2f%%", loaded_fraction * 100.0)

    def train(self, mode: bool = True):
        # This encoder is a fixed target in CSWAM, even when the parent
        # model switches to train mode.
        super().train(False)
        self.encoder.eval()
        return self

    @staticmethod
    def _to_dense_grid(tokens: torch.Tensor, *, t: int, h: int, w: int, dim: int) -> torch.Tensor:
        if tokens.ndim == 5:
            # Some wrappers already return [B, T, H, W, D] or [B, D, T, H, W].
            if tokens.shape[-1] == dim and tuple(tokens.shape[1:4]) == (t, h, w):
                return tokens.permute(0, 4, 1, 2, 3).contiguous()
            if tokens.shape[1] == dim and tuple(tokens.shape[2:]) == (t, h, w):
                return tokens.contiguous()
        if tokens.ndim != 3:
            raise ValueError(
                "V-JEPA encoder output must be [B,N,D] or a 5D dense grid, "
                f"got shape {tuple(tokens.shape)}"
            )
        expected = t * h * w
        if tokens.shape[1] != expected or tokens.shape[2] != dim:
            raise ValueError(
                f"Cannot reshape V-JEPA tokens {tuple(tokens.shape)} into "
                f"[B,{dim},{t},{h},{w}], expected N={expected}."
            )
        return tokens.reshape(tokens.shape[0], t, h, w, dim).permute(0, 4, 1, 2, 3).contiguous()

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"V-JEPA clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        if clips.shape[1] != 3:
            raise ValueError(f"V-JEPA clips must have 3 channels, got {clips.shape[1]}")
        actual_num_frames = int(clips.shape[2])
        if actual_num_frames <= 0 or actual_num_frames % self.tubelet_size != 0:
            raise ValueError(
                "V-JEPA clip length must be positive and divisible by "
                f"tubelet_size={self.tubelet_size}, got {actual_num_frames}"
            )
        if actual_num_frames > self.num_frames:
            raise ValueError(
                f"V-JEPA clip length {actual_num_frames} exceeds the configured maximum "
                f"of {self.num_frames} frames."
            )
        if tuple(clips.shape[-2:]) != self.input_size:
            raise ValueError(
                f"V-JEPA input size must be {self.input_size}, got {tuple(clips.shape[-2:])}"
            )
        x = clips.to(device=self.mean.device, dtype=torch.float32).clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        # The parent training loop uses autocast for the trainable experts.
        # Disable it locally so frozen V-JEPA targets remain in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            encoder = self._compiled_encoder
            tokens = encoder(x) if encoder is not None else self.encoder(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if isinstance(tokens, dict):
            for key in ("tokens", "features", "x", "last_hidden_state"):
                if key in tokens:
                    tokens = tokens[key]
                    break
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"Unsupported V-JEPA output type: {type(tokens)}")
        return self._to_dense_grid(
            tokens,
            t=actual_num_frames // self.tubelet_size,
            h=self.input_size[0] // self.patch_size,
            w=self.input_size[1] // self.patch_size,
            dim=self.embed_dim,
        )
