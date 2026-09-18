from __future__ import annotations

import torch


def derive_jepa_future_from_video(
    sample: dict,
    *,
    expected_num_frames: int,
    expected_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive JEPA future RGB and padding directly from the Wan video batch."""
    video = sample.get("video")
    image_is_pad = sample.get("image_is_pad")
    if not isinstance(video, torch.Tensor):
        raise KeyError("JEPA training requires `sample['video']`.")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
    if video.shape[2] - 1 != int(expected_num_frames):
        raise ValueError(
            "Wan and JEPA future frame counts must match: "
            f"Wan={video.shape[2] - 1}, JEPA={expected_num_frames}"
        )
    if tuple(video.shape[-2:]) != tuple(expected_size):
        raise ValueError(
            f"Wan video spatial size must match JEPA input size {expected_size}, "
            f"got {tuple(video.shape[-2:])}"
        )
    if not isinstance(image_is_pad, torch.Tensor):
        raise KeyError("JEPA training requires `sample['image_is_pad']`.")
    if tuple(image_is_pad.shape) != (video.shape[0], video.shape[2]):
        raise ValueError(
            "`sample['image_is_pad']` must match the Wan video temporal shape: "
            f"expected {(video.shape[0], video.shape[2])}, got {tuple(image_is_pad.shape)}"
        )

    # copy=True guarantees that the in-place inverse normalization below does
    # not mutate the Wan input even when the incoming batch is already on GPU.
    future = video[:, :, 1:].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        copy=True,
    )
    future.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    future_is_pad = image_is_pad[:, 1:].to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    )
    return future, future_is_pad
