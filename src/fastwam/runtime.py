import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalized_cfg_value(value, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("-", "_")


def _make_train_dataset_cfg(data_cfg: DictConfig, *, use_full_train: bool) -> DictConfig:
    train_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg.train, resolve=True))
    if use_full_train:
        train_cfg.val_set_proportion = 0.0
        train_cfg.is_training_set = True
    return train_cfg


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_cswam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    jepa_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    jepa_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    vjepa_repo: str = "",
    vjepa_ckpt: str = "",
    vjepa_history_num_frames: int = 8,
    vjepa_future_num_frames: int = 8,
    jepa_history_offsets=None,
    jepa_future_offsets=None,
    allow_repeated_jepa_history_offsets: bool = False,
    vjepa_input_size=(384, 320),
    vjepa_patch_size: int = 16,
    vjepa_tubelet_size: int = 2,
    vjepa_embed_dim: int = 768,
    vjepa_strict_checkpoint: bool = False,
    compile_vjepa: bool = False,
    vjepa_compile_mode: str = "default",
    compile_vae: bool = False,
    vae_compile_mode: str = "default",
    vae_encode_micro_batch_size: int = 1,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Instantiate the CSWAM video, semantic, and action experts."""
    from .models.wan22.cswam import CSWAM

    def _as_dict(value, name: str):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
        return value

    if not vjepa_repo or not vjepa_ckpt:
        raise ValueError("`vjepa_repo` and `vjepa_ckpt` are required for CSWAM.")

    video_dit_config = _as_dict(video_dit_config, "video_dit_config")
    jepa_dit_config = _as_dict(jepa_dit_config, "jepa_dit_config")
    action_dit_config = _as_dict(action_dit_config, "action_dit_config")
    video_scheduler = _as_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_dict(action_scheduler, "action_scheduler")
    loss = _as_dict(loss, "loss")

    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required - set(action_scheduler)
    if missing:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing)}")

    return CSWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        jepa_dit_config=jepa_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        jepa_dit_pretrained_path=jepa_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vjepa_repo=str(vjepa_repo),
        vjepa_ckpt=str(vjepa_ckpt),
        vjepa_history_num_frames=int(vjepa_history_num_frames),
        vjepa_future_num_frames=int(vjepa_future_num_frames),
        jepa_history_offsets=None if jepa_history_offsets is None else [int(v) for v in jepa_history_offsets],
        jepa_future_offsets=None if jepa_future_offsets is None else [int(v) for v in jepa_future_offsets],
        allow_repeated_jepa_history_offsets=bool(allow_repeated_jepa_history_offsets),
        vjepa_input_size=tuple(int(v) for v in vjepa_input_size),
        vjepa_patch_size=int(vjepa_patch_size),
        vjepa_tubelet_size=int(vjepa_tubelet_size),
        vjepa_embed_dim=int(vjepa_embed_dim),
        vjepa_strict_checkpoint=bool(vjepa_strict_checkpoint),
        compile_vjepa=bool(compile_vjepa),
        vjepa_compile_mode=str(vjepa_compile_mode),
        compile_vae=bool(compile_vae),
        vae_compile_mode=str(vae_compile_mode),
        vae_encode_micro_batch_size=int(vae_encode_micro_batch_size),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_jepa=float(loss.get("lambda_jepa", 0.02)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )

def build_datasets(data_cfg: DictConfig):
    val_mode = _normalized_cfg_value(data_cfg.get("val_mode"), "reuse_train")
    valid_val_modes = {"reuse_train", "train", "train_sample", "none", "disabled", "off", "false", "separate"}
    if val_mode not in valid_val_modes:
        raise ValueError(
            f"Unsupported data.val_mode={val_mode!r}. "
            "Expected one of: reuse_train, train, train_sample, none, disabled, off, false, separate."
        )

    use_full_train = val_mode != "separate"
    train_ds = instantiate(_make_train_dataset_cfg(data_cfg, use_full_train=use_full_train))

    if val_mode in {"none", "disabled", "off", "false"}:
        logger.info("Validation dataset disabled via data.val_mode=%s; train uses full dataset.", val_mode)
        val_ds = None
    elif val_mode in {"reuse_train", "train", "train_sample"}:
        logger.info("Validation dataset reuses train dataset; train uses full dataset.")
        val_ds = train_ds
    else:
        if data_cfg.get("val") is None:
            raise ValueError("data.val_mode='separate' requires a data.val config.")
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
        use_rich_handler=False,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    logger.info("Initializing CSWAM on %s with dtype=%s.", model_device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)
    val_len = "None" if val_ds is None else len(val_ds)
    logger.info("Train/val dataset size: %s/%s", len(train_ds), val_len)
    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
