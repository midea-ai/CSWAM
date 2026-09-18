"""Persistent CSWAM server for split-environment RoboTwin evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
import signal
import socket
import struct
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cswam_rpc_adapter import get_model


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
RPC_PROTOCOL_VERSION = 2


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("CSWAM RPC peer disconnected.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> Any:
    (size,) = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"CSWAM RPC message is too large: {size} bytes.")
    return pickle.loads(_recv_exact(sock, size))


def _send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"CSWAM RPC response is too large: {len(payload)} bytes.")
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def _find_tensor(state: dict[str, Any], suffixes: tuple[str, ...]) -> torch.Tensor:
    matches = [
        value
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and key.endswith(suffixes)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one checkpoint tensor ending in {suffixes}, got {len(matches)}."
        )
    return matches[0]


def _stats_dimension(stats: dict[str, Any], kind: str, key: str) -> int:
    try:
        field = stats[kind][key]
    except KeyError as exc:
        raise ValueError(f"Stats are missing {kind}.{key}.") from exc
    for name in ("global_mean", "global_std", "global_min", "global_max"):
        value = field.get(name)
        if isinstance(value, list):
            return len(value)
    raise ValueError(f"Stats field {kind}.{key} has no global vector.")


def inspect_checkpoint_contract(
    checkpoint_path: str, config_path: str, stats_path: str
) -> dict[str, Any]:
    """Validate model/config/stats dimensions without constructing the model."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    stats_file = Path(stats_path).expanduser().resolve()
    for path in (checkpoint, config, stats_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(config)
    data_cfg = cfg.data.train if "train" in cfg.data else cfg.data
    resolved = OmegaConf.to_container(data_cfg, resolve=True)
    resolved_model = OmegaConf.to_container(cfg.model, resolve=True)
    processor_cfg = resolved["processor"]
    shape_meta = resolved["shape_meta"]
    raw_action_dim = int(shape_meta["action"][0]["shape"])
    raw_state_dim = int(shape_meta["state"][0]["shape"])
    config_action_dim = int(processor_cfg["action_output_dim"])
    config_state_dim = int(processor_cfg["proprio_output_dim"])

    with stats_file.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    action_key = str(shape_meta["action"][0]["key"])
    state_key = str(shape_meta["state"][0]["key"])
    stats_action_dim = _stats_dimension(stats, "action", action_key)
    stats_state_dim = _stats_dimension(stats, "state", state_key)

    load_kwargs = {"map_location": "cpu"}
    try:
        payload = torch.load(checkpoint, mmap=True, weights_only=False, **load_kwargs)
    except TypeError:
        payload = torch.load(checkpoint, **load_kwargs)
    mot_state = payload.get("mot")
    if not isinstance(mot_state, dict):
        raise ValueError("Checkpoint has no mot state dictionary.")
    action_encoder = _find_tensor(
        mot_state, ("mixtures.action.action_encoder.weight", "action_encoder.weight")
    )
    action_head = _find_tensor(
        mot_state,
        (
            "mixtures.action.head.weight",
            "mixtures.action.head.head.weight",
            "action.head.head.weight",
        ),
    )
    checkpoint_action_input_dim = int(action_encoder.shape[1])
    checkpoint_action_output_dim = int(action_head.shape[0])

    proprio_state = payload.get("proprio_encoder")
    if not isinstance(proprio_state, dict):
        raise ValueError("Checkpoint has no proprio_encoder state dictionary.")
    proprio_weight = _find_tensor(proprio_state, ("weight",))
    checkpoint_state_dim = int(proprio_weight.shape[1])

    metadata = payload.get("cswam")
    metadata_key = "cswam"
    if not isinstance(metadata, dict):
        metadata = payload.get("jepawam01")
        metadata_key = "jepawam01"
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint has neither cswam nor supported legacy metadata.")
    checkpoint_format = metadata.get("format")
    if checkpoint_format not in {"cswam-v1", "jepawam01-v2"}:
        raise ValueError(f"Unsupported CSWAM checkpoint format: {checkpoint_format!r}.")

    expected_pairs = {
        "checkpoint action input vs config": (
            checkpoint_action_input_dim,
            config_action_dim,
        ),
        "checkpoint action output vs config": (
            checkpoint_action_output_dim,
            config_action_dim,
        ),
        "checkpoint proprio vs config": (checkpoint_state_dim, config_state_dim),
        "stats action vs raw data": (stats_action_dim, raw_action_dim),
        "stats state vs raw data": (stats_state_dim, raw_state_dim),
        "history frame count vs config": (
            int(metadata.get("history_num_frames", -1)),
            int(resolved_model["vjepa_history_num_frames"]),
        ),
        "future frame count vs config": (
            int(metadata.get("future_num_frames", -1)),
            int(resolved_model["vjepa_future_num_frames"]),
        ),
        "history offsets vs config": (
            list(metadata.get("history_offsets", [])),
            list(resolved_model["jepa_history_offsets"]),
        ),
        "future offsets vs config": (
            list(metadata.get("future_offsets", [])),
            list(resolved_model["jepa_future_offsets"]),
        ),
        "V-JEPA input size vs config": (
            list(metadata.get("input_size", [])),
            list(resolved_model["vjepa_input_size"]),
        ),
        "V-JEPA patch size vs config": (
            int(metadata.get("patch_size", -1)),
            int(resolved_model["vjepa_patch_size"]),
        ),
        "V-JEPA tubelet size vs config": (
            int(metadata.get("tubelet_size", -1)),
            int(resolved_model["vjepa_tubelet_size"]),
        ),
        "V-JEPA embedding dim vs config": (
            int(metadata.get("embed_dim", -1)),
            int(resolved_model["vjepa_embed_dim"]),
        ),
    }
    mismatches = {
        name: values for name, values in expected_pairs.items() if values[0] != values[1]
    }
    report = {
        "model_family": "cswam",
        "checkpoint": str(checkpoint),
        "config": str(config),
        "stats": str(stats_file),
        "metadata_key": metadata_key,
        "checkpoint_format": checkpoint_format,
        "checkpoint_action_dim": checkpoint_action_output_dim,
        "checkpoint_proprio_dim": checkpoint_state_dim,
        "config_action_dim": config_action_dim,
        "config_proprio_dim": config_state_dim,
        "raw_action_dim": raw_action_dim,
        "raw_proprio_dim": raw_state_dim,
        "stats_action_dim": stats_action_dim,
        "stats_proprio_dim": stats_state_dim,
        "action_horizon": int(resolved["num_frames"]) - 1,
        "history_offsets": list(metadata.get("history_offsets", [])),
    }
    del payload
    if mismatches:
        raise ValueError(f"CSWAM evaluation contract mismatch: {mismatches}; report={report}")
    return report


class CSWAMRpcServer:
    def __init__(self, model: Any, host: str, port: int) -> None:
        self.model = model
        self.host = host
        self.port = int(port)
        self.running = True
        self.server_socket: socket.socket | None = None

    def stop(self, *_args: Any) -> None:
        self.running = False
        if self.server_socket is not None:
            self.server_socket.close()

    def _capabilities(self) -> dict[str, Any]:
        capabilities = dict(self.model.rpc_capabilities())
        capabilities.update(
            {
                "protocol_version": RPC_PROTOCOL_VERSION,
                "supports_reset": True,
                "action_mode": "chunk",
                "model_family": "cswam",
            }
        )
        return capabilities

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "ping":
            return {
                "ok": True,
                "status": "ready",
                "capabilities": self._capabilities(),
            }
        if operation == "reset":
            self.model.rpc_reset()
            return {"ok": True}
        if operation != "act":
            raise ValueError(f"Unsupported CSWAM RPC operation: {operation!r}.")

        if request.get("seed") is not None:
            self.model.seed = int(request["seed"])
        actions = self.model.predict_action_chunk(
            observation=request["observation"],
            instruction=str(request["instruction"]),
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(f"CSWAM returned invalid actions with shape {actions.shape}.")
        return {"ok": True, "actions": actions}

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self.server_socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(8)
            server.settimeout(1.0)
            print(f"[cswam-rpc] ready host={self.host} port={self.port}", flush=True)
            while self.running:
                try:
                    connection, _address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with connection:
                    connection.settimeout(None)
                    while self.running:
                        try:
                            request = _recv_message(connection)
                        except ConnectionError:
                            break
                        try:
                            response = self._handle(request)
                        except Exception as exc:
                            response = {
                                "ok": False,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        _send_message(connection, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--vjepa-repo")
    parser.add_argument("--vjepa-checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--vae-device-mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--text-cache-size", type=int, default=64)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_checkpoint_contract(args.checkpoint, args.config, args.stats)
    print(json.dumps(report, indent=2), flush=True)
    if args.inspect_only:
        return
    if not args.vjepa_repo or not args.vjepa_checkpoint:
        raise ValueError(
            "--vjepa-repo and --vjepa-checkpoint are required unless --inspect-only is used."
        )

    model = get_model(
        {
            "ckpt_setting": args.checkpoint,
            "dataset_stats_path": args.stats,
            "sim_cfg_path": args.config,
            "vjepa_repo": args.vjepa_repo,
            "vjepa_ckpt": args.vjepa_checkpoint,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "num_inference_steps": args.num_inference_steps,
            "rand_device": args.device,
            "text_encoder_device": args.text_encoder_device,
            "vae_device_mode": args.vae_device_mode,
            "text_cache_size": args.text_cache_size,
            "seed": None,
        }
    )
    server = CSWAMRpcServer(model=model, host=args.host, port=args.port)
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve_forever()


if __name__ == "__main__":
    main()
