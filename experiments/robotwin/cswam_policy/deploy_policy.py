"""Simulator-side CSWAM policy with sparse temporal observation capture."""

from __future__ import annotations

import pickle
import socket
import struct
import time
from collections import deque
from typing import Any

import numpy as np
from PIL import Image


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("CSWAM RPC server disconnected.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"CSWAM RPC request is too large: {len(payload)} bytes.")
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def _recv_message(sock: socket.socket) -> Any:
    (size,) = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"CSWAM RPC response is too large: {size} bytes.")
    return pickle.loads(_recv_exact(sock, size))


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(
        rgb.resize(size_wh, resample=resampling.BILINEAR), dtype=np.uint8
    )


def build_mosaic(observation: dict[str, Any]) -> np.ndarray:
    cameras = observation["observation"]
    head = _resize_rgb(cameras["head_camera"]["rgb"], (320, 256))
    left = _resize_rgb(cameras["left_camera"]["rgb"], (160, 128))
    right = _resize_rgb(cameras["right_camera"]["rgb"], (160, 128))
    wrists = np.concatenate([left, right], axis=1)
    return np.ascontiguousarray(np.concatenate([head, wrists], axis=0))


class CSWAMPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int,
        model_seed: int | None,
        connect_timeout_s: float,
        request_timeout_s: float,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.replan_steps = int(replan_steps)
        self.model_seed = model_seed
        self.request_timeout_s = float(request_timeout_s)
        self.sock, self.capabilities = self._connect(float(connect_timeout_s))

        if self.capabilities.get("model_family") != "cswam":
            raise ValueError(
                "The RPC endpoint is not a CSWAM server: "
                f"{self.capabilities.get('model_family')!r}."
            )
        self.action_horizon = int(self.capabilities["action_horizon"])
        self.observation_stride = int(self.capabilities["observation_stride"])
        self.history_offsets = tuple(
            int(value) for value in self.capabilities["history_offsets"]
        )
        self.history_num_frames = int(self.capabilities["history_num_frames"])
        if self.replan_steps <= 0 or self.replan_steps > self.action_horizon:
            raise ValueError(
                f"replan_steps must be in [1,{self.action_horizon}], "
                f"got {self.replan_steps}."
            )
        if self.observation_stride <= 0:
            raise ValueError(
                f"observation_stride must be positive, got {self.observation_stride}."
            )
        if self.replan_steps % self.observation_stride != 0:
            raise ValueError(
                f"replan_steps={self.replan_steps} must be divisible by "
                f"observation_stride={self.observation_stride}."
            )
        if len(self.history_offsets) != self.history_num_frames:
            raise ValueError("CSWAM history metadata is inconsistent.")
        if any(offset % self.observation_stride for offset in self.history_offsets):
            raise ValueError(
                "CSWAM history offsets must align with observation_stride: "
                f"{self.history_offsets}."
            )

        self.history_sample_offsets = tuple(
            offset // self.observation_stride for offset in self.history_offsets
        )
        self.history_frames: deque[np.ndarray] = deque(
            maxlen=1 - min(self.history_sample_offsets)
        )
        self.control_step = 0
        self.last_observation_step: int | None = None
        print(
            "[cswam-policy] "
            f"connected={self.host}:{self.port} "
            f"generate={self.action_horizon} replan={self.replan_steps} "
            f"history_offsets={self.history_offsets} "
            f"observation_stride={self.observation_stride}",
            flush=True,
        )

    def _connect(self, timeout_s: float) -> tuple[socket.socket, dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.settimeout(self.request_timeout_s)
                _send_message(sock, {"op": "ping"})
                response = _recv_message(sock)
                if not response.get("ok"):
                    raise RuntimeError(f"CSWAM RPC health check failed: {response}")
                capabilities = dict(response.get("capabilities", {}))
                if int(capabilities.get("protocol_version", 0)) < 2:
                    raise RuntimeError("CSWAM RPC protocol version 2 or newer is required.")
                if capabilities.get("supports_reset") is not True:
                    raise RuntimeError("CSWAM RPC server does not support reset.")
                return sock, capabilities
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            f"Timed out connecting to CSWAM RPC server at {self.host}:{self.port}: "
            f"{last_error}"
        )

    def _append_observation(self, observation: dict[str, Any]) -> None:
        if self.last_observation_step == self.control_step:
            return
        self.history_frames.append(build_mosaic(observation))
        self.last_observation_step = self.control_step

    def _sample_history(self) -> np.ndarray:
        if not self.history_frames:
            raise RuntimeError("CSWAM history is empty.")
        frames = list(self.history_frames)
        latest = len(frames) - 1
        selected = [
            frames[max(0, latest + offset)]
            for offset in self.history_sample_offsets
        ]
        return np.ascontiguousarray(np.stack(selected, axis=0))

    def _request_actions(
        self, observation: dict[str, Any], instruction: str
    ) -> np.ndarray:
        self._append_observation(observation)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        request = {
            "op": "act",
            "instruction": str(instruction),
            "seed": self.model_seed,
            "observation": {
                "mosaic": self.history_frames[-1],
                "jepa_history_video": self._sample_history(),
                "joint_action": {"vector": state},
            },
        }
        _send_message(self.sock, request)
        response = _recv_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(
                f"CSWAM RPC inference failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid CSWAM action chunk with shape {actions.shape}.")
        if len(actions) < self.replan_steps:
            raise ValueError(
                f"CSWAM returned {len(actions)} actions, fewer than "
                f"replan_steps={self.replan_steps}."
            )
        return actions

    def run_chunk(self, task_env: Any, observation: dict[str, Any]) -> None:
        actions = self._request_actions(observation, task_env.get_instruction())
        for index, action in enumerate(actions[: self.replan_steps]):
            task_env.take_action(action, action_type="qpos")
            self.control_step += 1
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break
            is_sample_step = self.control_step % self.observation_stride == 0
            is_chunk_boundary = index + 1 == self.replan_steps
            if is_sample_step and not is_chunk_boundary:
                self._append_observation(task_env.get_obs())

    def reset(self) -> None:
        self.history_frames.clear()
        self.control_step = 0
        self.last_observation_step = None
        _send_message(self.sock, {"op": "reset"})
        response = _recv_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(
                f"CSWAM RPC reset failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )


def get_model(usr_args: dict[str, Any]) -> CSWAMPolicy:
    model_seed = usr_args.get("model_seed")
    return CSWAMPolicy(
        host=str(usr_args.get("model_host", "127.0.0.1")),
        port=int(usr_args.get("model_port", 8772)),
        replan_steps=int(usr_args.get("replan_steps", 24)),
        model_seed=None if model_seed is None else int(model_seed),
        connect_timeout_s=float(usr_args.get("connect_timeout_s", 900)),
        request_timeout_s=float(usr_args.get("request_timeout_s", 1800)),
    )


def eval(task_env: Any, model: CSWAMPolicy, observation: dict[str, Any]) -> None:
    model.run_chunk(task_env, observation)


def reset_model(model: CSWAMPolicy) -> None:
    model.reset()
