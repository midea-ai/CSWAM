from __future__ import annotations

from collections import defaultdict

import torch


class CudaEventTrace:
    """Record named, consecutive CUDA intervals without synchronizing."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._names: list[str] = []
        self._events: list[torch.cuda.Event] = []
        if self.enabled:
            self._events.append(self._record_event())

    @staticmethod
    def _record_event() -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        self._names.append(str(name))
        self._events.append(self._record_event())

    def elapsed_ms(self) -> dict[str, float]:
        """Return summed milliseconds per name after the caller synchronizes."""
        elapsed: dict[str, float] = defaultdict(float)
        for name, start, end in zip(self._names, self._events, self._events[1:]):
            elapsed[name] += float(start.elapsed_time(end))
        return dict(elapsed)

    @property
    def has_intervals(self) -> bool:
        return bool(self._names)
