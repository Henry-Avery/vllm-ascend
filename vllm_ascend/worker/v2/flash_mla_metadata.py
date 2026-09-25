# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 graph-external metadata submission and buffer reuse fences.

Adapted from the target-only MRV2 lifecycle in reference de31c53d. MRV2
waits outside capture, so this path needs ordinary events, not ExternalEvent.
"""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceMetadataTask:
    run: Callable[[], None]
    group_id: int


class DeviceMetadataExecutor:
    """Keep metadata buffers owned until their model consumer is queued."""

    def __init__(self):
        self.stream = torch.npu.Stream()
        self._inputs_ready = torch.npu.Event()
        self._buffer_reusable = torch.npu.Event()
        self._ready: dict[int, torch.npu.Event] = {}
        self._waited: set[int] = set()
        self._groups: set[int] = set()
        self._has_reuse_fence = False
        self.submission_in_flight = False

    def submit(self, tasks: Iterable[DeviceMetadataTask]):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Device metadata submission must stay outside graph capture")
        if self.submission_in_flight:
            raise RuntimeError("The previous device metadata submission has not been released")
        tasks = tuple(tasks)
        if not tasks:
            raise ValueError("At least one device metadata task is required")
        self._groups = {task.group_id for task in tasks}
        if len(self._groups) != len(tasks):
            raise ValueError("Device metadata groups must be unique per submission")
        for group in self._groups:
            if group not in self._ready:
                self._ready[group] = torch.npu.Event()
        self._waited.clear()
        self.submission_in_flight = True
        self._inputs_ready.record(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self._inputs_ready)
            if self._has_reuse_fence:
                self.stream.wait_event(self._buffer_reusable)
            for task in tasks:
                task.run()
                self._ready[task.group_id].record(self.stream)

    def wait(self, group_id: int):
        if not self.submission_in_flight or group_id not in self._groups:
            raise RuntimeError("No device metadata submission for this group")
        if group_id not in self._waited:
            torch.npu.current_stream().wait_event(self._ready[group_id])
            self._waited.add(group_id)

    def release(self):
        if not self.submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        stream = torch.npu.current_stream()
        if self._waited != self._groups:
            # A task may throw after queuing work but before recording ready.
            # Join that partial submission as well before allowing reuse.
            stream.wait_stream(self.stream)
        self._buffer_reusable.record(stream)
        self._has_reuse_fence = True
        self.submission_in_flight = False


_device_metadata_executor: ContextVar[DeviceMetadataExecutor | None] = ContextVar(
    "ascend_mrv2_device_metadata_executor", default=None
)


def get_device_metadata_executor() -> DeviceMetadataExecutor | None:
    return _device_metadata_executor.get()


@contextmanager
def device_metadata_context(executor: DeviceMetadataExecutor | None):
    if executor is None or get_device_metadata_executor() is executor:
        yield
        return
    token = _device_metadata_executor.set(executor)
    try:
        yield
    finally:
        try:
            if executor.submission_in_flight:
                executor.release()
        finally:
            _device_metadata_executor.reset(token)
