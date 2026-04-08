#!/usr/bin/env python3
"""
shm_camera.py — Thin, quiet wrappers around SharedMemoryManager.

SharedMemoryManager.py prints debug messages on every read/write call.
This module suppresses all that output and provides two clean classes:

    SHMProducer  — owns a shared memory stream, pushes BGR numpy frames
    SHMConsumer  — subscribes to a stream, returns BGR numpy frames

Frames are stored as RGB internally (matching the existing client_upstream.py
convention); the wrappers handle the BGR↔RGB conversion transparently.
"""

import contextlib
import io
import os
import sys
import time

import cv2
import numpy as np


# ── internal helpers ──────────────────────────────────────────────────────────

def _ensure_path(lib_dir: str) -> None:
    """Add lib_dir to sys.path once so SharedMemoryManager can be imported."""
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)


def _quiet(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with Python-level stdout suppressed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _lib_path(lib_dir: str) -> str:
    return os.path.join(lib_dir, "libSharedMemoryVideoBuffers.so")


def _check_lib(lib_dir: str) -> None:
    path = _lib_path(lib_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"libSharedMemoryVideoBuffers.so not found in '{lib_dir}'.\n"
            f"Set shared_memory.lib_dir in configuration.json to the "
            f"SharedMemoryVideoBuffers/src/python/ directory."
        )


# ── Producer ──────────────────────────────────────────────────────────────────

class SHMProducer:
    """Create a named shared memory video stream and push BGR frames into it.

    Only ONE producer should exist for a given stream_name at a time.

    Args:
        lib_dir:     Path to the directory containing libSharedMemoryVideoBuffers.so
                     and SharedMemoryManager.py.
        descriptor:  Shared memory file descriptor name (e.g. "voithos_video.shm").
        stream_name: Logical stream name (e.g. "voithos_cam").
        width, height: Frame dimensions in pixels.
        channels:    Number of channels (default 3 = RGB/BGR).
    """

    def __init__(self, lib_dir: str, descriptor: str, stream_name: str,
                 width: int, height: int, channels: int = 3):
        _check_lib(lib_dir)
        _ensure_path(lib_dir)
        from SharedMemoryManager import SharedMemoryManager  # noqa: PLC0415

        self._smm = _quiet(
            SharedMemoryManager,
            _lib_path(lib_dir),
            descriptor=descriptor,
            frameName=stream_name,
            connect=False,
            width=width,
            height=height,
            channels=channels,
        )

    def push(self, bgr_frame: np.ndarray) -> None:
        """Write *bgr_frame* (H×W×3 uint8, BGR) to shared memory."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        _quiet(self._smm.copy_numpy_to_shared_memory, rgb)


# ── Consumer ──────────────────────────────────────────────────────────────────

class SHMConsumer:
    """Read BGR frames from a shared memory stream.

    Connection is attempted lazily inside get_frame() so that the consumer
    can be constructed before the producer is running.

    Args:
        lib_dir:     Path to the directory with the .so and SharedMemoryManager.py.
        descriptor:  Shared memory descriptor (must match the producer).
        stream_name: Logical stream name (must match the producer).
        retry_interval: Seconds to wait between connection retries (default 1).
    """

    def __init__(self, lib_dir: str, descriptor: str, stream_name: str,
                 retry_interval: float = 1.0):
        _check_lib(lib_dir)
        _ensure_path(lib_dir)
        self._lib_dir        = lib_dir
        self._descriptor     = descriptor
        self._stream_name    = stream_name
        self._retry_interval = retry_interval
        self._smm            = None   # connected lazily
        self._last_attempt   = 0.0

    def _try_connect(self) -> bool:
        now = time.monotonic()
        if now - self._last_attempt < self._retry_interval:
            return False
        self._last_attempt = now
        try:
            from SharedMemoryManager import SharedMemoryManager  # noqa: PLC0415
            smm = _quiet(
                SharedMemoryManager,
                _lib_path(self._lib_dir),
                descriptor=self._descriptor,
                frameName=self._stream_name,
                connect=True,
            )
            self._smm = smm
            return True
        except Exception as e:
            print(f"[shm_camera] Waiting for producer ({e})", flush=True)
            return False

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame as a BGR uint8 ndarray, or None if unavailable."""
        if self._smm is None:
            if not self._try_connect():
                return None

        try:
            raw = _quiet(self._smm.read_from_shared_memory)
        except Exception:
            # Lost connection — reset so we reconnect next call
            self._smm = None
            return None

        if raw is None or raw.size == 0:
            return None

        # SHM stores RGB; convert to OpenCV BGR
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
