"""Webcam input source using OpenCV."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import AsyncIterator

# Suppress macOS AVFoundation deprecation warning
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import numpy as np

from src.core.types import Frame
from src.inputs.base import InputSource


class WebcamInput(InputSource):
    """Capture frames from a webcam using OpenCV.

    Example:
        webcam = WebcamInput(device_id=0, fps=1)
        async for frame in webcam.stream():
            print(f"Got frame: {frame.shape}")
    """

    def __init__(
        self,
        device_id: int = 0,
        fps: float = 1.0,
        resolution: tuple[int, int] = (640, 480),
        auto_resize: bool = True,
        max_size: int = 512,
    ):
        """Initialize webcam input.

        Args:
            device_id: Camera device ID (0 for default camera)
            fps: Frames per second to capture (1.0 = one frame per second)
            resolution: Capture resolution (width, height)
            auto_resize: Whether to resize frames for API efficiency
            max_size: Maximum dimension when auto_resize is True
        """
        self.device_id = device_id
        self.fps = fps
        self.resolution = resolution
        self.auto_resize = auto_resize
        self.max_size = max_size
        self._cap: cv2.VideoCapture | None = None
        self._running = False

    async def stream(self) -> AsyncIterator[Frame]:
        """Stream frames from the webcam."""
        self._cap = cv2.VideoCapture(self.device_id)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.device_id}")

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self._running = True
        interval = 1.0 / self.fps

        while self._running:
            # Read frame in thread pool to avoid blocking
            ret, bgr_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._cap.read
            )

            if not ret:
                continue

            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            frame = Frame(
                data=rgb_frame,
                timestamp=datetime.now(),
                source=f"webcam:{self.device_id}",
            )

            # Resize if needed
            if self.auto_resize:
                frame = frame.resize(self.max_size)

            yield frame

            await asyncio.sleep(interval)

    async def close(self) -> None:
        """Release the webcam."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_live(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"Webcam({self.device_id})"
