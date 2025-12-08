"""RTSP stream input source for IP cameras."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

import cv2

from src.core.types import Frame
from src.inputs.base import InputSource


class RTSPInput(InputSource):
    """Capture frames from an RTSP stream (IP cameras).

    Supports automatic reconnection if the stream drops.

    Example:
        # Connect to an IP camera
        camera = RTSPInput(
            url="rtsp://admin:password@192.168.1.100:554/stream",
            fps=1.0
        )
        async for frame in camera.stream():
            print(f"Got frame from IP camera")
    """

    def __init__(
        self,
        url: str,
        fps: float = 1.0,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
        auto_resize: bool = True,
        max_size: int = 512,
        timeout_seconds: float = 10.0,
    ):
        """Initialize RTSP input.

        Args:
            url: RTSP URL (e.g., rtsp://user:pass@host:554/stream)
            fps: Frames per second to capture
            auto_reconnect: Whether to reconnect on stream failure
            reconnect_delay: Seconds to wait before reconnecting
            max_reconnect_attempts: Maximum reconnection attempts (0 = unlimited)
            auto_resize: Whether to resize frames
            max_size: Maximum dimension when auto_resize is True
            timeout_seconds: Connection timeout
        """
        self.url = url
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.auto_resize = auto_resize
        self.max_size = max_size
        self.timeout_seconds = timeout_seconds

        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._reconnect_count = 0

    async def stream(self) -> AsyncIterator[Frame]:
        """Stream frames from the RTSP source."""
        self._running = True
        interval = 1.0 / self.fps

        while self._running:
            # Connect if not connected
            if self._cap is None or not self._cap.isOpened():
                connected = await self._connect()
                if not connected:
                    if self.auto_reconnect:
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                    else:
                        break

            # Read frame
            ret, bgr_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._cap.read
            )

            if not ret:
                print(f"RTSP stream read failed, reconnecting...")
                await self._disconnect()

                if self.auto_reconnect:
                    continue
                else:
                    break

            # Reset reconnect counter on successful read
            self._reconnect_count = 0

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            frame = Frame(
                data=rgb_frame,
                timestamp=datetime.now(),
                source=f"rtsp:{self._sanitized_url}",
            )

            if self.auto_resize:
                frame = frame.resize(self.max_size)

            yield frame

            await asyncio.sleep(interval)

    async def _connect(self) -> bool:
        """Attempt to connect to the RTSP stream."""
        if self.max_reconnect_attempts > 0:
            if self._reconnect_count >= self.max_reconnect_attempts:
                print(f"Max reconnection attempts reached ({self.max_reconnect_attempts})")
                return False

        self._reconnect_count += 1
        print(f"Connecting to RTSP stream (attempt {self._reconnect_count})...")

        # Set up VideoCapture with RTSP-specific options
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        # Configure for RTSP
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Wait for connection with timeout
        start_time = asyncio.get_event_loop().time()
        while not self._cap.isOpened():
            if asyncio.get_event_loop().time() - start_time > self.timeout_seconds:
                print(f"RTSP connection timeout after {self.timeout_seconds}s")
                return False
            await asyncio.sleep(0.1)

        print(f"Connected to RTSP stream")
        return True

    async def _disconnect(self) -> None:
        """Disconnect from the RTSP stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    async def close(self) -> None:
        """Stop streaming and release resources."""
        self._running = False
        await self._disconnect()

    @property
    def is_live(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"RTSP({self._sanitized_url})"

    @property
    def _sanitized_url(self) -> str:
        """Return URL with credentials hidden."""
        # Hide password in URL for logging
        import re
        return re.sub(r"://[^:]+:[^@]+@", "://***:***@", self.url)
