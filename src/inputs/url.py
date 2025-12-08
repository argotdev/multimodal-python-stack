"""URL-based input source for remote images and videos."""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image

from src.core.types import Frame
from src.inputs.base import InputSource


class URLInput(InputSource):
    """Fetch images or video frames from URLs.

    Supports:
    - Single images (PNG, JPEG, WebP, etc.)
    - Video URLs (will extract frames)
    - Repeated polling of URLs that update

    Example:
        # Single image
        source = URLInput("https://example.com/camera.jpg")

        # Polling a webcam image URL every 5 seconds
        source = URLInput(
            "https://example.com/live/snapshot.jpg",
            poll_interval=5.0,
            repeat=True
        )
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        url: str,
        poll_interval: float = 0.0,
        repeat: bool = False,
        auto_resize: bool = True,
        max_size: int = 512,
        timeout_seconds: float = 30.0,
        headers: dict[str, str] | None = None,
    ):
        """Initialize URL input.

        Args:
            url: URL to fetch (image or video)
            poll_interval: Seconds between fetches (0 = no polling)
            repeat: Whether to repeatedly fetch the URL
            auto_resize: Whether to resize frames
            max_size: Maximum dimension when auto_resize is True
            timeout_seconds: HTTP request timeout
            headers: Optional HTTP headers
        """
        self.url = url
        self.poll_interval = poll_interval
        self.repeat = repeat
        self.auto_resize = auto_resize
        self.max_size = max_size
        self.timeout_seconds = timeout_seconds
        self.headers = headers or {}

        self._running = False
        self._client: httpx.AsyncClient | None = None

    async def stream(self) -> AsyncIterator[Frame]:
        """Stream frames from the URL."""
        self._running = True
        self._client = httpx.AsyncClient(timeout=self.timeout_seconds)

        try:
            # Detect content type
            parsed = urlparse(self.url)
            path = Path(parsed.path)
            ext = path.suffix.lower()

            if ext in self.VIDEO_EXTENSIONS:
                # For videos, download and extract frames
                async for frame in self._stream_video():
                    yield frame
            else:
                # For images, fetch directly
                async for frame in self._stream_image():
                    yield frame

        finally:
            await self._client.aclose()

    async def _stream_image(self) -> AsyncIterator[Frame]:
        """Fetch and yield image frames."""
        while self._running:
            try:
                response = await self._client.get(self.url, headers=self.headers)
                response.raise_for_status()

                # Load image from bytes
                image_data = response.content
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                rgb_array = np.array(img)

                frame = Frame(
                    data=rgb_array,
                    timestamp=datetime.now(),
                    source=f"url:{self._shortened_url}",
                )

                if self.auto_resize:
                    frame = frame.resize(self.max_size)

                yield frame

            except httpx.HTTPError as e:
                print(f"HTTP error fetching {self._shortened_url}: {e}")

            except Exception as e:
                print(f"Error processing image from {self._shortened_url}: {e}")

            # Handle repeat/polling
            if not self.repeat:
                break

            if self.poll_interval > 0:
                await asyncio.sleep(self.poll_interval)
            else:
                break

    async def _stream_video(self) -> AsyncIterator[Frame]:
        """Download video and extract frames."""
        import tempfile
        import cv2

        try:
            # Download video to temp file
            response = await self._client.get(self.url, headers=self.headers)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            # Extract frames using OpenCV
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(fps))  # Sample ~1 frame per second

            frame_count = 0
            while self._running and cap.isOpened():
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                frame = Frame(
                    data=rgb_frame,
                    timestamp=datetime.now(),
                    source=f"url:{self._shortened_url}",
                )

                if self.auto_resize:
                    frame = frame.resize(self.max_size)

                yield frame

                # Simulate real-time playback
                await asyncio.sleep(1.0 / (fps / frame_skip))

            cap.release()

            # Clean up temp file
            import os
            os.unlink(temp_path)

        except httpx.HTTPError as e:
            print(f"HTTP error fetching video {self._shortened_url}: {e}")

        except Exception as e:
            print(f"Error processing video from {self._shortened_url}: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        self._running = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def is_live(self) -> bool:
        # Live if we're polling repeatedly
        return self.repeat and self.poll_interval > 0

    @property
    def name(self) -> str:
        return f"URL({self._shortened_url})"

    @property
    def _shortened_url(self) -> str:
        """Shortened URL for display."""
        parsed = urlparse(self.url)
        path = parsed.path
        if len(path) > 30:
            path = "..." + path[-27:]
        return f"{parsed.netloc}{path}"
