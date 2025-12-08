"""File-based input sources for video and audio files."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np

from src.core.types import AudioChunk, Frame
from src.inputs.base import InputSource


class VideoFileInput(InputSource):
    """Read frames from a video file.

    Supports any format that OpenCV/ffmpeg can read (mp4, avi, mov, etc.)

    Example:
        video = VideoFileInput("recording.mp4", fps=1.0)
        async for frame in video.stream():
            print(f"Frame at {frame.timestamp}")
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 1.0,
        loop: bool = False,
        auto_resize: bool = True,
        max_size: int = 512,
    ):
        """Initialize video file input.

        Args:
            path: Path to video file
            fps: Output frames per second (samples from video)
            loop: Whether to loop the video
            auto_resize: Whether to resize frames
            max_size: Maximum dimension when auto_resize is True
        """
        self.path = Path(path)
        self.fps = fps
        self.loop = loop
        self.auto_resize = auto_resize
        self.max_size = max_size
        self._cap: cv2.VideoCapture | None = None
        self._running = False

        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

    async def stream(self) -> AsyncIterator[Frame]:
        """Stream frames from the video file."""
        self._cap = cv2.VideoCapture(str(self.path))

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / self.fps))
        frame_count = 0

        self._running = True
        interval = 1.0 / self.fps

        while self._running:
            ret, bgr_frame = await asyncio.get_event_loop().run_in_executor(
                None, self._cap.read
            )

            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1

            # Skip frames to match desired fps
            if frame_count % frame_skip != 0:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            frame = Frame(
                data=rgb_frame,
                timestamp=datetime.now(),
                source=f"file:{self.path.name}",
            )

            if self.auto_resize:
                frame = frame.resize(self.max_size)

            yield frame

            await asyncio.sleep(interval)

    async def close(self) -> None:
        """Release the video capture."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_live(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"VideoFile({self.path.name})"


class AudioFileInput(InputSource):
    """Read audio from an audio file.

    Uses ffmpeg-python to read various audio formats.

    Example:
        audio = AudioFileInput("meeting.wav", chunk_duration=5.0)
        async for chunk in audio.stream():
            print(f"Got {chunk.duration_seconds}s of audio")
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        loop: bool = False,
    ):
        """Initialize audio file input.

        Args:
            path: Path to audio file
            sample_rate: Output sample rate
            chunk_duration: Duration of each chunk in seconds
            loop: Whether to loop the audio
        """
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.loop = loop
        self._running = False

        if not self.path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from the file."""
        import ffmpeg

        self._running = True
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        while self._running:
            # Use ffmpeg to read and resample audio
            try:
                out, _ = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: (
                        ffmpeg.input(str(self.path))
                        .output("pipe:", format="f32le", acodec="pcm_f32le",
                                ac=1, ar=self.sample_rate)
                        .run(capture_stdout=True, capture_stderr=True)
                    ),
                )
            except ffmpeg.Error as e:
                raise RuntimeError(f"Failed to read audio: {e.stderr.decode()}")

            # Convert to numpy array
            audio_data = np.frombuffer(out, dtype=np.float32)

            # Yield in chunks
            for i in range(0, len(audio_data), chunk_samples):
                if not self._running:
                    break

                chunk = audio_data[i : i + chunk_samples]
                if len(chunk) < chunk_samples // 2:
                    break  # Skip very short final chunks

                yield AudioChunk(
                    data=chunk,
                    sample_rate=self.sample_rate,
                    timestamp=datetime.now(),
                    source=f"file:{self.path.name}",
                )

                # Simulate real-time playback speed
                await asyncio.sleep(len(chunk) / self.sample_rate)

            if not self.loop:
                break

    async def close(self) -> None:
        """Stop reading the file."""
        self._running = False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"AudioFile({self.path.name})"


class FileInput(InputSource):
    """Auto-detect and read video or audio from a file.

    Automatically selects VideoFileInput or AudioFileInput based on file type.

    Example:
        source = FileInput("recording.mp4")  # Uses VideoFileInput
        source = FileInput("audio.wav")      # Uses AudioFileInput
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def __init__(self, path: str | Path, **kwargs):
        """Initialize file input.

        Args:
            path: Path to media file
            **kwargs: Additional arguments passed to underlying input
        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = self.path.suffix.lower()

        if ext in self.VIDEO_EXTENSIONS:
            self._source: InputSource = VideoFileInput(path, **kwargs)
        elif ext in self.AUDIO_EXTENSIONS:
            self._source = AudioFileInput(path, **kwargs)
        else:
            # Default to video (OpenCV will try to read it)
            self._source = VideoFileInput(path, **kwargs)

    async def stream(self) -> AsyncIterator[Frame | AudioChunk]:
        """Stream from the detected source type."""
        async for item in self._source.stream():
            yield item

    async def close(self) -> None:
        """Close the underlying source."""
        await self._source.close()

    @property
    def is_live(self) -> bool:
        return self._source.is_live

    @property
    def name(self) -> str:
        return self._source.name
