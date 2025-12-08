"""Microphone input source using sounddevice."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

import numpy as np

from src.core.types import AudioChunk
from src.inputs.base import InputSource


class MicrophoneInput(InputSource):
    """Capture audio from microphone using sounddevice.

    Example:
        mic = MicrophoneInput(sample_rate=16000, chunk_duration=5.0)
        async for chunk in mic.stream():
            print(f"Got {chunk.duration_seconds}s of audio")
    """

    def __init__(
        self,
        device_id: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0,
    ):
        """Initialize microphone input.

        Args:
            device_id: Audio device ID (None for default)
            sample_rate: Sample rate in Hz (16000 recommended for speech)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self._running = False
        self._stream = None

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from the microphone."""
        import sounddevice as sd

        chunk_size = int(self.sample_rate * self.chunk_duration)
        self._running = True

        # Use a queue to pass data from callback
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            if self._running:
                # Put a copy of the data in the queue
                asyncio.get_event_loop().call_soon_threadsafe(
                    queue.put_nowait, indata.copy()
                )

        # Start the audio stream
        self._stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            blocksize=chunk_size,
        )

        self._stream.start()

        try:
            while self._running:
                try:
                    # Wait for audio data with timeout
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)

                    # Flatten if needed and convert to float32
                    audio_data = data.flatten().astype(np.float32)

                    yield AudioChunk(
                        data=audio_data,
                        sample_rate=self.sample_rate,
                        timestamp=datetime.now(),
                        source=f"microphone:{self.device_id or 'default'}",
                    )

                except asyncio.TimeoutError:
                    continue

        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()

    async def close(self) -> None:
        """Stop the microphone stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_live(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"Microphone({self.device_id or 'default'})"
