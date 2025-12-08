"""Base input source protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.core.types import AudioChunk, Frame


class InputSource(ABC):
    """Abstract base class for all input sources.

    Input sources yield either Frame or AudioChunk objects
    through their async stream() method.

    Example:
        async for item in source.stream():
            if isinstance(item, Frame):
                process_frame(item)
            elif isinstance(item, AudioChunk):
                process_audio(item)
    """

    @abstractmethod
    async def stream(self) -> AsyncIterator[Frame | AudioChunk]:
        """Stream frames or audio chunks.

        Yields:
            Frame or AudioChunk objects as they become available.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (camera handles, file handles, etc.)."""
        ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Whether this is a live stream (vs pre-recorded file).

        Live streams continue indefinitely until stopped.
        File sources end when the file is exhausted.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this source."""
        return self.__class__.__name__


class CompositeInput(InputSource):
    """Combine multiple input sources into one stream.

    Useful for combining webcam + microphone, or multiple cameras.

    Example:
        source = CompositeInput(
            WebcamInput(device_id=0),
            MicrophoneInput()
        )
        async for item in source.stream():
            # Items from both sources interleaved
            ...
    """

    def __init__(self, *sources: InputSource):
        self.sources = list(sources)
        self._running = False

    async def stream(self) -> AsyncIterator[Frame | AudioChunk]:
        """Stream from all sources concurrently."""
        import asyncio

        self._running = True
        queues: list[asyncio.Queue[Frame | AudioChunk | None]] = [
            asyncio.Queue() for _ in self.sources
        ]

        async def feed_queue(source: InputSource, queue: asyncio.Queue) -> None:
            try:
                async for item in source.stream():
                    if not self._running:
                        break
                    await queue.put(item)
            finally:
                await queue.put(None)  # Signal completion

        # Start all source tasks
        tasks = [
            asyncio.create_task(feed_queue(src, q))
            for src, q in zip(self.sources, queues)
        ]

        # Yield items as they arrive
        active_queues = set(range(len(queues)))
        while active_queues and self._running:
            for i in list(active_queues):
                try:
                    item = queues[i].get_nowait()
                    if item is None:
                        active_queues.discard(i)
                    else:
                        yield item
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

        # Clean up
        for task in tasks:
            task.cancel()

    async def close(self) -> None:
        """Close all sources."""
        self._running = False
        for source in self.sources:
            await source.close()

    @property
    def is_live(self) -> bool:
        """Live if any source is live."""
        return any(s.is_live for s in self.sources)

    @property
    def name(self) -> str:
        return f"Composite({', '.join(s.name for s in self.sources)})"
