"""Core agent loop - the heart of the multimodal agent framework.

This module implements the main agent loop in ~150 lines, providing:
- Async processing of video frames and audio chunks
- Unified interface for any vision-language model
- Structured tool calling with automatic execution
- Sliding window memory management
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Protocol

from src.core.config import AgentConfig
from src.core.types import (
    AgentEvent,
    AudioChunk,
    Frame,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

if TYPE_CHECKING:
    from src.memory.sliding_window import SlidingWindowMemory


class InputSource(Protocol):
    """Protocol for all input sources (webcam, mic, files, etc.)."""

    async def stream(self) -> AsyncIterator[Frame | AudioChunk]:
        """Yield frames or audio chunks."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


class VisionLanguageModel(Protocol):
    """Protocol for vision-language models (GPT-4o, Claude, Gemini, etc.)."""

    async def analyze(
        self,
        frames: list[Frame],
        audio_transcript: str | None,
        tools: list[ToolDefinition],
        context: list[Message],
        system_prompt: str,
    ) -> AsyncIterator[AgentEvent]:
        """Analyze frames/audio and yield events (messages, tool calls)."""
        ...


class Tool(Protocol):
    """Protocol for executable tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        ...


class Memory(Protocol):
    """Protocol for context memory management."""

    def add(self, event: AgentEvent) -> None:
        """Add an event to memory."""
        ...

    def get_context(self, max_tokens: int) -> list[Message]:
        """Get recent context up to max_tokens."""
        ...

    def clear(self) -> None:
        """Clear all memory."""
        ...


@dataclass
class AgentLoop:
    """The core multimodal agent loop.

    This is a minimal, battle-tested design that:
    - Processes frames/audio from any input source
    - Sends to any vision-language model
    - Executes tool calls automatically
    - Manages conversation context

    Example:
        agent = AgentLoop(model=my_model, memory=my_memory)
        agent.register_tool(SlackAlertTool(...))
        await agent.run(WebcamInput())
    """

    model: VisionLanguageModel
    memory: Memory
    config: AgentConfig = field(default_factory=AgentConfig)
    tools: dict[str, Tool] = field(default_factory=dict)

    # Internal state
    _frame_buffer: list[Frame] = field(default_factory=list, init=False)
    _audio_buffer: str = field(default="", init=False)
    _running: bool = field(default=False, init=False)
    _transcriber: Any = field(default=None, init=False)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool

    def set_transcriber(self, transcriber: Any) -> None:
        """Set the audio transcriber (e.g., Whisper)."""
        self._transcriber = transcriber

    async def run(
        self,
        input_source: InputSource,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> None:
        """Run the agent loop on an input source.

        Args:
            input_source: Source of frames/audio (webcam, file, etc.)
            on_event: Optional callback for each event produced
        """
        self._running = True
        last_process_time = datetime.now()

        try:
            async for item in input_source.stream():
                if not self._running:
                    break

                # Buffer frames or transcribe audio
                if isinstance(item, Frame):
                    self._frame_buffer.append(item)
                elif isinstance(item, AudioChunk):
                    transcript = await self._transcribe(item)
                    self._audio_buffer += transcript

                # Check if we should process
                elapsed_ms = (datetime.now() - last_process_time).total_seconds() * 1000
                if self._should_process(elapsed_ms):
                    async for event in self._process_buffer():
                        if on_event:
                            await on_event(event)
                        # Handle tool calls
                        if isinstance(event, ToolCall):
                            result = await self._execute_tool(event)
                            self.memory.add(result)
                            if on_event:
                                await on_event(result)
                    last_process_time = datetime.now()
        finally:
            await input_source.close()

    async def process_once(
        self,
        frames: list[Frame] | None = None,
        audio_transcript: str | None = None,
    ) -> list[AgentEvent]:
        """Process frames/audio once without running the loop.

        Useful for one-shot analysis or testing.
        """
        if frames:
            self._frame_buffer = frames
        if audio_transcript:
            self._audio_buffer = audio_transcript

        events = []
        async for event in self._process_buffer():
            events.append(event)
            if isinstance(event, ToolCall):
                result = await self._execute_tool(event)
                events.append(result)
                self.memory.add(result)
        return events

    async def _process_buffer(self) -> AsyncIterator[AgentEvent]:
        """Process buffered frames and audio through the model."""
        if not self._frame_buffer and not self._audio_buffer:
            return

        context = self.memory.get_context(self.config.max_context_tokens)
        tool_defs = [self._tool_to_definition(t) for t in self.tools.values()]

        frames_to_send = self._frame_buffer[-self.config.max_frames :]

        async for event in self.model.analyze(
            frames=frames_to_send,
            audio_transcript=self._audio_buffer if self._audio_buffer else None,
            tools=tool_defs,
            context=context,
            system_prompt=self.config.system_prompt,
        ):
            self.memory.add(event)
            yield event

        # Clear buffers after processing
        self._frame_buffer.clear()
        self._audio_buffer = ""

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool and return the result."""
        tool = self.tools.get(tool_call.name)
        if not tool:
            return ToolResult(error=f"Unknown tool: {tool_call.name}")

        try:
            result = await asyncio.wait_for(
                tool.execute(**tool_call.arguments),
                timeout=self.config.tool_timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            return ToolResult(error=f"Tool {tool_call.name} timed out")
        except Exception as e:
            return ToolResult(error=f"Tool {tool_call.name} failed: {e}")

    async def _transcribe(self, chunk: AudioChunk) -> str:
        """Transcribe an audio chunk to text."""
        if self._transcriber is None:
            return ""
        return await self._transcriber.transcribe(chunk)

    def _should_process(self, elapsed_ms: float) -> bool:
        """Determine if we should process the buffer."""
        # Check frame batch size
        if len(self._frame_buffer) >= self.config.frame_batch_size:
            # Also check minimum interval
            if elapsed_ms >= self.config.frame_interval_ms:
                return True

        # Check audio buffer
        if len(self._audio_buffer) >= self.config.min_audio_chars:
            return True

        return False

    def _tool_to_definition(self, tool: Tool) -> ToolDefinition:
        """Convert a tool to a ToolDefinition."""
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
