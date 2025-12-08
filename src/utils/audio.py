"""Audio transcription utilities using OpenAI Whisper API."""

from __future__ import annotations

import io
import os
import wave
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from src.core.types import AudioChunk


class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper API.

    This class provides audio transcription capabilities for
    the multimodal agent pipeline.

    Example:
        transcriber = WhisperTranscriber()

        # Transcribe an audio chunk
        chunk = AudioChunk(data=audio_array, sample_rate=16000)
        text = await transcriber.transcribe(chunk)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        language: str | None = None,
    ):
        """Initialize Whisper transcriber.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Whisper model to use (currently only "whisper-1")
            language: Optional language hint (ISO-639-1 code)
        """
        self.model = model
        self.language = language
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def transcribe(
        self,
        chunk: AudioChunk,
        prompt: str | None = None,
    ) -> str:
        """Transcribe an audio chunk to text.

        Args:
            chunk: Audio chunk to transcribe
            prompt: Optional prompt to guide transcription

        Returns:
            Transcribed text
        """
        # Convert audio chunk to WAV format in memory
        wav_buffer = self._to_wav_buffer(chunk)

        # Prepare transcription arguments
        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": ("audio.wav", wav_buffer, "audio/wav"),
        }

        if self.language:
            kwargs["language"] = self.language

        if prompt:
            kwargs["prompt"] = prompt

        # Call Whisper API
        transcript = await self.client.audio.transcriptions.create(**kwargs)

        return transcript.text

    async def transcribe_with_timestamps(
        self,
        chunk: AudioChunk,
    ) -> dict[str, Any]:
        """Transcribe with word-level timestamps.

        Args:
            chunk: Audio chunk to transcribe

        Returns:
            Dictionary with text and timestamps
        """
        wav_buffer = self._to_wav_buffer(chunk)

        transcript = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", wav_buffer, "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        return {
            "text": transcript.text,
            "words": transcript.words if hasattr(transcript, "words") else [],
            "duration": transcript.duration if hasattr(transcript, "duration") else None,
        }

    def _to_wav_buffer(self, chunk: AudioChunk) -> io.BytesIO:
        """Convert AudioChunk to WAV format in memory.

        Args:
            chunk: Audio chunk to convert

        Returns:
            BytesIO buffer containing WAV data
        """
        buffer = io.BytesIO()

        # Ensure audio is in correct format
        audio_data = chunk.data

        # Convert float32 to int16 if needed
        if audio_data.dtype == np.float32:
            # Normalize and convert to int16
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        # Write WAV file
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(chunk.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer


class LocalWhisperTranscriber:
    """Transcribe audio using local Whisper model (stub).

    For offline use, you can integrate with local Whisper:
    - openai-whisper (official)
    - faster-whisper (optimized)
    - whisper.cpp (CPU optimized)

    This is a stub showing the interface.
    """

    def __init__(self, model_size: str = "base"):
        """Initialize local Whisper.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self._model = None

        # In production:
        # import whisper
        # self._model = whisper.load_model(model_size)

    async def transcribe(self, chunk: AudioChunk, prompt: str | None = None) -> str:
        """Transcribe using local model."""
        if self._model is None:
            return "[Local Whisper not loaded - stub implementation]"

        # In production:
        # audio_np = chunk.data.astype(np.float32)
        # result = self._model.transcribe(audio_np, fp16=False)
        # return result["text"]

        return "[Transcription placeholder]"
