"""Input sources for multimodal agents."""

from src.inputs.base import InputSource
from src.inputs.webcam import WebcamInput
from src.inputs.microphone import MicrophoneInput
from src.inputs.file import FileInput, VideoFileInput, AudioFileInput
from src.inputs.rtsp import RTSPInput
from src.inputs.url import URLInput

__all__ = [
    "InputSource",
    "WebcamInput",
    "MicrophoneInput",
    "FileInput",
    "VideoFileInput",
    "AudioFileInput",
    "RTSPInput",
    "URLInput",
]
