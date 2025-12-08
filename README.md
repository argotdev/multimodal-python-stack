# The 2026 Python Stack for Real-Time Multimodal Agents

Build production-ready vision-language agents in under 300 lines of Python.

```python
from src.core.agent import AgentLoop
from src.inputs.webcam import WebcamInput
from src.models import create_model
from src.memory import SlidingWindowMemory
from src.tools import SlackAlertTool

# Create agent
agent = AgentLoop(
    model=create_model("openai", "gpt-4o-mini"),
    memory=SlidingWindowMemory(),
)
agent.register_tool(SlackAlertTool(webhook_url="..."))

# Run on webcam
await agent.run(WebcamInput())
```

## Features

- **Minimal Core**: ~150 lines for the complete agent loop
- **6 Model Providers**: OpenAI, Anthropic, Google, Groq, Fireworks, Together
- **5 Input Sources**: Webcam, microphone, files, RTSP streams, URLs
- **Plug-and-Play Tools**: Slack, Notion, PLC, robot arm
- **Battle-Tested**: Protocol-driven, async-first design

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run an Example

```bash
# Basic webcam demo
python examples/01_basic_webcam.py

# Security monitor with Slack alerts
python examples/02_security_monitor.py

# Manufacturing quality inspector
python examples/03_quality_inspector.py

# Meeting assistant
python examples/04_meeting_assistant.py

# Benchmark all providers
python examples/05_benchmark_providers.py
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Inputs    │────▶│    Agent    │────▶│   Tools     │
│             │     │    Loop     │     │             │
│ • Webcam    │     │             │     │ • Slack     │
│ • Mic       │     │ ┌─────────┐ │     │ • Notion    │
│ • Files     │     │ │ Buffer  │ │     │ • PLC       │
│ • RTSP      │     │ └────┬────┘ │     │ • Robot     │
│ • URLs      │     │      ▼      │     └─────────────┘
└─────────────┘     │ ┌─────────┐ │
                    │ │ Model   │ │     ┌─────────────┐
                    │ └────┬────┘ │────▶│   Memory    │
                    │      ▼      │     │             │
                    │ ┌─────────┐ │     │ • Sliding   │
                    │ │ Output  │ │     │   Window    │
                    │ └─────────┘ │     └─────────────┘
                    └─────────────┘
```

## Model Providers

| Provider | Models | Best For |
|----------|--------|----------|
| **OpenAI** | gpt-4o, gpt-4o-mini | Tool calling, general use |
| **Anthropic** | claude-3.5-sonnet, claude-3.5-haiku | Reasoning, safety |
| **Google** | gemini-1.5-flash, gemini-1.5-pro | Cost efficiency, video |
| **Groq** | llama-3.2-90b-vision, llama-3.2-11b-vision | Speed |
| **Fireworks** | firellava-13b, phi-3-vision | Open models |
| **Together** | llama-3.2-11b-vision, llama-3.2-90b-vision | Open models |

## Input Sources

```python
# Webcam
from src.inputs import WebcamInput
source = WebcamInput(device_id=0, fps=1.0)

# Microphone
from src.inputs import MicrophoneInput
source = MicrophoneInput(sample_rate=16000)

# Video/Audio Files
from src.inputs import FileInput
source = FileInput("recording.mp4")

# RTSP Stream (IP Cameras)
from src.inputs import RTSPInput
source = RTSPInput("rtsp://user:pass@192.168.1.100:554/stream")

# URLs
from src.inputs import URLInput
source = URLInput("https://example.com/image.jpg")

# Combined (webcam + mic)
from src.inputs.base import CompositeInput
source = CompositeInput(WebcamInput(), MicrophoneInput())
```

## Tools

### Slack Alerts

```python
from src.tools import SlackAlertTool

tool = SlackAlertTool(
    webhook_url="https://hooks.slack.com/...",
    default_channel="#alerts"
)
agent.register_tool(tool)
```

### Notion Run-Sheet

```python
from src.tools import NotionRunSheetTool

tool = NotionRunSheetTool(
    api_key="secret_...",
    database_id="abc123..."
)
agent.register_tool(tool)
```

### PLC Control (Industrial)

```python
from src.tools import PLCWriteTool

tool = PLCWriteTool(simulate=True)  # Use simulate=False for real PLCs
agent.register_tool(tool)
```

### Robot Arm

```python
from src.tools import RobotArmTool

tool = RobotArmTool(simulate=True)  # Use simulate=False for real robots
agent.register_tool(tool)
```

## Configuration

```python
from src.core.config import AgentConfig

config = AgentConfig(
    # Frame processing
    max_frames=4,           # Max frames per request
    frame_batch_size=1,     # Process after N frames
    frame_interval_ms=1000, # Min ms between captures

    # Audio processing
    min_audio_chars=50,     # Process after N transcribed chars

    # Context management
    max_context_tokens=4000,
    max_context_messages=20,

    # System prompt
    system_prompt="You are a helpful assistant...",
)

agent = AgentLoop(model=model, memory=memory, config=config)
```

## Benchmarking

Run benchmarks across all providers:

```bash
python examples/05_benchmark_providers.py
```

Or programmatically:

```python
from src.utils.benchmark import BenchmarkRunner
from src.models import create_model

runner = BenchmarkRunner()
runner.add_model(create_model("openai", "gpt-4o-mini"))
runner.add_model(create_model("anthropic", "claude-3-5-haiku-latest"))

results = await runner.run_all(iterations=10)
print(runner.to_markdown_table())
```

## Project Structure

```
multimodal-python-stack/
├── src/
│   ├── core/
│   │   ├── agent.py      # Main AgentLoop (~150 lines)
│   │   ├── types.py      # Frame, AudioChunk, Message, etc.
│   │   └── config.py     # AgentConfig
│   ├── inputs/
│   │   ├── webcam.py     # WebcamInput
│   │   ├── microphone.py # MicrophoneInput
│   │   ├── file.py       # FileInput, VideoFileInput, AudioFileInput
│   │   ├── rtsp.py       # RTSPInput
│   │   └── url.py        # URLInput
│   ├── models/
│   │   ├── openai.py     # OpenAIVisionModel
│   │   ├── anthropic.py  # AnthropicVisionModel
│   │   ├── google.py     # GoogleVisionModel
│   │   ├── groq.py       # GroqVisionModel
│   │   ├── fireworks.py  # FireworksVisionModel
│   │   └── together.py   # TogetherVisionModel
│   ├── tools/
│   │   ├── slack.py      # SlackAlertTool
│   │   ├── notion.py     # NotionRunSheetTool
│   │   ├── plc.py        # PLCWriteTool
│   │   └── robot.py      # RobotArmTool
│   ├── memory/
│   │   └── sliding_window.py
│   └── utils/
│       ├── audio.py      # WhisperTranscriber
│       ├── image.py      # Frame utilities
│       └── benchmark.py  # BenchmarkRunner
├── examples/
│   ├── 01_basic_webcam.py
│   ├── 02_security_monitor.py
│   ├── 03_quality_inspector.py
│   ├── 04_meeting_assistant.py
│   └── 05_benchmark_providers.py
└── docs/
    └── blog_post.md
```

## Requirements

- Python 3.11+
- API keys for at least one provider

## License

MIT

## Contributing

Contributions welcome! Please read the contributing guidelines first.
