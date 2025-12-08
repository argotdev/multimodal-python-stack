# Internal Setup Guide: Multimodal Python Stack

This document provides detailed instructions for setting up, testing, and running the multimodal agents framework. It covers everything from API key configuration to hardware requirements and troubleshooting.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [API Key Configuration](#api-key-configuration)
4. [Hardware Requirements](#hardware-requirements)
5. [Running the Examples](#running-the-examples)
6. [Testing Individual Components](#testing-individual-components)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
8. [Development Workflow](#development-workflow)
9. [Cost Management](#cost-management)
10. [Production Deployment](#production-deployment)

---

## Prerequisites

### Required Software

| Software | Version | Purpose | Installation |
|----------|---------|---------|--------------|
| Python | 3.11+ | Runtime | `brew install python@3.11` or pyenv |
| pip/uv | Latest | Package manager | Comes with Python / `pip install uv` |
| ffmpeg | 6.0+ | Audio/video processing | `brew install ffmpeg` |
| PortAudio | Latest | Microphone access | `brew install portaudio` |

### Verify Installation

```bash
# Check Python version
python3 --version  # Should be 3.11+

# Check ffmpeg
ffmpeg -version

# Check PortAudio (for microphone)
brew list portaudio  # On macOS
```

### macOS-Specific Setup

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.11 ffmpeg portaudio

# Grant camera/microphone permissions
# System Preferences > Privacy & Security > Camera/Microphone
# Add Terminal.app or your IDE
```

### Linux-Specific Setup

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip ffmpeg portaudio19-dev libportaudio2

# Fedora
sudo dnf install python3.11 ffmpeg portaudio-devel
```

---

## Environment Setup

### Option 1: Using uv (Recommended)

```bash
# Install uv
pip install uv

# Create virtual environment
cd multimodal-python-stack
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Poetry

```bash
# Install poetry
pip install poetry

# Install dependencies
poetry install

# Activate shell
poetry shell
```

### Verify Installation

```bash
# Test imports
python -c "
from src.core.agent import AgentLoop
from src.models import create_model
from src.inputs import WebcamInput
print('All imports successful!')
"
```

---

## API Key Configuration

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

### Step 2: Configure API Keys

Edit `.env` with your actual API keys:

```bash
# ===========================================
# MODEL PROVIDERS
# ===========================================

# OpenAI - Required for GPT-4o and Whisper transcription
# Get key: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-...

# Anthropic - Required for Claude models
# Get key: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-api03-...

# Google - Required for Gemini models
# Get key: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIza...

# Groq - Required for fast Llama inference
# Get key: https://console.groq.com/keys
GROQ_API_KEY=gsk_...

# Fireworks - Required for FireLLaVA
# Get key: https://fireworks.ai/api-keys
FIREWORKS_API_KEY=fw_...

# Together - Required for Together models
# Get key: https://api.together.xyz/settings/api-keys
TOGETHER_API_KEY=...

# ===========================================
# TOOL INTEGRATIONS
# ===========================================

# Slack Webhooks - For alert examples
# Create: https://api.slack.com/messaging/webhooks
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...

# Notion - For run-sheet logging examples
# Create integration: https://www.notion.so/my-integrations
# Then share a database with your integration
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=...
```

### Step 3: Verify API Keys

```bash
# Test OpenAI
python -c "
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
print('OpenAI:', response.choices[0].message.content)
"

# Test Anthropic
python -c "
import os
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv()
client = Anthropic()
response = client.messages.create(
    model='claude-3-5-haiku-latest',
    max_tokens=10,
    messages=[{'role': 'user', 'content': 'Say hello'}]
)
print('Anthropic:', response.content[0].text)
"

# Test Google
python -c "
import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Say hello')
print('Google:', response.text)
"
```

### Getting API Keys - Detailed Instructions

#### OpenAI

1. Go to https://platform.openai.com/signup
2. Create account or sign in
3. Navigate to API Keys: https://platform.openai.com/api-keys
4. Click "Create new secret key"
5. Copy the key (starts with `sk-proj-`)
6. Add billing: https://platform.openai.com/account/billing

**Pricing (as of 2025):**
- GPT-4o: $5.00/1M input, $15.00/1M output
- GPT-4o-mini: $0.15/1M input, $0.60/1M output
- Whisper: $0.006/minute

#### Anthropic

1. Go to https://console.anthropic.com/
2. Create account or sign in
3. Navigate to API Keys: https://console.anthropic.com/settings/keys
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

**Pricing:**
- Claude 3.5 Sonnet: $3.00/1M input, $15.00/1M output
- Claude 3.5 Haiku: $0.80/1M input, $4.00/1M output

#### Google (Gemini)

1. Go to https://aistudio.google.com/
2. Sign in with Google account
3. Click "Get API key" in the top right
4. Create API key for a new or existing project
5. Copy the key (starts with `AIza`)

**Pricing:**
- Gemini 1.5 Flash: $0.075/1M input, $0.30/1M output (cheapest!)
- Gemini 1.5 Pro: $1.25/1M input, $5.00/1M output

#### Groq

1. Go to https://console.groq.com/
2. Create account or sign in
3. Navigate to API Keys: https://console.groq.com/keys
4. Click "Create API Key"
5. Copy the key (starts with `gsk_`)

**Pricing:**
- Llama 3.2 90B Vision: $0.11/1M tokens
- Llama 3.2 11B Vision: $0.05/1M tokens

#### Slack Webhook

1. Go to https://api.slack.com/apps
2. Click "Create New App" > "From scratch"
3. Name it (e.g., "Multimodal Agent") and select workspace
4. Go to "Incoming Webhooks" in sidebar
5. Toggle "Activate Incoming Webhooks" ON
6. Click "Add New Webhook to Workspace"
7. Select the channel for alerts
8. Copy the webhook URL

#### Notion Integration

1. Go to https://www.notion.so/my-integrations
2. Click "New integration"
3. Name it (e.g., "Multimodal Agent")
4. Select workspace
5. Copy the "Internal Integration Token" (starts with `secret_`)

**Database Setup:**
1. Create a new Notion database with these properties:
   - Title (title type)
   - Status (select: pending, in_progress, completed, blocked)
   - Notes (rich text)
   - Timestamp (date)
   - Tags (multi-select, optional)
2. Click "..." menu > "Add connections" > Select your integration
3. Copy the database ID from the URL:
   - URL: `https://www.notion.so/myworkspace/abc123def456...`
   - Database ID: `abc123def456` (32-character hex string)

---

## Hardware Requirements

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 4 cores | For video encoding/decoding |
| RAM | 8 GB | 16 GB recommended |
| Storage | 1 GB free | For dependencies and temp files |
| Camera | USB or built-in | For webcam examples |
| Microphone | Any | For audio examples |

### Camera Setup

#### Built-in Webcam (Laptop)

Works out of the box. Device ID is typically `0`.

```python
from src.inputs import WebcamInput
webcam = WebcamInput(device_id=0)
```

#### External USB Camera

```bash
# List available cameras (macOS)
system_profiler SPCameraDataType

# List available cameras (Linux)
v4l2-ctl --list-devices
```

```python
# Usually device_id=1 for external camera
webcam = WebcamInput(device_id=1)
```

#### IP Camera (RTSP)

```python
from src.inputs import RTSPInput

# Common RTSP URL formats:
# Hikvision: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
# Dahua: rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
# Generic: rtsp://user:pass@ip:port/path

camera = RTSPInput(
    url="rtsp://admin:password@192.168.1.100:554/stream",
    fps=1.0,
    auto_reconnect=True
)
```

#### Testing Camera

```python
# Quick camera test
import cv2

cap = cv2.VideoCapture(0)  # Try 0, 1, 2...
if cap.isOpened():
    ret, frame = cap.read()
    print(f"Camera working: {ret}, Frame shape: {frame.shape if ret else 'N/A'}")
    cap.release()
else:
    print("Camera not found")
```

### Microphone Setup

#### Grant Permissions (macOS)

1. System Preferences > Privacy & Security > Microphone
2. Enable for Terminal.app or your IDE

#### List Audio Devices

```python
import sounddevice as sd
print(sd.query_devices())
```

Output:
```
   0 MacBook Pro Microphone, Core Audio (1 in, 0 out)
>  1 MacBook Pro Speakers, Core Audio (0 in, 2 out)
   2 External USB Mic, Core Audio (1 in, 0 out)
```

Use the index as `device_id`:

```python
from src.inputs import MicrophoneInput
mic = MicrophoneInput(device_id=0)  # Built-in mic
mic = MicrophoneInput(device_id=2)  # External USB mic
```

#### Testing Microphone

```python
import sounddevice as sd
import numpy as np

duration = 3  # seconds
sample_rate = 16000

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
print(f"Recorded {len(audio)} samples, max amplitude: {np.max(np.abs(audio)):.4f}")
```

---

## Running the Examples

### Example 1: Basic Webcam

**Requirements:** OpenAI API key, webcam

```bash
python examples/01_basic_webcam.py
```

**What it does:**
- Captures frames from webcam every 3 seconds
- Sends to GPT-4o-mini for description
- Prints observations to console

**Expected output:**
```
Starting basic webcam agent...
Press Ctrl+C to stop

[Agent] I see a person sitting at a desk with a laptop, appears to be in a home office setting.

[Agent] The scene is similar, the person is now typing on the keyboard.
```

### Example 2: Security Monitor

**Requirements:** Anthropic or OpenAI API key, webcam or RTSP camera, Slack webhook (optional)

```bash
# With webcam (demo mode)
python examples/02_security_monitor.py

# With RTSP camera
RTSP_URL="rtsp://user:pass@192.168.1.100:554/stream" python examples/02_security_monitor.py
```

**What it does:**
- Monitors camera feed every 5 seconds
- Detects people, unusual activity, hazards
- Sends Slack alerts when something is detected

**Expected output:**
```
==================================================
Security Monitor
==================================================
  ✓ Anthropic (claude-3-5-haiku)
Slack alerts enabled
Using webcam as demo...

Monitoring started. Press Ctrl+C to stop.
--------------------------------------------------
[Observation] The frame shows an empty room with a desk and chair. No people or unusual activity detected.
--------------------------------------------------
[Observation] A person has entered the frame from the left side. They appear to be walking toward the desk.
[Alert Triggered] send_slack_alert: Person detected entering monitored area
[Alert Sent] ✓
--------------------------------------------------
```

### Example 3: Quality Inspector

**Requirements:** Google or OpenAI API key, webcam, Notion API (optional)

```bash
python examples/03_quality_inspector.py
```

**What it does:**
- Simulates manufacturing line inspection
- Analyzes each frame for defects
- Logs all inspections to Notion
- Triggers PLC reject on failures (simulated)

**Expected output:**
```
============================================================
Manufacturing Quality Inspector
============================================================
Using Gemini 1.5 Flash (cost-optimized)
Notion logging disabled (set NOTION_API_KEY and NOTION_DATABASE_ID)
PLC control enabled (simulation mode)

------------------------------------------------------------
Quality inspection started. Press Ctrl+C to stop.
------------------------------------------------------------

[Inspector] Analyzing product in frame. The item appears to be...
[✓ PASSED] Inspection #2024-01-15T10:23:45
[Stats] Total: 1 | Passed: 1 | Failed: 0 | Pass Rate: 100.0% | Rate: 12.0/min

[Inspector] Analyzing product in frame. I notice a visible scratch...
[✗ FAILED] Inspection #2024-01-15T10:23:47
   Reason: Surface scratch detected on upper left quadrant, approximately 2cm long
[PLC] Reject triggered - Register 100
```

### Example 4: Meeting Assistant

**Requirements:** OpenAI API key (for GPT-4o and Whisper), webcam, microphone

```bash
python examples/04_meeting_assistant.py
```

**What it does:**
- Records audio continuously
- Captures video periodically (every 30s)
- Transcribes speech with Whisper
- Extracts action items and decisions
- Logs to Notion, sends summaries to Slack

**Expected output:**
```
============================================================
Meeting Assistant
============================================================
Using GPT-4o for meeting analysis
Notion action items enabled
Slack summaries enabled

Using microphone + webcam

------------------------------------------------------------
Meeting recording started. Press Ctrl+C to end meeting.
------------------------------------------------------------

[Assistant] The meeting has begun. I can see 3 people in the room...

📋 ACTION ITEM: [ACTION] John to prepare Q4 budget proposal
[Duration: 5 min | Action Items: 1 | Decisions: 0]

✅ DECISION: [DECISION] Team agreed to postpone launch to March
[Duration: 8 min | Action Items: 1 | Decisions: 1]

📤 Slack Summary Sent
```

### Example 5: Benchmark Providers

**Requirements:** API keys for providers you want to test

```bash
python examples/05_benchmark_providers.py
```

**What it does:**
- Tests all available providers
- Runs standardized scenarios
- Generates latency and cost tables
- Saves results to JSON

**Expected output:**
```
======================================================================
Multimodal Model Benchmark
======================================================================

Checking available providers...
  ✓ OpenAI (gpt-4o, gpt-4o-mini)
  ✓ Anthropic (claude-3-5-haiku, claude-3-5-sonnet)
  ✓ Google (gemini-1.5-flash)
  ✗ Groq (set GROQ_API_KEY)
  ✗ Fireworks (set FIREWORKS_API_KEY)
  ✗ Together (set TOGETHER_API_KEY)

Running benchmarks on 5 models...
Scenarios: single_frame, multi_frame, detailed_analysis, tool_calling

----------------------------------------------------------------------
Starting benchmarks (this may take a few minutes)...
----------------------------------------------------------------------

Benchmarking openai/gpt-4o-mini - single_frame
Benchmarking openai/gpt-4o-mini - multi_frame
...

======================================================================
Results
======================================================================

LATENCY (p50, milliseconds)
----------------------------------------------------------------------
| Provider | Model | single_frame | multi_frame | tool_calling |
|---|---|---:|---:|---:|
| openai | gpt-4o-mini | 423ms | 612ms | 489ms |
| openai | gpt-4o | 834ms | 1156ms | 923ms |
| anthropic | claude-3-5-haiku-latest | 367ms | 542ms | 421ms |
...

Results saved to: benchmarks/results/benchmark_results.json
```

---

## Testing Individual Components

### Test Input Sources

```python
# Test webcam
import asyncio
from src.inputs import WebcamInput

async def test_webcam():
    webcam = WebcamInput(device_id=0, fps=1.0)
    count = 0
    async for frame in webcam.stream():
        print(f"Frame {count}: shape={frame.shape}, source={frame.source}")
        count += 1
        if count >= 3:
            break
    await webcam.close()

asyncio.run(test_webcam())
```

```python
# Test microphone
import asyncio
from src.inputs import MicrophoneInput

async def test_mic():
    mic = MicrophoneInput(sample_rate=16000, chunk_duration=2.0)
    count = 0
    async for chunk in mic.stream():
        print(f"Chunk {count}: samples={len(chunk.data)}, duration={chunk.duration_seconds:.2f}s")
        count += 1
        if count >= 3:
            break
    await mic.close()

asyncio.run(test_mic())
```

### Test Model Providers

```python
# Test model with a single frame
import asyncio
import numpy as np
from src.models import create_model
from src.core.types import Frame

async def test_model(provider, model_id):
    model = create_model(provider, model_id)

    # Create a test frame (random colored image)
    test_frame = Frame(
        data=np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
        source="test"
    )

    print(f"Testing {provider}/{model_id}...")
    async for event in model.analyze(
        frames=[test_frame],
        audio_transcript=None,
        tools=[],
        context=[],
        system_prompt="Describe this image briefly."
    ):
        print(f"  Event: {type(event).__name__}")
        if hasattr(event, 'content'):
            print(f"  Content: {event.content[:100]}...")

# Test each provider
asyncio.run(test_model("openai", "gpt-4o-mini"))
asyncio.run(test_model("anthropic", "claude-3-5-haiku-latest"))
asyncio.run(test_model("google", "gemini-1.5-flash"))
```

### Test Tools

```python
# Test Slack tool (requires SLACK_WEBHOOK_URL)
import asyncio
import os
from dotenv import load_dotenv
from src.tools import SlackAlertTool

load_dotenv()

async def test_slack():
    tool = SlackAlertTool(
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        default_channel="#test-alerts"
    )

    result = await tool.execute(
        message="Test alert from multimodal agent",
        severity="info"
    )

    print(f"Slack result: {result}")

asyncio.run(test_slack())
```

```python
# Test Notion tool (requires NOTION_API_KEY and NOTION_DATABASE_ID)
import asyncio
import os
from dotenv import load_dotenv
from src.tools import NotionRunSheetTool

load_dotenv()

async def test_notion():
    tool = NotionRunSheetTool(
        api_key=os.getenv("NOTION_API_KEY"),
        database_id=os.getenv("NOTION_DATABASE_ID")
    )

    result = await tool.execute(
        title="Test Entry from Multimodal Agent",
        status="completed",
        notes="This is a test entry created by the setup script."
    )

    print(f"Notion result: {result}")

asyncio.run(test_notion())
```

### Test Memory

```python
from src.memory import SlidingWindowMemory
from src.core.types import Message
from datetime import datetime

memory = SlidingWindowMemory(max_messages=5)

# Add some messages
for i in range(7):
    memory.add(Message(
        role="user" if i % 2 == 0 else "assistant",
        content=f"Message {i}",
        timestamp=datetime.now()
    ))

# Check what's retained
context = memory.get_context()
print(f"Messages in memory: {len(context)}")
for msg in context:
    print(f"  {msg.role}: {msg.content}")
```

---

## Common Issues & Troubleshooting

### Camera Issues

**Problem:** `cv2.VideoCapture` returns False

```python
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # False
```

**Solutions:**

1. **Check permissions (macOS):**
   - System Preferences > Privacy & Security > Camera
   - Enable for Terminal.app

2. **Try different device IDs:**
   ```python
   for i in range(5):
       cap = cv2.VideoCapture(i)
       if cap.isOpened():
           print(f"Camera found at index {i}")
           cap.release()
   ```

3. **Check if camera is in use by another app:**
   ```bash
   # macOS
   lsof | grep -i camera
   ```

4. **Restart camera service (macOS):**
   ```bash
   sudo killall VDCAssistant
   sudo killall AppleCameraAssistant
   ```

### Microphone Issues

**Problem:** `sounddevice.PortAudioError`

**Solutions:**

1. **Install PortAudio:**
   ```bash
   # macOS
   brew install portaudio

   # Then reinstall sounddevice
   pip uninstall sounddevice
   pip install sounddevice
   ```

2. **Check permissions (macOS):**
   - System Preferences > Privacy & Security > Microphone
   - Enable for Terminal.app

3. **List devices and use explicit ID:**
   ```python
   import sounddevice as sd
   print(sd.query_devices())
   # Use the correct index in MicrophoneInput(device_id=X)
   ```

### API Errors

**Problem:** `openai.AuthenticationError`

**Solution:** Check API key is set correctly:
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(f"Key starts with: {os.getenv('OPENAI_API_KEY', '')[:10]}...")
```

**Problem:** `anthropic.RateLimitError`

**Solution:** Add delays between requests or upgrade plan:
```python
import asyncio
await asyncio.sleep(1)  # Add between requests
```

**Problem:** `google.api_core.exceptions.ResourceExhausted`

**Solution:** Gemini has strict rate limits. Add delays:
```python
config = AgentConfig(frame_interval_ms=5000)  # Slow down
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root or add to path:
```python
import sys
sys.path.insert(0, '/path/to/multimodal-python-stack')
```

Or set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/multimodal-python-stack"
```

### Memory/Performance Issues

**Problem:** High memory usage with video

**Solution:** Reduce frame size and buffer:
```python
webcam = WebcamInput(
    max_size=256,  # Smaller frames
    fps=0.5,       # Slower capture
)

config = AgentConfig(
    max_frames=2,  # Keep fewer frames
    max_context_messages=10,  # Smaller context
)
```

**Problem:** Slow response times

**Solutions:**

1. Use faster models:
   ```python
   model = create_model("groq", "llama-3.2-11b-vision-preview")  # Fastest
   model = create_model("google", "gemini-1.5-flash")  # Fast and cheap
   ```

2. Reduce frame size:
   ```python
   webcam = WebcamInput(max_size=256)
   ```

3. Use low detail mode (OpenAI):
   ```python
   model = OpenAIVisionModel(model_id="gpt-4o-mini", image_detail="low")
   ```

---

## Development Workflow

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_core.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=src tests/
```

### Code Formatting

```bash
# Install ruff
pip install ruff

# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Checking

```bash
# Install mypy
pip install mypy

# Run type checker
mypy src/
```

### Adding a New Model Provider

1. Create `src/models/newprovider.py`:

```python
from src.models.base import VisionLanguageModel, ModelInfo

class NewProviderVisionModel(VisionLanguageModel):
    provider = "newprovider"

    MODELS = {
        "model-name": ModelInfo(
            model_id="model-name",
            provider="newprovider",
            display_name="Model Name",
            max_images=10,
            supports_video=False,
            supports_tools=True,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            context_window=128000,
        ),
    }

    async def analyze(self, frames, audio_transcript, tools, context, system_prompt):
        # Implementation
        ...
```

2. Register in `src/models/__init__.py`:

```python
from src.models.newprovider import NewProviderVisionModel

PROVIDERS = {
    # ...existing providers...
    "newprovider": NewProviderVisionModel,
}
```

3. Add API key to `.env.example`:

```bash
NEWPROVIDER_API_KEY=...
```

### Adding a New Tool

1. Create `src/tools/newtool.py`:

```python
from src.tools.base import Tool
from src.core.types import ToolResult

class NewTool(Tool):
    name = "new_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "First argument"},
        },
        "required": ["arg1"]
    }

    async def execute(self, arg1: str, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(output={"result": "success"})
```

2. Register in `src/tools/__init__.py`:

```python
from src.tools.newtool import NewTool
__all__ = [..., "NewTool"]
```

---

## Cost Management

### Estimating Costs

Use the cost calculator:

```python
from src.models import create_model

model = create_model("openai", "gpt-4o-mini")

# Estimate cost for a session
frames_per_minute = 12  # 1 frame every 5 seconds
minutes = 60
total_frames = frames_per_minute * minutes

# ~85 tokens per image (low detail)
# ~50 tokens output per response
tokens_in = total_frames * 85
tokens_out = total_frames * 50

cost_in = (tokens_in / 1000) * model.cost_per_1k_input_tokens
cost_out = (tokens_out / 1000) * model.cost_per_1k_output_tokens
total_cost = cost_in + cost_out

print(f"Estimated cost for 1 hour: ${total_cost:.2f}")
```

### Cost Comparison (1 hour of monitoring)

| Provider | Model | Est. Cost/Hour |
|----------|-------|----------------|
| Google | gemini-1.5-flash | $0.03 |
| Groq | llama-3.2-11b | $0.05 |
| OpenAI | gpt-4o-mini | $0.15 |
| Anthropic | claude-3.5-haiku | $0.60 |
| OpenAI | gpt-4o | $3.00 |
| Anthropic | claude-3.5-sonnet | $4.50 |

### Setting Budget Alerts

```python
class BudgetTracker:
    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent = 0.0

    def add_cost(self, cost: float):
        self.spent += cost
        if self.spent > self.budget * 0.8:
            print(f"WARNING: 80% of budget used (${self.spent:.2f}/${self.budget})")
        if self.spent > self.budget:
            raise Exception(f"Budget exceeded: ${self.spent:.2f}/${self.budget}")

# Usage
tracker = BudgetTracker(budget_usd=10.0)
# Call tracker.add_cost() after each API call
```

---

## Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY examples/ examples/

# Run
CMD ["python", "examples/02_security_monitor.py"]
```

```bash
# Build and run
docker build -t multimodal-agent .
docker run --env-file .env --device /dev/video0 multimodal-agent
```

### Environment Variables for Production

```bash
# Production .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate limiting
MAX_REQUESTS_PER_MINUTE=60

# Cost controls
BUDGET_USD_PER_HOUR=5.0

# Monitoring
SENTRY_DSN=https://...
```

### Health Checks

```python
# healthcheck.py
import asyncio
from src.models import create_model

async def check_health():
    checks = {}

    # Check OpenAI
    try:
        model = create_model("openai", "gpt-4o-mini")
        # Quick test
        checks["openai"] = "ok"
    except Exception as e:
        checks["openai"] = f"error: {e}"

    # Check camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        checks["camera"] = "ok" if cap.isOpened() else "error: not found"
        cap.release()
    except Exception as e:
        checks["camera"] = f"error: {e}"

    return checks

if __name__ == "__main__":
    results = asyncio.run(check_health())
    for check, status in results.items():
        print(f"{check}: {status}")
```

### Logging Setup

```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Usage
logger.info("agent_started", model="gpt-4o-mini", input="webcam")
logger.info("frame_processed", latency_ms=423, tokens=156)
logger.warning("rate_limit_approaching", requests=58, limit=60)
logger.error("api_error", provider="openai", error="timeout")
```

---

## Quick Reference

### Start Commands

```bash
# Basic webcam
python examples/01_basic_webcam.py

# Security monitor
python examples/02_security_monitor.py

# Quality inspector
python examples/03_quality_inspector.py

# Meeting assistant
python examples/04_meeting_assistant.py

# Benchmarks
python examples/05_benchmark_providers.py
```

### Environment Variables

| Variable | Required For | Example |
|----------|--------------|---------|
| `OPENAI_API_KEY` | OpenAI, Whisper | `sk-proj-...` |
| `ANTHROPIC_API_KEY` | Anthropic | `sk-ant-...` |
| `GOOGLE_API_KEY` | Gemini | `AIza...` |
| `GROQ_API_KEY` | Groq | `gsk_...` |
| `SLACK_WEBHOOK_URL` | Slack alerts | `https://hooks.slack.com/...` |
| `NOTION_API_KEY` | Notion logging | `secret_...` |
| `NOTION_DATABASE_ID` | Notion logging | `abc123...` |
| `RTSP_URL` | IP cameras | `rtsp://user:pass@ip:port/path` |

### Model Quick Reference

| Use Case | Recommended | Command |
|----------|-------------|---------|
| Cheapest | Gemini Flash | `create_model("google", "gemini-1.5-flash")` |
| Fastest | Groq Llama | `create_model("groq", "llama-3.2-11b-vision-preview")` |
| Best balance | GPT-4o-mini | `create_model("openai", "gpt-4o-mini")` |
| Best quality | GPT-4o | `create_model("openai", "gpt-4o")` |
| Best reasoning | Claude Sonnet | `create_model("anthropic", "claude-3-5-sonnet-latest")` |
