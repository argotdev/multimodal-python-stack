#!/usr/bin/env python3
"""Security monitor example - RTSP camera with Slack alerts.

This example demonstrates a security monitoring agent that:
- Connects to an RTSP camera stream (IP camera)
- Monitors for people, unusual activity, and safety hazards
- Sends Slack alerts with appropriate severity levels
- Saves all captured frames to a local directory

Run:
    python examples/02_security_monitor.py

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable
    - SLACK_WEBHOOK_URL environment variable (optional)
    - RTSP camera URL (or uses webcam by default)

Frames are saved to: ./captured_frames/YYYY-MM-DD/
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import AgentLoop
from src.core.config import AgentConfig
from src.core.types import Frame, Message, ToolCall, ToolResult
from src.inputs.rtsp import RTSPInput
from src.inputs.webcam import WebcamInput
from src.memory.sliding_window import SlidingWindowMemory
from src.models import create_model
from src.tools.slack import SlackAlertTool
from src.utils.image import FrameSaver


SYSTEM_PROMPT = """You are a security monitoring agent watching a camera feed.

Your job is to analyze each frame for:
1. People entering or leaving the area
2. Unusual movement patterns or behaviors
3. Suspicious objects or packages left unattended
4. Safety hazards (spills, obstacles, blocked exits)
5. Any other security concerns

When you observe something notable:
- Use send_slack_alert with severity "info" for routine observations
- Use send_slack_alert with severity "warning" for potential issues
- Use send_slack_alert with severity "critical" for immediate threats

Be concise but specific in your alerts. Include:
- What you observed
- Where in the frame (if relevant)
- Why it's concerning

If nothing notable, just acknowledge the frame is clear.
"""


class FrameSavingInput:
    """Wrapper that saves frames while passing them through."""

    def __init__(self, source, saver: FrameSaver):
        self.source = source
        self.saver = saver

    async def stream(self):
        async for frame in self.source.stream():
            # Save each frame
            path = self.saver.save(frame)
            print(f"[Frame Saved] {path}")
            yield frame

    async def close(self):
        await self.source.close()


async def main():
    """Run the security monitor."""
    print("=" * 50)
    print("Security Monitor")
    print("=" * 50)

    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Using OpenAI instead.")
        model = create_model("openai", "gpt-4o-mini")
    else:
        # Use Claude 3.5 Haiku - fast and cost-effective
        model = create_model("anthropic", "claude-3-5-haiku-latest")

    memory = SlidingWindowMemory(max_messages=20)

    agent = AgentLoop(
        model=model,
        memory=memory,
        config=AgentConfig(
            frame_batch_size=1,
            frame_interval_ms=5000,  # Check every 5 seconds
            system_prompt=SYSTEM_PROMPT,
        ),
    )

    # Register Slack tool
    if os.getenv("SLACK_WEBHOOK_URL"):
        agent.register_tool(
            SlackAlertTool(
                webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
                default_channel="#security-alerts",
            )
        )
        print("Slack alerts enabled")
    else:
        print("Warning: SLACK_WEBHOOK_URL not set. Alerts will not be sent.")

    # Set up frame saver
    frame_saver = FrameSaver(
        output_dir="./captured_frames",
        format="jpg",
        quality=85,
        create_subdirs=True,  # Creates YYYY-MM-DD subdirectories
    )
    print(f"Frames will be saved to: {frame_saver.output_dir.absolute()}")

    # Choose input source
    rtsp_url = os.getenv("RTSP_URL")
    if rtsp_url:
        print(f"Connecting to RTSP stream...")
        raw_input = RTSPInput(
            url=rtsp_url,
            fps=0.2,  # 1 frame every 5 seconds
            auto_reconnect=True,
            reconnect_delay=5.0,
        )
    else:
        print("No RTSP_URL set, using webcam as demo...")
        raw_input = WebcamInput(
            device_id=0,
            fps=0.2,
            max_size=512,
        )

    # Wrap input to save frames
    input_source = FrameSavingInput(raw_input, frame_saver)

    print("\nMonitoring started. Press Ctrl+C to stop.\n")
    print("-" * 50)

    async def on_event(event):
        if isinstance(event, Message) and event.role == "assistant":
            if not event.metadata.get("chunk"):
                print(f"[Observation] {event.content}")
        elif isinstance(event, ToolCall):
            print(f"[Alert Triggered] {event.name}: {event.arguments.get('message', '')}")
        elif isinstance(event, ToolResult):
            if event.success:
                print(f"[Alert Sent] ✓")
            else:
                print(f"[Alert Failed] {event.error}")
        print("-" * 50)

    try:
        await agent.run(input_source, on_event=on_event)
    except KeyboardInterrupt:
        print("\n\nShutting down security monitor...")
        agent.stop()
        print(f"Total frames saved: {frame_saver.saved_count}")
        print(f"Frames location: {frame_saver.output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
