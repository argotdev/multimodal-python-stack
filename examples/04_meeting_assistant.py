#!/usr/bin/env python3
"""Meeting assistant example - Audio + video with action items.

This example demonstrates a meeting assistant agent that:
- Listens to audio and periodically captures video
- Transcribes speech using Whisper
- Identifies action items and decisions
- Logs to Notion and sends summaries to Slack

Run:
    python examples/04_meeting_assistant.py

Requirements:
    - OPENAI_API_KEY environment variable (for GPT-4o and Whisper)
    - SLACK_WEBHOOK_URL (optional for summaries)
    - NOTION_API_KEY and NOTION_DATABASE_ID (optional for action items)
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import AgentLoop
from src.core.config import AgentConfig
from src.core.types import AudioChunk, Message, ToolCall, ToolResult
from src.inputs.base import CompositeInput
from src.inputs.microphone import MicrophoneInput
from src.inputs.webcam import WebcamInput
from src.memory.sliding_window import SlidingWindowMemory
from src.models import create_model
from src.tools.notion import NotionRunSheetTool
from src.tools.slack import SlackAlertTool
from src.utils.audio import WhisperTranscriber


SYSTEM_PROMPT = """You are a meeting assistant. Your job is to:

1. TRACK DISCUSSIONS
   - Note key topics being discussed
   - Identify when topics change
   - Summarize main points

2. CAPTURE ACTION ITEMS
   When someone commits to doing something, create an action item:
   - Use update_notion_runsheet with title="[ACTION] description"
   - Set status to "pending"
   - Include who is responsible in notes

3. RECORD DECISIONS
   When a decision is made:
   - Use update_notion_runsheet with title="[DECISION] description"
   - Set status to "completed"
   - Note any context in notes

4. PERIODIC SUMMARIES
   Every few minutes, send a brief summary to Slack:
   - Key topics discussed
   - Decisions made
   - Action items assigned
   Use send_slack_alert with severity "info"

5. VISUAL CONTEXT (from periodic video frames)
   - Note who is present in the room
   - Observe any visual materials being shown
   - Note the meeting environment

Be concise and focus on actionable information.
If you hear someone say "action item" or "I'll do X", always create a Notion entry.
"""


class MeetingStats:
    """Track meeting statistics."""

    def __init__(self):
        self.action_items = []
        self.decisions = []
        self.topics = []
        self.start_time = datetime.now()

    def add_action(self, description: str, assignee: str = ""):
        self.action_items.append({"description": description, "assignee": assignee})

    def add_decision(self, description: str):
        self.decisions.append(description)

    def duration_minutes(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 60

    def summary(self) -> str:
        return (
            f"Duration: {self.duration_minutes():.0f} min | "
            f"Action Items: {len(self.action_items)} | "
            f"Decisions: {len(self.decisions)}"
        )


async def main():
    """Run the meeting assistant."""
    print("=" * 60)
    print("Meeting Assistant")
    print("=" * 60)

    # Use GPT-4o for best reasoning with multimodal
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for this example")
        return

    model = create_model("openai", "gpt-4o")
    print("Using GPT-4o for meeting analysis")

    # Set up transcriber
    transcriber = WhisperTranscriber()

    memory = SlidingWindowMemory(max_messages=30)
    stats = MeetingStats()

    agent = AgentLoop(
        model=model,
        memory=memory,
        config=AgentConfig(
            frame_batch_size=1,
            frame_interval_ms=30000,  # Video every 30 seconds
            min_audio_chars=200,  # Process after ~200 chars of transcript
            system_prompt=SYSTEM_PROMPT,
        ),
    )

    # Set transcriber
    agent.set_transcriber(transcriber)

    # Register tools
    if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID"):
        agent.register_tool(NotionRunSheetTool())
        print("Notion action items enabled")
    else:
        print("Notion disabled (set NOTION_API_KEY and NOTION_DATABASE_ID)")

    if os.getenv("SLACK_WEBHOOK_URL"):
        agent.register_tool(
            SlackAlertTool(
                webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
                default_channel="#meeting-notes",
            )
        )
        print("Slack summaries enabled")
    else:
        print("Slack disabled (set SLACK_WEBHOOK_URL)")

    # Set up combined input (microphone + webcam)
    try:
        mic = MicrophoneInput(
            sample_rate=16000,
            chunk_duration=10.0,  # 10 second audio chunks
        )
        webcam = WebcamInput(
            device_id=0,
            fps=0.033,  # 1 frame per 30 seconds
            max_size=512,
        )
        input_source = CompositeInput(mic, webcam)
        print("\nUsing microphone + webcam")
    except Exception as e:
        print(f"Could not set up audio/video: {e}")
        print("Running in demo mode with webcam only...")
        input_source = WebcamInput(device_id=0, fps=0.1, max_size=512)

    print("\n" + "-" * 60)
    print("Meeting recording started. Press Ctrl+C to end meeting.")
    print("-" * 60 + "\n")

    async def on_event(event):
        if isinstance(event, Message) and event.role == "assistant":
            if not event.metadata.get("chunk"):
                # Print observations
                content = event.content[:300]
                if len(event.content) > 300:
                    content += "..."
                print(f"[Assistant] {content}")

        elif isinstance(event, ToolCall):
            if event.name == "update_notion_runsheet":
                title = event.arguments.get("title", "")
                if title.startswith("[ACTION]"):
                    stats.add_action(title, event.arguments.get("notes", ""))
                    print(f"\n📋 ACTION ITEM: {title}")
                elif title.startswith("[DECISION]"):
                    stats.add_decision(title)
                    print(f"\n✅ DECISION: {title}")
                else:
                    print(f"\n📝 Note: {title}")

            elif event.name == "send_slack_alert":
                print(f"\n📤 Slack Summary Sent")

        elif isinstance(event, ToolResult):
            if not event.success:
                print(f"[Error] {event.error}")

        # Status line
        print(f"\r[{stats.summary()}]", end="", flush=True)

    try:
        await agent.run(input_source, on_event=on_event)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Meeting Summary")
        print("=" * 60)
        print(f"Duration: {stats.duration_minutes():.0f} minutes")
        print(f"\nAction Items ({len(stats.action_items)}):")
        for item in stats.action_items:
            print(f"  • {item['description']}")
            if item['assignee']:
                print(f"    Assigned to: {item['assignee']}")
        print(f"\nDecisions ({len(stats.decisions)}):")
        for decision in stats.decisions:
            print(f"  • {decision}")
        print("=" * 60)
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
