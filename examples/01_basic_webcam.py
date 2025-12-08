#!/usr/bin/env python3
"""Basic webcam example - the simplest multimodal agent.

This example shows the minimal code needed to run a multimodal agent:
- Capture frames from your webcam
- Send them to a vision-language model
- Print the model's observations

Run:
    python examples/01_basic_webcam.py

Requirements:
    - OPENAI_API_KEY environment variable (or use .env file)
    - Webcam connected
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import AgentLoop
from src.core.config import AgentConfig
from src.core.types import Message
from src.inputs.webcam import WebcamInput
from src.memory.sliding_window import SlidingWindowMemory
from src.models import create_model


async def main():
    """Run a basic webcam agent."""
    print("Starting basic webcam agent...")
    print("Press Ctrl+C to stop\n")

    # Create model (uses OPENAI_API_KEY from environment)
    model = create_model("openai", "gpt-4o-mini")

    # Create memory
    memory = SlidingWindowMemory(max_messages=10)

    # Create agent with simple config
    agent = AgentLoop(
        model=model,
        memory=memory,
        config=AgentConfig(
            frame_batch_size=1,  # Process each frame
            frame_interval_ms=3000,  # Every 3 seconds
            system_prompt="You are observing a webcam feed. Briefly describe what you see in 1-2 sentences.",
        ),
    )

    # Create webcam input
    webcam = WebcamInput(
        device_id=0,  # Default camera
        fps=0.33,  # ~1 frame every 3 seconds
        max_size=512,  # Resize for efficiency
    )

    # Callback to handle events
    async def on_event(event):
        if isinstance(event, Message) and event.role == "assistant":
            # Only print complete responses (not chunks)
            if not event.metadata.get("chunk"):
                print(f"[Agent] {event.content}\n")

    # Run the agent
    try:
        await agent.run(webcam, on_event=on_event)
    except KeyboardInterrupt:
        print("\nStopping agent...")
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
