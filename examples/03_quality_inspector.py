#!/usr/bin/env python3
"""Quality inspector example - Manufacturing QA with Notion logging.

This example demonstrates a manufacturing quality control agent that:
- Inspects products via webcam (simulating a production line)
- Detects defects (scratches, dents, misalignment)
- Logs all inspections to Notion database
- Triggers PLC for reject mechanism on failed items

Run:
    python examples/03_quality_inspector.py

Requirements:
    - GOOGLE_API_KEY environment variable (uses Gemini for cost efficiency)
    - NOTION_API_KEY and NOTION_DATABASE_ID (optional for logging)
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import AgentLoop
from src.core.config import AgentConfig
from src.core.types import Message, ToolCall, ToolResult
from src.inputs.webcam import WebcamInput
from src.memory.sliding_window import SlidingWindowMemory
from src.models import create_model
from src.tools.notion import NotionRunSheetTool
from src.tools.plc import PLCWriteTool


SYSTEM_PROMPT = """You are a quality control inspector for a manufacturing line.

For each product image, perform a thorough inspection:

1. VISUAL DEFECTS
   - Scratches, scuffs, or surface damage
   - Dents or deformations
   - Discoloration or staining
   - Cracks or chips

2. DIMENSIONAL CHECKS
   - Alignment appears correct
   - Shape matches expected profile
   - No warping or bending

3. LABELING
   - Label is present and readable
   - Text is clear and complete
   - Barcode/QR code visible (if applicable)

ACTIONS:
- For EVERY inspection, log to Notion using update_notion_runsheet:
  - title: "Inspection #[timestamp]" or product identifier if visible
  - status: "completed" if passed, "blocked" if failed
  - notes: Brief summary of findings

- If defect found:
  1. Log with status "blocked" and describe the defect
  2. Use write_plc_register to trigger reject (register 100, value 1)

- If passed:
  1. Log with status "completed"

Be consistent and thorough. Every item must be logged.
"""


class InspectionCounter:
    """Track inspection statistics."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.now()

    def record_pass(self):
        self.total += 1
        self.passed += 1

    def record_fail(self):
        self.total += 1
        self.failed += 1

    def summary(self) -> str:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total / elapsed * 60 if elapsed > 0 else 0
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        return (
            f"Total: {self.total} | "
            f"Passed: {self.passed} | "
            f"Failed: {self.failed} | "
            f"Pass Rate: {pass_rate:.1f}% | "
            f"Rate: {rate:.1f}/min"
        )


async def main():
    """Run the quality inspector."""
    print("=" * 60)
    print("Manufacturing Quality Inspector")
    print("=" * 60)

    # Use Gemini 1.5 Flash for cost efficiency
    if os.getenv("GOOGLE_API_KEY"):
        model = create_model("google", "gemini-1.5-flash")
        print("Using Gemini 1.5 Flash (cost-optimized)")
    elif os.getenv("OPENAI_API_KEY"):
        model = create_model("openai", "gpt-4o-mini")
        print("Using GPT-4o Mini")
    else:
        print("Error: No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY")
        return

    memory = SlidingWindowMemory(max_messages=10)
    counter = InspectionCounter()

    agent = AgentLoop(
        model=model,
        memory=memory,
        config=AgentConfig(
            frame_batch_size=1,
            frame_interval_ms=2000,  # Inspect every 2 seconds
            system_prompt=SYSTEM_PROMPT,
        ),
    )

    # Register Notion tool (optional)
    if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID"):
        agent.register_tool(NotionRunSheetTool())
        print("Notion logging enabled")
    else:
        print("Notion logging disabled (set NOTION_API_KEY and NOTION_DATABASE_ID)")

    # Register PLC tool (simulation mode)
    plc_tool = PLCWriteTool(simulate=True)
    agent.register_tool(plc_tool)
    print("PLC control enabled (simulation mode)")

    # Set up webcam
    webcam = WebcamInput(
        device_id=0,
        fps=0.5,  # 1 frame every 2 seconds
        resolution=(1280, 720),  # Higher resolution for QC
        max_size=768,  # Larger for detail
    )

    print("\n" + "-" * 60)
    print("Quality inspection started. Press Ctrl+C to stop.")
    print("-" * 60 + "\n")

    async def on_event(event):
        if isinstance(event, Message) and event.role == "assistant":
            if not event.metadata.get("chunk"):
                print(f"[Inspector] {event.content[:200]}...")

        elif isinstance(event, ToolCall):
            if event.name == "update_notion_runsheet":
                status = event.arguments.get("status", "")
                if status == "completed":
                    counter.record_pass()
                    print(f"[✓ PASSED] {event.arguments.get('title', '')}")
                elif status == "blocked":
                    counter.record_fail()
                    print(f"[✗ FAILED] {event.arguments.get('title', '')}")
                    print(f"   Reason: {event.arguments.get('notes', 'No details')}")

            elif event.name == "write_plc_register":
                print(f"[PLC] Reject triggered - Register {event.arguments.get('register_address')}")

        elif isinstance(event, ToolResult):
            if not event.success:
                print(f"[Error] {event.error}")

        # Print summary periodically
        if counter.total > 0 and counter.total % 5 == 0:
            print(f"\n[Stats] {counter.summary()}\n")

    try:
        await agent.run(webcam, on_event=on_event)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Quality Inspection Summary")
        print("=" * 60)
        print(counter.summary())
        print("=" * 60)
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
