#!/usr/bin/env python3
"""Debug test - verify each component works."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def test_webcam():
    """Test webcam capture."""
    print("=" * 50)
    print("TEST 1: Webcam Capture")
    print("=" * 50)

    from src.inputs.webcam import WebcamInput

    webcam = WebcamInput(device_id=0, fps=1.0, max_size=256)
    print("Webcam initialized")

    count = 0
    async for frame in webcam.stream():
        print(f"  Frame {count}: shape={frame.data.shape}, size={len(frame.to_base64())} bytes base64")
        count += 1
        if count >= 2:
            break

    await webcam.close()
    print("Webcam test PASSED\n")
    return True


async def test_openai():
    """Test OpenAI API with an image."""
    print("=" * 50)
    print("TEST 2: OpenAI Vision API")
    print("=" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  SKIPPED: OPENAI_API_KEY not set")
        return False

    print(f"  API key: {api_key[:10]}...")

    from openai import AsyncOpenAI
    import numpy as np
    from src.core.types import Frame

    client = AsyncOpenAI()

    # Create a simple test image (red square)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0]  # Red
    frame = Frame(data=test_image, source="test")

    print("  Sending test image to GPT-4o-mini...")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame.to_base64()}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": "What color is this image? Reply in one word."},
                    ],
                }
            ],
            max_tokens=10,
        )
        result = response.choices[0].message.content
        print(f"  Response: {result}")
        print("OpenAI test PASSED\n")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_model_wrapper():
    """Test our model wrapper."""
    print("=" * 50)
    print("TEST 3: Model Wrapper (OpenAIVisionModel)")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIPPED: OPENAI_API_KEY not set")
        return False

    from src.models import create_model
    from src.core.types import Frame
    import numpy as np

    model = create_model("openai", "gpt-4o-mini")
    print(f"  Created model: {model.provider}/{model.model_id}")

    # Create test frame
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[:, :] = [0, 255, 0]  # Green
    frame = Frame(data=test_image, source="test")

    print("  Calling model.analyze()...")

    try:
        response_text = ""
        async for event in model.analyze(
            frames=[frame],
            audio_transcript=None,
            tools=[],
            context=[],
            system_prompt="You describe images briefly.",
        ):
            print(f"    Event: {type(event).__name__}", end="")
            if hasattr(event, "content"):
                response_text += event.content
                print(f" - content: '{event.content[:50]}...'")
            else:
                print()

        print(f"  Full response: {response_text}")
        print("Model wrapper test PASSED\n")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_agent():
    """Test the full agent with webcam."""
    print("=" * 50)
    print("TEST 4: Full Agent Loop")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIPPED: OPENAI_API_KEY not set")
        return False

    from src.core.agent import AgentLoop
    from src.core.config import AgentConfig
    from src.core.types import Message
    from src.inputs.webcam import WebcamInput
    from src.memory.sliding_window import SlidingWindowMemory
    from src.models import create_model

    model = create_model("openai", "gpt-4o-mini")
    memory = SlidingWindowMemory(max_messages=5)

    agent = AgentLoop(
        model=model,
        memory=memory,
        config=AgentConfig(
            frame_batch_size=1,
            frame_interval_ms=100,  # Very short for testing
            system_prompt="Describe what you see in 5 words or less.",
        ),
    )

    webcam = WebcamInput(device_id=0, fps=0.5, max_size=256)

    print("  Running agent for up to 10 seconds...")

    events_received = []

    async def on_event(event):
        events_received.append(event)
        print(f"    Event: {type(event).__name__}", end="")
        if hasattr(event, "content"):
            print(f" - {event.content[:50]}...")
        else:
            print()

    # Run with timeout
    try:
        await asyncio.wait_for(
            agent.run(webcam, on_event=on_event),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        agent.stop()
        await webcam.close()

    print(f"  Received {len(events_received)} events")

    if events_received:
        print("Full agent test PASSED\n")
        return True
    else:
        print("Full agent test FAILED - no events received\n")
        return False


async def main():
    print("\nMultimodal Agent Debug Test Suite\n")

    results = {}

    # Test 1: Webcam
    try:
        results["webcam"] = await test_webcam()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        results["webcam"] = False

    # Test 2: OpenAI direct
    try:
        results["openai"] = await test_openai()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        results["openai"] = False

    # Test 3: Model wrapper
    try:
        results["model_wrapper"] = await test_model_wrapper()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results["model_wrapper"] = False

    # Test 4: Full agent
    try:
        results["full_agent"] = await test_full_agent()
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        results["full_agent"] = False

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
