#!/usr/bin/env python3
"""Benchmark example - Compare all model providers.

This example runs benchmarks across all available providers and
generates comparison tables for latency and cost.

Run:
    python examples/05_benchmark_providers.py

Requirements:
    Set API keys for providers you want to test:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    - GROQ_API_KEY
    - FIREWORKS_API_KEY
    - TOGETHER_API_KEY
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_model, list_models
from src.utils.benchmark import BenchmarkRunner, DEFAULT_SCENARIOS


async def main():
    """Run benchmarks across all available providers."""
    print("=" * 70)
    print("Multimodal Model Benchmark")
    print("=" * 70)
    print()

    # Show available models
    print("Checking available providers...")
    available_models = []

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available_models.extend([
            ("openai", "gpt-4o-mini"),
            ("openai", "gpt-4o"),
        ])
        print("  ✓ OpenAI (gpt-4o, gpt-4o-mini)")
    else:
        print("  ✗ OpenAI (set OPENAI_API_KEY)")

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.extend([
            ("anthropic", "claude-3-5-haiku-latest"),
            ("anthropic", "claude-3-5-sonnet-latest"),
        ])
        print("  ✓ Anthropic (claude-3-5-haiku, claude-3-5-sonnet)")
    else:
        print("  ✗ Anthropic (set ANTHROPIC_API_KEY)")

    # Google
    if os.getenv("GOOGLE_API_KEY"):
        available_models.extend([
            ("google", "gemini-1.5-flash"),
        ])
        print("  ✓ Google (gemini-1.5-flash)")
    else:
        print("  ✗ Google (set GOOGLE_API_KEY)")

    # Groq
    if os.getenv("GROQ_API_KEY"):
        available_models.extend([
            ("groq", "llama-3.2-11b-vision-preview"),
        ])
        print("  ✓ Groq (llama-3.2-11b-vision)")
    else:
        print("  ✗ Groq (set GROQ_API_KEY)")

    # Fireworks
    if os.getenv("FIREWORKS_API_KEY"):
        available_models.extend([
            ("fireworks", "firellava-13b"),
        ])
        print("  ✓ Fireworks (firellava-13b)")
    else:
        print("  ✗ Fireworks (set FIREWORKS_API_KEY)")

    # Together
    if os.getenv("TOGETHER_API_KEY"):
        available_models.extend([
            ("together", "llama-3.2-11b-vision"),
        ])
        print("  ✓ Together (llama-3.2-11b-vision)")
    else:
        print("  ✗ Together (set TOGETHER_API_KEY)")

    print()

    if not available_models:
        print("No providers available. Please set at least one API key.")
        return

    print(f"Running benchmarks on {len(available_models)} models...")
    print(f"Scenarios: {', '.join(s.name for s in DEFAULT_SCENARIOS)}")
    print()

    # Create benchmark runner
    runner = BenchmarkRunner()

    # Add available models
    for provider, model_id in available_models:
        try:
            model = create_model(provider, model_id)
            runner.add_model(model)
            print(f"  Added: {provider}/{model_id}")
        except Exception as e:
            print(f"  Failed to add {provider}/{model_id}: {e}")

    print()
    print("-" * 70)
    print("Starting benchmarks (this may take a few minutes)...")
    print("-" * 70)
    print()

    # Run benchmarks
    iterations = 5  # Reduce for faster testing
    results = await runner.run_all(iterations=iterations, warmup=1)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()

    # Print latency table
    print("LATENCY (p50, milliseconds)")
    print("-" * 70)
    print(runner.to_markdown_table(metric="p50"))
    print()

    # Print cost table
    print("COST (USD per benchmark run)")
    print("-" * 70)
    print(runner.to_cost_table())
    print()

    # Save results
    results_dir = Path(__file__).parent.parent / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "benchmark_results.json"
    runner.save_results(results_file)
    print(f"Results saved to: {results_file}")

    # Print detailed stats
    print()
    print("=" * 70)
    print("Detailed Statistics")
    print("=" * 70)

    for result in results:
        print(f"\n{result.provider}/{result.model_id} - {result.scenario}:")
        print(f"  Latency: p50={result.p50:.0f}ms, p95={result.p95:.0f}ms, avg={result.avg:.0f}ms")
        print(f"  Tokens: in={result.tokens_in}, out={result.tokens_out}")
        print(f"  Cost: ${result.cost_usd:.4f}")
        print(f"  Success Rate: {result.success_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
