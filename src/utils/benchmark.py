"""Benchmarking utilities for model comparison."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from src.core.types import Frame, Message, ToolDefinition
from src.models.base import VisionLanguageModel


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    provider: str
    model_id: str
    scenario: str
    latency_ms: list[float]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    errors: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def p50(self) -> float:
        """Median latency."""
        return median(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        if not self.latency_ms:
            return 0.0
        sorted_lat = sorted(self.latency_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.latency_ms:
            return 0.0
        sorted_lat = sorted(self.latency_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg(self) -> float:
        """Average latency."""
        return mean(self.latency_ms) if self.latency_ms else 0.0

    @property
    def std(self) -> float:
        """Standard deviation of latency."""
        return stdev(self.latency_ms) if len(self.latency_ms) > 1 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = len(self.latency_ms) + self.errors
        return (len(self.latency_ms) / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "avg": self.avg,
            "std": self.std,
            "success_rate": self.success_rate,
        }


@dataclass
class BenchmarkScenario:
    """A benchmark test scenario."""

    name: str
    description: str
    num_frames: int
    prompt: str
    tools: list[ToolDefinition] = field(default_factory=list)


# Default benchmark scenarios
DEFAULT_SCENARIOS = [
    BenchmarkScenario(
        name="single_frame",
        description="Basic single image analysis",
        num_frames=1,
        prompt="Describe what you see in this image briefly.",
    ),
    BenchmarkScenario(
        name="multi_frame",
        description="Analyze 4 frames for changes",
        num_frames=4,
        prompt="Analyze these 4 frames and describe any changes or movement.",
    ),
    BenchmarkScenario(
        name="detailed_analysis",
        description="Detailed single image analysis",
        num_frames=1,
        prompt="Provide a detailed analysis including: objects, actions, text visible, scene context, and any potential concerns.",
    ),
    BenchmarkScenario(
        name="tool_calling",
        description="Image analysis with tool calling",
        num_frames=1,
        prompt="Analyze this image. If you see any issues or concerns, use the alert tool to report them.",
        tools=[
            ToolDefinition(
                name="send_alert",
                description="Send an alert about an issue",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    },
                    "required": ["message", "severity"],
                },
            )
        ],
    ),
]


class BenchmarkRunner:
    """Run benchmarks across multiple models and scenarios.

    Example:
        runner = BenchmarkRunner()

        # Add models to test
        runner.add_model(OpenAIVisionModel("gpt-4o-mini"))
        runner.add_model(AnthropicVisionModel("claude-3-5-haiku-latest"))

        # Run benchmarks
        results = await runner.run_all(iterations=10)

        # Save results
        runner.save_results("benchmarks/results/results.json")

        # Generate markdown table
        table = runner.to_markdown_table()
    """

    def __init__(
        self,
        scenarios: list[BenchmarkScenario] | None = None,
        test_frames: list[Frame] | None = None,
    ):
        """Initialize benchmark runner.

        Args:
            scenarios: Test scenarios (uses defaults if not provided)
            test_frames: Test frames to use (generates random if not provided)
        """
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.test_frames = test_frames or self._generate_test_frames()
        self.models: list[VisionLanguageModel] = []
        self.results: list[BenchmarkResult] = []

    def add_model(self, model: VisionLanguageModel) -> None:
        """Add a model to benchmark."""
        self.models.append(model)

    async def run_all(
        self,
        iterations: int = 10,
        warmup: int = 1,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks.

        Args:
            iterations: Number of iterations per scenario
            warmup: Number of warmup runs (not counted)

        Returns:
            List of BenchmarkResult objects
        """
        self.results = []

        for model in self.models:
            for scenario in self.scenarios:
                print(f"Benchmarking {model.provider}/{model.model_id} - {scenario.name}")

                result = await self._run_benchmark(
                    model, scenario, iterations, warmup
                )
                self.results.append(result)

        return self.results

    async def _run_benchmark(
        self,
        model: VisionLanguageModel,
        scenario: BenchmarkScenario,
        iterations: int,
        warmup: int,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        frames = self.test_frames[: scenario.num_frames]
        latencies = []
        errors = 0
        total_tokens_in = 0
        total_tokens_out = 0

        # Warmup runs
        for _ in range(warmup):
            try:
                async for _ in model.analyze(
                    frames=frames,
                    audio_transcript=None,
                    tools=scenario.tools,
                    context=[],
                    system_prompt="",
                ):
                    pass
            except Exception:
                pass

        # Actual benchmark runs
        for i in range(iterations):
            try:
                start = time.perf_counter()
                response_text = ""

                async for event in model.analyze(
                    frames=frames,
                    audio_transcript=None,
                    tools=scenario.tools,
                    context=[],
                    system_prompt="",
                ):
                    if isinstance(event, Message):
                        response_text += event.content

                end = time.perf_counter()
                latencies.append((end - start) * 1000)

                # Estimate tokens
                tokens_in = self._estimate_tokens(frames, scenario.prompt)
                tokens_out = len(response_text) // 4
                total_tokens_in += tokens_in
                total_tokens_out += tokens_out

            except Exception as e:
                print(f"  Error on iteration {i}: {e}")
                errors += 1

        # Calculate cost
        cost = (
            (total_tokens_in / 1000) * model.cost_per_1k_input_tokens
            + (total_tokens_out / 1000) * model.cost_per_1k_output_tokens
        )

        return BenchmarkResult(
            provider=model.provider,
            model_id=model.model_id,
            scenario=scenario.name,
            latency_ms=latencies,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            cost_usd=cost,
            errors=errors,
        )

    def _estimate_tokens(self, frames: list[Frame], prompt: str) -> int:
        """Estimate input tokens."""
        # Rough estimate: 85 tokens per 512x512 image for low detail
        image_tokens = len(frames) * 85
        text_tokens = len(prompt) // 4
        return image_tokens + text_tokens

    def _generate_test_frames(self, count: int = 4) -> list[Frame]:
        """Generate random test frames."""
        import numpy as np

        frames = []
        for i in range(count):
            # Generate a random colored frame with some patterns
            data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            frames.append(
                Frame(data=data, source=f"test_frame_{i}")
            )
        return frames

    def save_results(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_markdown_table(self, metric: str = "p50") -> str:
        """Generate a markdown table of results.

        Args:
            metric: Metric to display (p50, p95, p99, avg)

        Returns:
            Markdown table string
        """
        if not self.results:
            return "No results to display"

        # Group by scenario
        scenarios = sorted(set(r.scenario for r in self.results))
        models = sorted(set((r.provider, r.model_id) for r in self.results))

        # Build header
        header = "| Provider | Model |"
        for scenario in scenarios:
            header += f" {scenario} |"
        header += "\n"

        # Build separator
        separator = "|---|---|"
        for _ in scenarios:
            separator += "---:|"
        separator += "\n"

        # Build rows
        rows = ""
        for provider, model_id in models:
            row = f"| {provider} | {model_id} |"
            for scenario in scenarios:
                result = next(
                    (r for r in self.results
                     if r.provider == provider
                     and r.model_id == model_id
                     and r.scenario == scenario),
                    None
                )
                if result:
                    value = getattr(result, metric)
                    row += f" {value:.0f}ms |"
                else:
                    row += " - |"
            rows += row + "\n"

        return header + separator + rows

    def to_cost_table(self) -> str:
        """Generate a markdown table of costs."""
        if not self.results:
            return "No results to display"

        scenarios = sorted(set(r.scenario for r in self.results))
        models = sorted(set((r.provider, r.model_id) for r in self.results))

        header = "| Provider | Model |"
        for scenario in scenarios:
            header += f" {scenario} |"
        header += "\n"

        separator = "|---|---|"
        for _ in scenarios:
            separator += "---:|"
        separator += "\n"

        rows = ""
        for provider, model_id in models:
            row = f"| {provider} | {model_id} |"
            for scenario in scenarios:
                result = next(
                    (r for r in self.results
                     if r.provider == provider
                     and r.model_id == model_id
                     and r.scenario == scenario),
                    None
                )
                if result:
                    row += f" ${result.cost_usd:.4f} |"
                else:
                    row += " - |"
            rows += row + "\n"

        return header + separator + rows
