from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.types import BenchmarkMetrics
from rich.console import Console

console = Console()


def _format_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_value(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _format_metrics(metrics: dict[str, Any]) -> None:
    for key, value in sorted(metrics.items()):
        if isinstance(value, dict):
            console.print(f"[bold]{_format_key(key)}:[/bold]")
            for sub_key, sub_value in sorted(value.items()):
                console.print(f"  {sub_key:20s} {_format_value(sub_value)}")
        else:
            console.print(f"{_format_key(key)}: {_format_value(value)}")


@dataclass
class BaseBenchmarkCallbacks:
    def started(self, eval_name: str, split: str, predictions_path: Path) -> None: ...

    def loading_predictions(self) -> None: ...

    def preparing_benchmark(self) -> None: ...

    def running_benchmark(self) -> None: ...

    def completed(self, metrics: BenchmarkMetrics) -> None: ...


@dataclass
class ConsoleCallbacks(BaseBenchmarkCallbacks):
    def started(self, eval_name: str, split: str, predictions_path: Path) -> None:
        console.print("[bold]Benchmark Configuration:[/bold]")
        console.print(f"  Eval: {eval_name}")
        console.print(f"  Split: {split}")
        console.print(f"  Predictions: {predictions_path}")
        console.print()

    def loading_predictions(self) -> None:
        console.print("  [cyan]1/3[/cyan] Loading predictions...")

    def preparing_benchmark(self) -> None:
        console.print("  [cyan]2/3[/cyan] Preparing benchmark data...")

    def running_benchmark(self) -> None:
        console.print("  [cyan]3/3[/cyan] Running benchmark...")

    def completed(self, metrics: BenchmarkMetrics) -> None:
        console.print()
        console.print("✓ Benchmark complete")
        console.print()
        console.print("[bold]═══ Benchmark Results ═══[/bold]")
        console.print(f"Eval: {metrics.eval_name}")
        console.print(f"Model: {metrics.model_name}")
        console.print(f"Split: {metrics.split}")
        console.print(f"Engine: {metrics.inference_engine}")
        console.print(f"Samples: {metrics.num_samples}")
        console.print()

        _format_metrics(metrics.metrics)

