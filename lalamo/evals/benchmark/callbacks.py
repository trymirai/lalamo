from dataclasses import dataclass
from pathlib import Path

from evals.types import BenchmarkMetrics
from rich.console import Console

console = Console()


@dataclass
class BaseBenchmarkCallbacks:
    def started(self, eval_name: str, split: str, predictions_path: Path) -> None:
        pass

    def loading_predictions(self) -> None:
        pass

    def loading_ground_truth(self) -> None:
        pass

    def preparing_benchmark(self) -> None:
        pass

    def running_benchmark(self) -> None:
        pass

    def completed(self, metrics: BenchmarkMetrics) -> None:
        pass


@dataclass
class ConsoleCallbacks(BaseBenchmarkCallbacks):
    def started(self, eval_name: str, split: str, predictions_path: Path) -> None:
        console.print("[bold]Benchmark Configuration:[/bold]")
        console.print(f"  Eval: {eval_name}")
        console.print(f"  Split: {split}")
        console.print(f"  Predictions: {predictions_path}")
        console.print()

    def loading_predictions(self) -> None:
        console.print("  [cyan]1/4[/cyan] Loading predictions...")

    def loading_ground_truth(self) -> None:
        console.print("  [cyan]2/4[/cyan] Loading ground truth...")

    def preparing_benchmark(self) -> None:
        console.print("  [cyan]3/4[/cyan] Preparing benchmark data...")

    def running_benchmark(self) -> None:
        console.print("  [cyan]4/4[/cyan] Running benchmark...")

    def completed(self, metrics: BenchmarkMetrics) -> None:
        console.print()
        console.print("✓ Benchmark complete")
        console.print()
        console.print("[bold]═══ Benchmark Results ═══[/bold]")
        console.print(f"Eval: {metrics.eval_name}")
        console.print(f"Model: {metrics.model_name}")
        console.print(f"Split: {metrics.split}")
        console.print()
        console.print(f"[bold green]Overall Accuracy: {metrics.overall_accuracy:.2%}[/bold green]")
        console.print(f"Correct: {metrics.correct}/{metrics.total_examples}")
        console.print(f"Incorrect: {metrics.incorrect}/{metrics.total_examples}")

        if metrics.category_metrics:
            console.print()
            console.print("[bold]Category Breakdown:[/bold]")
            for category, accuracy in sorted(metrics.category_metrics.items()):
                console.print(f"  {category:20s} {accuracy:.2%}")

        if metrics.custom_metrics:
            console.print()
            console.print("[bold]Custom Metrics:[/bold]")
            for metric_name, value in metrics.custom_metrics.items():
                console.print(f"  {metric_name}: {value}")

