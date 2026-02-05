from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from lalamo.models.common import BatchSizesComputedEvent

console = Console()


@dataclass
class BaseRunInferenceCallbacks:
    eval_name: str
    model_path: Path
    num_few_shot: int
    category: str | None
    max_examples: int | None
    batch_size: int | None
    vram_gb: float | None
    dataset_path: Path | None = None
    output_path: Path | None = None
    max_vram: int | None = None
    total_rows: int = 0

    def started(self) -> None:
        pass

    def loading_test_dataset(self) -> None:
        pass

    def loading_validation_dataset(self) -> None:
        pass

    def skipped_validation_dataset(self) -> None:
        pass

    def formatting_prompts(self) -> None:
        pass

    def preparing_input(self) -> None:
        pass

    def running_inference(self) -> None:
        pass

    def loading_model(self) -> None:
        pass

    def batch_sizes_computed(self, event: BatchSizesComputedEvent) -> None:
        pass

    def generation_progress(self, rows_processed: int) -> None:
        pass

    def parsing_output(self) -> None:
        pass

    def completed(self, predictions_path: Path, count: int) -> None:
        pass


@dataclass
class ConsoleRunInferenceCallbacks(BaseRunInferenceCallbacks):
    def started(self) -> None:
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Eval: {self.eval_name}")
        console.print(f"  Model: {self.model_path}")
        console.print(f"  Few-shot (k): {self.num_few_shot}")
        console.print(f"  Batch size: {self.batch_size or 'auto'}")
        console.print(f"  VRAM limit: {f'{self.vram_gb} GB' if self.vram_gb else 'auto-detect'}")
        console.print(f"  Category: {self.category or 'all'}")
        console.print(f"  Max examples: {self.max_examples or 'all'}")
        console.print()
        console.print(f"[bold]Running {self.num_few_shot}-shot inference...[/bold]")

    def loading_test_dataset(self) -> None:
        console.print("  [cyan]1/5[/cyan] Loading test dataset...")

    def loading_validation_dataset(self) -> None:
        console.print("  [cyan]2/5[/cyan] Loading validation dataset...")

    def skipped_validation_dataset(self) -> None:
        console.print("  [cyan]2/5[/cyan] Skipped (0-shot mode)")

    def formatting_prompts(self) -> None:
        console.print("  [cyan]3/5[/cyan] Formatting prompts...")

    def preparing_input(self) -> None:
        console.print("  [cyan]4/5[/cyan] Preparing input...")

    def running_inference(self) -> None:
        console.print("  [cyan]5/5[/cyan] Running inference...")

    def loading_model(self) -> None:
        console.print("      Loading model...")

    def batch_sizes_computed(self, _event: BatchSizesComputedEvent) -> None:
        console.print("      Batch sizes computed, warming up...")

    def generation_progress(self, rows_processed: int) -> None:
        actual_processed = rows_processed + 1
        report_threshold = max(1, self.total_rows // 10)
        if (
            actual_processed in (1, self.total_rows)
            or actual_processed - getattr(self, "_last_reported", 0) >= report_threshold
        ):
            percentage = (actual_processed / self.total_rows * 100) if self.total_rows > 0 else 0
            console.print(
                f"      Generating... {actual_processed}/{self.total_rows} ({percentage:.0f}%)",
            )
            self._last_reported = actual_processed

    def parsing_output(self) -> None:
        console.print("      Parsing outputs...")

    def completed(self, predictions_path: Path, count: int) -> None:
        console.print()
        console.print(f"[green]✓[/green] Saved {count} predictions to: {predictions_path}")

