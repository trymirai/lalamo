from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from lalamo.models.common import BatchSizesComputedEvent

console = Console()


@dataclass
class BaseRunInferenceCallbacks:
    eval_repo: str
    model_path: Path
    limit: int | None
    batch_size: int | None
    vram_gb: float | None
    dataset_path: Path | None = None
    output_path: Path | None = None
    max_vram: int | None = None
    total_rows: int = 0

    def started(self) -> None:
        pass

    def inference_config_loaded(self, adapter_config: dict[str, Any], overrides: dict[str, Any]) -> None:
        pass

    def loading_datasets(self) -> None:
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
        console.print(f"  Eval: {self.eval_repo}")
        console.print(f"  Model: {self.model_path}")
        console.print(f"  Batch size: {self.batch_size or 'auto'}")
        console.print(f"  VRAM limit: {f'{self.vram_gb} GB' if self.vram_gb else 'auto-detect'}")
        console.print(f"  Limit: {self.limit or 'all'}")

    def inference_config_loaded(self, adapter_config: dict[str, Any], overrides: dict[str, Any]) -> None:
        if overrides:
            changes = [f"{k}: {adapter_config[k]} → {v}" for k, v in overrides.items()]
            console.print(f"  [dim]Inference config overrides: {', '.join(changes)}[/dim]")
        else:
            console.print(f"  [dim]Using adapter defaults for inference config[/dim]")
        console.print()
        console.print("[bold]Running inference...[/bold]")

    def loading_datasets(self) -> None:
        console.print("  [cyan]•[/cyan] Loading datasets...")

    def formatting_prompts(self) -> None:
        console.print("  [cyan]•[/cyan] Formatting prompts...")

    def preparing_input(self) -> None:
        console.print("  [cyan]•[/cyan] Preparing input...")

    def running_inference(self) -> None:
        console.print("  [cyan]•[/cyan] Running inference...")

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

