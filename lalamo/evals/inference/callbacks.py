from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from lalamo.evals.inference.engines import CustomAPIEngineConfig, LalamoEngineConfig
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks

console = Console()


@dataclass
class BaseRunInferenceCallbacks(BaseEngineCallbacks):
    def started(self) -> None:
        pass

    def inference_config_loaded(self, adapter_config: dict[str, Any], overrides: dict[str, Any]) -> None:
        pass

    def completed(self, predictions_path: Path, count: int) -> None:
        pass


@dataclass
class ConsoleRunInferenceCallbacks(BaseRunInferenceCallbacks):
    eval_repo: str
    model_path: Path | None
    limit: int | None
    engine_type: str | None = None
    engine_config: CustomAPIEngineConfig | LalamoEngineConfig | None = None

    _stack: ExitStack = field(default_factory=ExitStack)
    _progress: Progress | None = None
    _generation_task: TaskID | None = None

    def started(self) -> None:
        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Eval: {self.eval_repo}")

        if self.engine_type:
            console.print(f"  Engine: [cyan]{self.engine_type}[/cyan]")

        if self.engine_config:
            console.print("  [dim]Engine config:[/dim]")
            for key, value in vars(self.engine_config).items():
                if key == "api_key":
                    display_value = "***" if value else None
                else:
                    display_value = value
                console.print(f"   - [dim]{key}: {display_value}[/dim]")

        if self.model_path:
            console.print(f"  Model: {self.model_path}")

        console.print(f"  Limit: {self.limit or 'all'}")

    def inference_config_loaded(self, adapter_config: dict[str, Any], overrides: dict[str, Any]) -> None:
        console.print("  [dim]Inference config (adapter defaults):[/dim]")
        for key, value in adapter_config.items():
            console.print(f"   - [dim]{key}: {value}[/dim]")

        if overrides:
            console.print("  [dim]Inference config overrides:[/dim]")
            for key, value in overrides.items():
                console.print(f"   - [dim]{key}: {adapter_config[key]} → {value}[/dim]")

        console.print()
        console.print("[bold]Running inference...[/bold]")

    def unsupported_inference_params(self, params: list[str]) -> None:
        console.print("[yellow]Warning: unsupported inference config parameters:[/yellow]")
        for param in params:
            console.print(f" - {param}")

    def generation_started(self, total_rows: int) -> None:
        self._progress = self._stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True,
            ),
        )
        self._generation_task = self._progress.add_task(
            "🔮 [cyan]Generating replies...[/cyan]",
            total=total_rows,
        )

    def generation_progress(self, rows_processed: int) -> None:
        if self._progress is not None and self._generation_task is not None:
            self._progress.update(self._generation_task, completed=rows_processed)

    def finished_generation(self) -> None:
        if self._progress is not None and self._generation_task is not None:
            self._progress.update(self._generation_task, description="✅ Completed")
        self._stack.close()

    def completed(self, predictions_path: Path, count: int) -> None:
        console.print()
        console.print(f"[green]✓[/green] Saved {count} predictions to: {predictions_path}")

