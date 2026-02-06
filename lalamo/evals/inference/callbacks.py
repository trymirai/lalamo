from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks

console = Console()


@dataclass
class BaseRunInferenceCallbacks(BaseEngineCallbacks):
    eval_repo: str
    model_path: Path
    limit: int | None
    batch_size: int | None
    vram_gb: float | None

    def started(self) -> None:
        pass

    def inference_config_loaded(self, adapter_config: dict[str, Any], overrides: dict[str, Any]) -> None:
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

    def completed(self, predictions_path: Path, count: int) -> None:
        console.print()
        console.print(f"[green]✓[/green] Saved {count} predictions to: {predictions_path}")

