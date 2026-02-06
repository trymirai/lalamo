from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Confirm
from typer import Exit

console = Console()


@dataclass
class BaseConversionCallbacks:
    eval_repo: str
    output_dir: Path

    def output_dir_exists(self) -> None:
        pass

    def started(self) -> None:
        pass

    def downloading_file(self, filename: str) -> None:
        pass

    def finished_downloading_file(self, filename: str) -> None:
        pass

    def saving_dataset(self) -> None:
        pass

    def finished(self) -> None:
        pass


@dataclass
class ConsoleCallbacks(BaseConversionCallbacks):
    overwrite: bool = False

    stack: ExitStack = field(default_factory=ExitStack)
    progress: Progress | None = None
    downloading_tasks: dict[str, TaskID] = field(default_factory=dict)

    def started(self) -> None:
        console.print(f"🚀 Converting eval dataset [cyan]{self.eval_repo}[/cyan].")

        self.progress = self.stack.enter_context(
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ),
        )

    def output_dir_exists(self) -> None:
        if not self.overwrite and not Confirm().ask(
            rf"⚠️ Output directory [cyan]{self.output_dir}[/cyan] already exists. Continue?",
        ):
            raise Exit(0)

    def downloading_file(self, filename: str) -> None:
        assert self.progress is not None
        self.downloading_tasks[filename] = self.progress.add_task(f"Retrieving {filename}...")

    def finished_downloading_file(self, filename: str) -> None:
        assert self.progress is not None
        self.progress.remove_task(self.downloading_tasks[filename])

    def saving_dataset(self) -> None:
        pass

    def finished(self) -> None:
        if self.progress is not None:
            self.stack.close()
        console.print(f"✅ Dataset converted successfully to [cyan]{self.output_dir}[/cyan]")
