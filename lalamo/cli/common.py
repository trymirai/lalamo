import re
import sys
from enum import Enum
from pathlib import Path

import requests
import thefuzz.process
from click import Context as ClickContext
from click import Parameter as ClickParameter
from click import ParamType
from rich import box
from rich.console import Console
from rich.panel import Panel
from typer import Exit

from lalamo.model_import import ModelSpec
from lalamo.model_import.remote_registry import RegistryModel, fetch_available_models
from lalamo.model_registry import ModelRegistry

SCRIPT_NAME = Path(sys.argv[0]).name
DEFAULT_OUTPUT_DIR = Path("models")

console = Console()
err_console = Console(stderr=True)


class Precision(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ModelParser(ParamType):
    name: str = "Huggingface Model Repo"

    def convert(self, value: str, param: ClickParameter | None, ctx: ClickContext | None) -> ModelSpec:
        repo_to_model = ModelRegistry.build().repo_to_model
        result = repo_to_model.get(value)
        if result is None:
            closest_repo = _resolve_unique_match(value, list(repo_to_model))
            error_message_parts = [
                f'"{value}".',
            ]
            if closest_repo:
                error_message_parts.append(
                    f' Perhaps you meant "{closest_repo}"?',
                )
            error_message_parts.append(
                f"\n\nUse the `{SCRIPT_NAME} list-models` command to see the list of currently supported models.",
            )
            error_message = "".join(error_message_parts)
            return self.fail(error_message, param, ctx)
        return result


class RemoteModelParser(ParamType):
    name: str = "Pre-converted Model"

    def convert(self, value: str, param: ClickParameter | None, ctx: ClickContext | None) -> "RegistryModel":
        try:
            available_models = fetch_available_models()
        except (requests.RequestException, ValueError) as e:
            error_message = f"Failed to fetch model list from SDK. Check your internet connection.\n\nError: {e}"
            return self.fail(error_message, param, ctx)

        repo_to_model = {m.repo_id: m for m in available_models}
        model_spec = repo_to_model.get(value)
        if model_spec is None:
            unique_match = _resolve_unique_match(value, list(repo_to_model))
            if unique_match:
                model_spec = repo_to_model[unique_match]

        if model_spec is None:
            suggestions = _suggest_similar_models(value, available_models)
            error_message = f'Model "{value}" not found.'
            if suggestions:
                error_message += "\n\nDid you mean one of these?\n" + "\n".join(f"  - {s}" for s in suggestions)
            return self.fail(error_message, param, ctx)

        return model_spec


def _resolve_unique_match(
    query: str,
    repo_ids: list[str],
    min_score: float = 80,
    min_gap: float = 10,
) -> str | None:
    if not repo_ids:
        return None
    matches = sorted(thefuzz.process.extract(query, repo_ids), key=lambda m: m[1], reverse=True)
    if not matches or matches[0][1] < min_score:
        return None
    if len(matches) >= 2 and matches[0][1] - matches[1][1] < min_gap:
        return None
    return matches[0][0]


def _error(message: str) -> None:
    panel = Panel(message, box=box.ROUNDED, title="Error", title_align="left", border_style="red")
    err_console.print(panel)
    raise Exit(1)


def _suggest_similar_models(query: str, available_models: list[RegistryModel], limit: int = 3) -> list[str]:
    repo_ids = [m.repo_id for m in available_models]
    matches = thefuzz.process.extract(query, repo_ids, limit=limit)
    return [match[0] for match in matches if match[1] >= 50]


def _model_size_string_to_int(
    size_str: str,
    _regex: re.Pattern[str] = re.compile(r"(?P<number>(\d+)(\.\d*)?)(?P<suffix>[KMBT])"),
) -> float:
    match = _regex.match(size_str)
    factors = {
        "K": 1000**1,
        "M": 1000**2,
        "B": 1000**3,
        "T": 1000**4,
    }
    if match:
        return float(match.group("number")) * factors[match.group("suffix")]
    raise ValueError(f"Invalid size string: {size_str}")
