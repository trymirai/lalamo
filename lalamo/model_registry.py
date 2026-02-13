import functools
import os
import traceback as tb
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points

from lalamo.common import LalamoWarning
from lalamo.model_import.model_specs import ALL_MODEL_LISTS
from lalamo.model_import.model_specs.common import ModelSpec, build_quantized_models

__all__ = [
    "ModelRegistry",
]


def load_third_party_specs(group: str) -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    failed_plugins: set[str] = set()

    for entry_point in entry_points().select(group=group):
        try:
            plugin_specs = entry_point.load()()
            if not all(isinstance(s, ModelSpec) for s in plugin_specs):
                failed_plugins.add(entry_point.name)
                continue
            specs.extend(plugin_specs)
        except Exception:  # noqa: BLE001
            failed_plugins.add(entry_point.name)
            if os.getenv("LALAMO_VERBOSE"):
                warnings.warn(
                    f"\033[31m{entry_point.name} failed\033[0m with: {tb.format_exc()}\n",
                    LalamoWarning,
                    stacklevel=2,
                )

    if failed_plugins:
        warnings.warn(
            f"The following lalamo plugins have failed to import: {failed_plugins}. "
            "To see detailed log set the environment variable LALAMO_VERBOSE to something truthy.",
            LalamoWarning,
            stacklevel=2,
        )
    return tuple(specs)


@dataclass(frozen=True)
class ModelRegistry:
    models: tuple[ModelSpec, ...]
    repo_to_model: Mapping[str, ModelSpec]

    @classmethod
    @functools.cache
    def build(cls, allow_third_party_plugins: bool = True) -> "ModelRegistry":
        base_models = [model for model_list in ALL_MODEL_LISTS for model in model_list]
        quantized_models = build_quantized_models(base_models)
        models = tuple(base_models + quantized_models)

        if allow_third_party_plugins:
            models += load_third_party_specs("lalamo_plugins.specs.v1")

        return ModelRegistry(models, {model.repo: model for model in models})
