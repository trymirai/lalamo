from lalamo.common import LalamoWarning
import os
import traceback as tb
import warnings

from .common import ModelSpec

from importlib.metadata import entry_points
from typing import Callable, Dict, Any

__all__ = ["THIRD_PARTY_MODELS"]


def load_plugins(group: str) -> dict[str, dict]:
    entries = {entry.name: entry.load for entry in entry_points().select(group=group)}

    specs, failed_plugins = set(), set()
    for entry_name, entry_load_func in entries.items():
        try:
            specs.add(entry_load_func())
        except Exception:  # noqa: BLE001
            failed_plugins.add(entry_name)
            if os.getenv("LALAMO_DEBUG"):
                warnings.warn(
                    f"\033[31m{entry_name} failed\033[0m with: {tb.format_exc()}\n",
                    LalamoWarning,
                    stacklevel=2,
                )
    if failed_plugins:
        warnings.warn(
            f"The following lalamo plugins have failed to import: {failed_plugins}. "
            "To see detailed log set the environment variable LALAMO_DEBUG to something truthy.",
            LalamoWarning,
            stacklevel=2,
        )
    return specs


THIRD_PARTY_MODELS: list[ModelSpec] = load_plugins("lalamo_plugins.specs.v1")
