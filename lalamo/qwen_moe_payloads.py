from __future__ import annotations

import json
from pathlib import Path

import cattrs

_CONVERTER = cattrs.Converter()
_CONVERTER.register_structure_hook(Path, lambda value, _: Path(value))
_CONVERTER.register_unstructure_hook(Path, lambda value: str(value))


def read_payload[T](path: Path, cls: type[T]) -> T:
    return _CONVERTER.structure(json.loads(path.read_text()), cls)


def write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CONVERTER.unstructure(payload), indent=2))


def structure_payload[T](payload: object, cls: type[T]) -> T:
    return _CONVERTER.structure(payload, cls)


def unstructure_payload(payload: object) -> object:
    return _CONVERTER.unstructure(payload)


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem or dataset


def alpha_label(alpha: float) -> str:
    return str(alpha)


def variant_label(alpha: float) -> str:
    return "baseline" if alpha == 0.0 else alpha_label(alpha)
