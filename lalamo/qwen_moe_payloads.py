from __future__ import annotations

import json
from pathlib import Path

import cattrs

_converter = cattrs.Converter()
_converter.register_unstructure_hook(Path, lambda value: str(value))


def write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_converter.unstructure(payload), indent=2))
