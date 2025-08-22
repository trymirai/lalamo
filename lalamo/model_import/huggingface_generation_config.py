import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cattrs

__all__ = ["HFGenerationConfig"]


@dataclass(frozen=True)
class HFGenerationConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)
    _converter.register_structure_hook(int | list[int] | None, lambda v, _: v)

    # -------- identity / bookkeeping --------
    _from_model_config: bool | None = None  # some Mistral & DeepSeek models
    transformers_version: str | None = None  # library version that saved the file

    # -------- special-token ids -------------
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    # -------- backend hints -----------------
    cache_implementation: str | None = None  # “hybrid” for Gemma 3/2

    # -------- sampling strategy -------------
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None

    # -------- length limits -----------------
    max_length: int | None = None  # seen in Llama 3, Gemma 2/3

    @classmethod
    def from_json(cls, json_path: Path | str) -> "HFGenerationConfig":
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)
