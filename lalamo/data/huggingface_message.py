import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
import polars as pl

from lalamo.message_processor import AssistantMessage, Message, UserMessage


@dataclass(frozen=True)
class HFMessage:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(dict | list | str | None, lambda v, _: v)
    role: str
    content: dict | list | str | None

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls._converter.structure(obj, cls)

    def as_message(self) -> Message:
        content = self.content
        match self.role:
            case "user":
                return UserMessage(content)
            case "assistant":
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                return AssistantMessage(None, content)
            case other:
                raise ValueError(f"Cannot convert {other} message")


def load_hf_parquet(path: Path | str) -> pl.LazyFrame:
    path = Path(path)
    return pl.scan_parquet(path)


def shuffle_dataset(frame: pl.LazyFrame, seed: int = 1337) -> pl.DataFrame:
    return frame.collect().sample(fraction=1.0, shuffle=True, seed=seed)
