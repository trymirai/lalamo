from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl

from lalamo.models.chat_codec import Message, ToolSchema, message_converter


@dataclass(frozen=True)
class HFConversation:
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] | None

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return message_converter.structure(obj, cls)


def load_hf_parquet(path: Path | str) -> pl.LazyFrame:
    return pl.scan_parquet(Path(path)).drop("metadata")


def shuffle_dataset(frame: pl.LazyFrame, seed: int = 1337) -> pl.DataFrame:
    return frame.collect().sample(fraction=1.0, shuffle=True, seed=seed)
