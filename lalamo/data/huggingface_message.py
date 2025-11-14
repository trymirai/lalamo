from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import cattrs
import polars as pl

from lalamo.message_processor import AssistantMessage, Message, UserMessage


@dataclass(frozen=True)
class HFMessage:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    role: str
    content: str

    @classmethod
    def from_dict(cls, obj: dict) -> Self:
        return cls._converter.structure(obj, cls)

    def as_message(self) -> Message:
        match self.role:
            case "user":
                return UserMessage(self.content)
            case "assistant":
                return AssistantMessage(None, self.content)
            case other:
                raise ValueError(f"Cannot convert {other} message")


def import_hf_parquet(path: Path | str) -> Iterable[list[Message]]:
    path = Path(path)

    dataframe = pl.scan_parquet(path).collect()

    for conversation in dataframe.get_column("conversation").shuffle(1337):
        yield [HFMessage.from_dict(message).as_message() for message in conversation]
