from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Annotated, Any, ClassVar, Self

import msgpack
from annotated_types import MinLen
from cattrs.preconf.msgpack import MsgpackConverter
from cattrs.preconf.msgpack import make_converter as make_msgpack_converter


@dataclass(frozen=True)
class LalamoCompletion:
    _converter: ClassVar[MsgpackConverter] = make_msgpack_converter()

    prefix_token_ids: list[int]
    completion_token_ids: list[int]

    def serialize(self) -> bytes:
        return self._converter.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes | IO[bytes]) -> Self:
        if isinstance(data, bytes):
            obj: Any = msgpack.unpackb(data, strict_map_key=False)
        else:
            obj = msgpack.unpack(data, strict_map_key=False)

        return cls._converter.structure(obj, cls)

    @classmethod
    def deserialize_many(cls, data: bytes | IO[bytes]) -> Iterator[Self]:
        if isinstance(data, bytes):
            unpacker = msgpack.Unpacker(strict_map_key=False)
            unpacker.feed(data)
        else:
            unpacker = msgpack.Unpacker(file_like=data, strict_map_key=False)

        for obj in unpacker:
            yield cls._converter.structure(obj, cls)


def save_completions(path: Path | str, completions: Iterable[LalamoCompletion]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fd:
        for completion in completions:
            fd.write(completion.serialize())


def load_completions(path: Path | str) -> list[LalamoCompletion]:
    return list(iter_completions(path))


def iter_completions(path: Path | str) -> Iterator[LalamoCompletion]:
    with Path(path).open("rb") as fd:
        yield from LalamoCompletion.deserialize_many(fd)
