from collections.abc import Iterable
from dataclasses import dataclass
from typing import IO, Any, ClassVar, Self

import msgpack
from cattrs.preconf.msgpack import MsgpackConverter
from cattrs.preconf.msgpack import make_converter as make_msgpack_converter


@dataclass(frozen=True)
class LalamoCompletion:
    _converter: ClassVar[MsgpackConverter] = make_msgpack_converter()

    prefix_token_ids: list[int]
    completion_token_ids: list[int]
    completion_token_logits: list[dict[int, float]]

    def __post_init__(self) -> None:
        if len(self.completion_token_ids) != len(self.completion_token_logits):
            raise ValueError(f"({len(self.completion_token_ids)=}) != ({len(self.completion_token_logits)=})")

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
    def deserialize_many(cls, data: bytes | IO[bytes]) -> Iterable[Self]:
        if isinstance(data, bytes):
            unpacker = msgpack.Unpacker(strict_map_key=False)
            unpacker.feed(data)
        else:
            unpacker = msgpack.Unpacker(file_like=data, strict_map_key=False)

        for obj in unpacker:
            yield cls._converter.structure(obj, cls)
