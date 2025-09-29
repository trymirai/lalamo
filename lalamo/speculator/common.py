from abc import abstractmethod
from collections.abc import Iterable
from typing import Self


class Speculator:
    @abstractmethod
    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, blob: bytes) -> Self:
        raise NotImplementedError
