from abc import abstractmethod
from dataclasses import dataclass

from lalamo.weight_matrix import WeightMatrixSpec

__all__ = [
    "QuantizedSpec",
]


@dataclass(frozen=True)
class QuantizedSpec(WeightMatrixSpec):
    @property
    @abstractmethod
    def input_block_size(self) -> int: ...

    @property
    @abstractmethod
    def output_block_size(self) -> int: ...

    @property
    @abstractmethod
    def rate(self) -> float: ...

    @property
    @abstractmethod
    def distortion(self) -> float: ...
