from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.modules.common import (
    LalamoModule,
)

__all__ = [
    "LinearBase",
    "LinearConfigBase",
]


class LinearBase[ConfigT: LinearConfigBase](LalamoModule[ConfigT]):
    output_dims: tuple[int, ...] = eqx.field(static=True)

    @property
    @abstractmethod
    def mixture_size(self) -> int | None: ...

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @property
    @abstractmethod
    def has_biases(self) -> bool: ...

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, " in_channels"],
    ) -> tuple[Float[Array, " out_channels"], ...]: ...

    def __post_init__(self) -> None:
        assert isinstance(self.output_dims, tuple)

    @staticmethod
    def _get_split_points(output_dims: Sequence[int]) -> tuple[int, ...]:
        result = []
        last_split_point = 0
        for dim in output_dims[:-1]:
            last_split_point += dim
            result.append(last_split_point)
        return tuple(result)


@dataclass(frozen=True)
class LinearConfigBase(ABC):
    @abstractmethod
    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> LinearBase: ...

    @abstractmethod
    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> LinearBase: ...

    @abstractmethod
    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...

    @abstractmethod
    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> LinearBase: ...
