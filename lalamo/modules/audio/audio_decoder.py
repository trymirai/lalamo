from abc import abstractmethod
from typing import Self

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule


class AudioDecoder[ConfigT](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @abstractmethod
    def export_weights(self) -> ParameterTree[Array]: ...

    @abstractmethod
    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self: ...

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @abstractmethod
    def audio_from_codes(self, indices: Array) -> Array: ...
