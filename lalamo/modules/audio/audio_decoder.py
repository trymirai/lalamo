from dataclasses import dataclass
from typing import Self

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.common import ParameterTree
from lalamo.modules.common import LalamoModule


@dataclass(frozen=True)
class AudioDecoderConfig:
    pass


class AudioDecoder(LalamoModule[AudioDecoderConfig]):
    @property
    def activation_precision(self) -> DTypeLike:
        return jnp.float32

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],  # noqa: ARG002
    ) -> Self:
        return self

    def __call__(self, text_tokens: Array) -> Array:
        return jnp.zeros((1, 2, 3))
