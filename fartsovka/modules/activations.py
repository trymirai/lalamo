from enum import Enum

import jax
from jax import jit
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import ParameterDict, DType
from fartsovka.samples import ModuleSample

__all__ = [
    "Activation",
    "silu",
]


@jit
def silu(x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return x * jax.nn.sigmoid(x)


class Activation(Enum):
    SILU = "silu"
    GELU = "gelu"

    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
        return ACTIVATION_FUNCTIONS[self](x)

    def export_samples(self, shape: [int], precision: DType, key: PRNGKeyArray) -> ParameterDict:
        x = jax.random.uniform(
            key,
            shape,
            minval=-2,
            maxval=2,
            dtype=precision,
        )
        y = self(x)
        sample = ModuleSample(inputs=(x,), outputs=(y,))
        return ParameterDict(value=sample.export())


ACTIVATION_FUNCTIONS = {
    Activation.SILU: silu,
    Activation.GELU: jax.nn.gelu,
}
