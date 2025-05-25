from enum import Enum

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float

__all__ = [
    "Activation",
    "silu",
]


@jit
def silu(x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return x / (1 + jnp.exp(-x))


class Activation(Enum):
    SILU = "silu"
    GELU = "gelu"

    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
        return ACTIVATION_FUNCTIONS[self](x)


ACTIVATION_FUNCTIONS = {
    Activation.SILU: silu,
    Activation.GELU: jax.nn.gelu,
}
