import jax
from jax import jit
from jaxtyping import Array, Float

__all__ = ["silu"]


@jit
def silu(x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return x * jax.nn.sigmoid(x)
