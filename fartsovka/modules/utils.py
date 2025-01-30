import jax
from jaxtyping import Array, Float

__all__ = [
    "apply_soft_capping",
]


def apply_soft_capping(
    values: Float[Array, "*"],
    soft_cap: float,
) -> Float[Array, "dst_tokens src_tokens"]:
    return jax.nn.tanh(values / soft_cap) * soft_cap
