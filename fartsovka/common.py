from typing import Any

import jax.numpy as jnp

__all__ = ["DEFAULT_PRECISION", "DType", "ParameterPath"]


type DType = jnp.dtype | Any  # This is a hack to make the typechecker happy with internals of JAX.


DEFAULT_PRECISION: DType = jnp.float32


class ParameterPath(str):
    __slots__ = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.split("."))

    def __truediv__(self, other: str | int) -> "ParameterPath":
        if not self:
            return ParameterPath(str(other))
        return ParameterPath(self + "." + str(other))
