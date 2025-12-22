from abc import abstractmethod

import jax
import jax.numpy as jnp
from attr import dataclass
from jaxtyping import Array, Float

from lalamo.modules.common import register_config_union

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "SiLU",
]


@dataclass(frozen=True)
class ActivationBase:
    @abstractmethod
    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]: ...


@dataclass(frozen=True)
class SiLU(ActivationBase):
    alpha: float = 1.0

    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
        return x / (1 + jnp.exp(-x * self.alpha))


@dataclass(frozen=True)
class GELU(ActivationBase):
    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
        return jax.nn.gelu(x)


@dataclass(frozen=True)
class Identity(ActivationBase):
    def __call__(self, x: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
        return x


Activation = SiLU | GELU | Identity


register_config_union(Activation)
