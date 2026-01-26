from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.modules.common import LalamoConfig, RegistryABC

__all__ = [
    "GELU",
    "ActivationBase",
    "Identity",
    "SiLU",
]


@dataclass(frozen=True)
class ActivationBase(LalamoConfig, RegistryABC):
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
