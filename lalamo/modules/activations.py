from abc import abstractmethod
from dataclasses import dataclass

import jax
from jaxtyping import Array, Float

from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "SiLU",
]


@dataclass(frozen=True)
class Activation(LalamoConfig, RegistryABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...


@dataclass(frozen=True)
class SiLU(Activation):
    alpha: float = 1.0

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * jax.nn.sigmoid(self.alpha * x)


@dataclass(frozen=True)
class GELU(Activation):
    approximate: bool = True

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.gelu(x, approximate=self.approximate)


@dataclass(frozen=True)
class Identity(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x
