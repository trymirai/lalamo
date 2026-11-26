from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

from .utils import apply_soft_capping

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "SiLU",
    "Sigmoid",
]


@dataclass(frozen=True)
class Activation(LalamoConfig, RegistryABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...


@dataclass(frozen=True)
class SiLU(Activation):
    alpha: float = 1.0
    soft_cap: float | None = None

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        if self.soft_cap is None:
            return x * jax.nn.sigmoid(self.alpha * x)
        upcast_x = x.astype(jnp.float32)
        return (apply_soft_capping(upcast_x, self.soft_cap) * jax.nn.sigmoid(self.alpha * upcast_x)).astype(x.dtype)


@dataclass(frozen=True)
class Sigmoid(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.sigmoid(x)


@dataclass(frozen=True)
class GELU(Activation):
    approximate: bool = True

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.gelu(x.astype(jnp.float32), approximate=self.approximate).astype(x.dtype)


@dataclass(frozen=True)
class Identity(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x
