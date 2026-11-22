from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "SiLU",
    "Sigmoid",
    "SoftCappedSiLU",
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
class SoftCappedSiLU(Activation):
    """SiLU whose linear factor is soft-capped while the sigmoid sees the raw input."""

    soft_cap: float

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        upcasted = x.astype(jnp.float32)
        capped = jnp.tanh(upcasted / self.soft_cap) * self.soft_cap
        return (capped * jax.nn.sigmoid(upcasted)).astype(x.dtype)


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
