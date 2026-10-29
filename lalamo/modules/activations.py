from abc import abstractmethod
from dataclasses import dataclass

import jax
import numpy as np
from jaxtyping import Array, Float

from lalamo.module import Keychain

from .flow import Context, Op

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "SiLU",
    "Sigmoid",
]


@dataclass(frozen=True)
class Activation(Op):
    @abstractmethod
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...

    def forward(self, x: Array, *operands: Array, ctx: Context, keychain: Keychain) -> Array:  # noqa: ARG002
        () = operands
        return self(x)


@dataclass(frozen=True)
class SiLU(Activation):
    alpha: float = 1.0

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        # JAX's strict policy also rejects explicit BF16/FP32 dtype promotion; NumPy resolves this static dtype.
        promoted = x.astype(np.promote_types(x.dtype, np.float32))
        return (promoted * jax.nn.sigmoid(self.alpha * promoted)).astype(x.dtype)


@dataclass(frozen=True)
class Sigmoid(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.sigmoid(x)


@dataclass(frozen=True)
class GELU(Activation):
    approximate: bool = True

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        promoted = x.astype(np.promote_types(x.dtype, np.float32))
        return jax.nn.gelu(promoted, approximate=self.approximate).astype(x.dtype)


@dataclass(frozen=True)
class Identity(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x
