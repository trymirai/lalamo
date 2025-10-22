from abc import abstractmethod

import jax
import jax.numpy as jnp
from attr import dataclass
from jaxtyping import Array, Float

from lalamo.modules.common import register_config_union

__all__ = [
    "GELU",
    "Activation",
    "SiLU",
]

def activation_from_str_id(str_id: str) -> type["ActivationBase"]:
    str_id = str_id.lower()
    assert str_id in [a.lower() for a in __all__], f"Unknown type of activation: {str_id}"
    if str_id == "gelu":
        return GELU
    elif str_id == "silu":
        return SiLU
    else:
        raise Exception("Should not be here")

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


Activation = SiLU | GELU


register_config_union(Activation)
