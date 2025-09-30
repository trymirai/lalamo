from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.linear.full_precision import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.quantization.incoherence import RHTFactors

__all__ = [
    "RHTLinear",
    "RHTLinearConfig",
]


@dataclass(frozen=True)
class RHTLinearConfig(FullPrecisionLinearConfig):
    rht_block_size: int

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "RHTLinear": ...

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "RHTLinear": ...

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "RHTLinear": ...

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "RHTLinear":
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "RHTLinear":
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class RHTLinear(FullPrecisionLinear):
    rht_factors: RHTFactors

    def __post_init__(self) -> None:
        super().__post_init__()
        (if_input_dim,) = self.rht_factors.input_factors.shape
        (if_output_dim,) = self.rht_factors.output_factors.shape
        if if_input_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: {if_input_dim} != {self.input_dim}")
        if if_output_dim != sum(self.output_dims):
            raise ValueError(f"Output dimension mismatch: {if_output_dim} != {sum(self.output_dims)}")

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        inputs = inputs * self.rht_factors.input_factors
        result = super().__call__(inputs)

        output_factors = jnp.split(self.rht_factors.output_factors, self._get_split_points(self.output_dims))

        return tuple(r * f for r, f in zip(result, output_factors, strict=True))

    def export_weights(self) -> ParameterTree:
        result = super().export_weights()
        assert isinstance(result, MutableMapping)
        result["input_factors"] = self.rht_factors.output_factors
        result["output_factors"] = self.rht_factors.output_factors
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["input_factors"], Array)
        assert isinstance(weights["output_factors"], Array)
        result = super().import_weights(weights)
        result.rht_factors = RHTFactors(
            input_factors=weights["input_factors"],
            output_factors=weights["output_factors"],
        )
        return result
