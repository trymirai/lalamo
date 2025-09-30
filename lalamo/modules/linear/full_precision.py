import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array

from .common import LinearBase, LinearConfigBase

__all__ = [
    "FullPrecisionLinear",
    "FullPrecisionLinearConfig",
]


@dataclass(frozen=True)
class FullPrecisionLinearConfig(LinearConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "FullPrecisionLinear":
        scale = 1 / math.sqrt(input_dim)
        weights = jax.random.uniform(
            key,
            (input_dim, sum(output_dims)),
            minval=-scale,
            maxval=scale,
            dtype=self.precision,
        )
        if has_biases:
            biases = jnp.zeros((sum(output_dims),), dtype=self.precision)
        else:
            biases = None

        return FullPrecisionLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            biases=biases,
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        key: PRNGKeyArray,
    ) -> "FullPrecisionLinear":
        subkeys = jax.random.split(key, mixture_size)
        return eqx.filter_vmap(lambda key: self.random_init(input_dim, output_dims, has_biases, key=key))(subkeys)

    def _empty_general(
        self,
        leading_dims: tuple[int, ...],
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        weights = dummy_array(
            (*leading_dims, input_dim, sum(output_dims)),
            dtype=self.precision,
        )
        if has_biases:
            biases = dummy_array((*leading_dims, sum(output_dims)), dtype=self.precision)
        else:
            biases = None

        return FullPrecisionLinear(
            config=self,
            output_dims=output_dims,
            weights=weights,
            biases=biases,
        )

    def empty(
        self,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._empty_general((), input_dim, output_dims, has_biases)

    def empty_mixture(
        self,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
    ) -> "FullPrecisionLinear":
        return self._empty_general((mixture_size,), input_dim, output_dims, has_biases)


class FullPrecisionLinear(LinearBase[FullPrecisionLinearConfig]):
    weights: Float[Array, "*components in_channels total_out_channels"]
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_dim(self) -> int:
        *_, input_dim, _ = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    def __post_init__(self) -> None:
        if self.weights.dtype != self.config.precision:
            raise ValueError(
                f"Weight dtype ({self.weights.dtype}) is not equal to specified precision ({self.config.precision}).",
            )
        *w_num_components, _, w_output_dim = self.weights.shape
        if w_output_dim != sum(self.output_dims):
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to sum of output dims ({sum(self.output_dims)}).",
            )
        if self.biases is None:
            return
        *b_num_components, b_output_dim = self.biases.shape
        if w_output_dim != b_output_dim:
            raise ValueError(
                f"Number of output channels in weights ({w_output_dim}) is not"
                f" equal to number of output channels in biases ({b_output_dim}).",
            )
        if self.biases.dtype != self.config.precision:
            raise ValueError(
                f"Bias dtype ({self.biases.dtype}) is not equal to specified precision ({self.config.precision}).",
            )
        if b_num_components != w_num_components:
            raise ValueError(
                f"Number of mixture components in weights ({w_num_components}) is not"
                f" equal to number of mixture components in biases ({b_num_components}).",
            )

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " in_channels"]) -> tuple[Float[Array, " out_channels"], ...]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        result = inputs @ self.weights
        if self.biases is not None:
            result = result + self.biases
        return tuple(jnp.split(result, self._get_split_points(self.output_dims)))

    def export_weights(self) -> ParameterTree:
        result = dict(weights=self.weights)
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=weights["weights"],
            biases=weights["biases"] if self.has_biases else None,
        )
