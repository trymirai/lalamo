from dataclasses import dataclass
from itertools import accumulate

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, field
from lalamo.weight_matrix import FullPrecisionSpec, Layout, MatmulConfig, ShapeDtypeSpec, WeightMatrix

__all__ = [
    "Linear",
    "LinearConfig",
]


@dataclass(frozen=True)
class LinearConfig(LalamoConfig):
    def init(
        self,
        initializer: Initializer,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        is_sharded: bool = True,
    ) -> "Linear":
        total_output_dim = sum(output_dims)
        return Linear(
            config=self,
            weights=initializer.weight_matrix(total_output_dim, input_dim, is_sharded=is_sharded),
            biases=initializer.zeros((total_output_dim,)) if has_biases else None,
            output_dims=output_dims,
        )

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        input_dim: int,
        output_dims: tuple[int, ...],
        has_biases: bool,
        *,
        is_sharded: bool = True,
    ) -> "Linear":
        total_output_dim = sum(output_dims)
        if has_biases:
            biases = initializer.zeros((mixture_size, total_output_dim))
        else:
            biases = None
        return Linear(
            config=self,
            weights=initializer.weight_matrix(total_output_dim, input_dim, mixture_size, is_sharded=is_sharded),
            biases=biases,
            output_dims=output_dims,
        )


class Linear(LalamoModule[LinearConfig]):
    output_dims: tuple[int, ...] = field(static=True)
    weights: WeightMatrix
    biases: Float[Array, "*components total_out_channels"] | None

    @property
    def input_dim(self) -> int:
        if isinstance(self.weights.spec, FullPrecisionSpec | ShapeDtypeSpec):  # noqa: SIM102
            if self.weights.spec.layout == Layout.INPUT_OUTPUT:
                *_, input_dim, _output_dim = self.weights.shape
                return input_dim

        *_, _output_dim, input_dim = self.weights.shape
        return input_dim

    @property
    def has_biases(self) -> bool:
        return self.biases is not None

    @property
    def num_outputs(self) -> int:
        return len(self.output_dims)

    @property
    def mixture_size(self) -> int | None:
        match self.weights.shape:
            case [num_components, _, _]:
                return num_components
            case _:
                return None

    @staticmethod
    def get_split_points(output_dims: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(accumulate(output_dims[:-1]))

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> tuple[Float[Array, " out_channels"], ...]:
        result = self.weights.dot(inputs, keychain=keychain, forward_pass_config=forward_pass_config)
        if self.biases is not None:
            result = result + self.biases.astype(result.dtype)
        return tuple(jnp.split(result, self.get_split_points(self.output_dims)))
