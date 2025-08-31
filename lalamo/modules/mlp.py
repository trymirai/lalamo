from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree

from .activations import Activation
from .common import LalamoModule, WeightLayout
from .linear import LinearBase, LinearConfig

__all__ = ["MLP", "MLPConfig"]


@dataclass(frozen=True)
class MLPConfig:
    linear_config: LinearConfig
    activation: Activation
    has_up_biases: bool
    has_down_biases: bool
    gate_clipping: tuple[float | None, float | None] | None
    up_clipping: tuple[float | None, float | None] | None

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "MLP":
        up_key, down_key = jax.random.split(key)
        return MLP(
            self,
            up_projection=self.linear_config.random_init(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty(self, model_dim: int, hidden_dim: int) -> "MLP":
        return MLP(
            self,
            up_projection=self.linear_config.empty(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "MLP":
        up_key, down_key = jax.random.split(key)
        return MLP(
            self,
            up_projection=self.linear_config.random_init_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
    ) -> "MLP":
        return MLP(
            self,
            up_projection=self.linear_config.empty_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )


class MLP(LalamoModule[MLPConfig]):
    up_projection: LinearBase
    down_projection: LinearBase

    @property
    def activation_precision(self) -> DTypeLike:
        return self.up_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.up_projection.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.down_projection.input_dim

    def __post_init__(self) -> None:
        up_output_dim, gate_output_dim = self.up_projection.output_dims
        if up_output_dim != gate_output_dim:
            raise ValueError(
                f"Up projection output dimension {up_output_dim} does not match"
                f" the gate output dimension {gate_output_dim}",
            )
        (down_output_dim,) = self.down_projection.output_dims
        if self.up_projection.input_dim != down_output_dim:
            raise ValueError(
                f"Down projection input dimension {down_output_dim} does not match"
                f" the up projection output dimension {self.up_projection.input_dim}",
            )

    @eqx.filter_jit
    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(inputs)
        if self.config.gate_clipping:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        return {
            "up_projection": self.up_projection.export_weights(weight_layout),
            "down_projection": self.down_projection.export_weights(weight_layout),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["up_projection"], Mapping)
        assert isinstance(weights["down_projection"], Mapping)
        return replace(
            self,
            up_projection=self.up_projection.import_weights(weights["up_projection"], weight_layout),
            down_projection=self.down_projection.import_weights(weights["down_projection"], weight_layout),
        )
