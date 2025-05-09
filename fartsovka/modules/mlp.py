from dataclasses import dataclass

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import ParameterDict

from .activations import Activation
from .common import FartsovkaModule, ModuleSample
from .linear import LinearBase, LinearConfig

__all__ = ["MLP", "MLPConfig"]


@dataclass
class MLPConfig:
    linear_config: LinearConfig
    activation: Activation

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "MLP":
        up_key, down_key = jax.random.split(key)
        return MLP(
            self,
            up_projection=self.linear_config.random_init(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=False,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init(
                hidden_dim,
                (model_dim,),
                has_biases=False,
                key=down_key,
            ),
        )


class MLP(FartsovkaModule):
    up_projection: LinearBase
    down_projection: LinearBase

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

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(x)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            up_projection=self.up_projection.export_weights(),
            down_projection=self.down_projection.export_weights(),
        )

    def export_samples(self, suffix_length: int, key: PRNGKeyArray) -> ParameterDict:
        up_projection_key, down_projection_key, activation_key, mlp_key = jax.random.split(key, num=4)

        x = jax.random.uniform(
            mlp_key,
            (suffix_length, self.model_dim),
            minval=-2,
            maxval=2,
            dtype=self.config.linear_config.precision,
        )
        y = self(x.T).T
        sample = ModuleSample(inputs=(x,), outputs=(y,))

        return ParameterDict(
            up_projection=self.up_projection.export_samples(suffix_length, up_projection_key),
            down_projection=self.down_projection.export_samples(suffix_length, down_projection_key),
            activation=self.config.activation.export_samples([suffix_length, self.hidden_dim], self.config.linear_config.precision, activation_key),
            value=sample.export(),
        )