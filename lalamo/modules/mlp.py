from dataclasses import dataclass

import jax
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterDict

from .activations import Activation
from .common import LalamoModule, WeightLayout
from .linear import LinearBase, LinearConfig

__all__ = ["MLP", "MLPConfig"]


@dataclass(frozen=True)
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


class MLP(LalamoModule):
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

    def __call__(self, inputs: Float[Array, " channels"]) -> Float[Array, " channels"]:
        up_proj, gate = self.up_projection(inputs)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(up_proj * gate)
        return result

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        return ParameterDict(
            up_projection=self.up_projection.export_weights(weight_layout),
            down_projection=self.down_projection.export_weights(weight_layout),
        )
