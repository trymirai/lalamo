from dataclasses import dataclass

from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import ParameterDict

from .activations import Activation
from .common import FartsovkaModule
from .linear import LinearBase, LinearConfig

__all__ = ["ResBlock", "ResBlockConfig"]


@dataclass
class ResBlockConfig:
    linear_config: LinearConfig
    activation: Activation = Activation.SILU
    
    def random_init(self, hidden_size: int, *, key: PRNGKeyArray) -> "ResBlock":
        return ResBlock(
            self,
            linear=self.linear_config.random_init(
                hidden_size,
                (hidden_size,),
                has_biases=True,
                key=key,
            ),
        )


class ResBlock(FartsovkaModule):

    linear: LinearBase

    @property
    def hidden_size(self) -> int:
        return self.linear.input_dim
    
    def __post_init__(self) -> None:
        (output_dim,) = self.linear.output_dims
        if self.linear.input_dim != output_dim:
            raise ValueError(
                f"Linear projection input dimension {self.linear.input_dim} does not match "
                f"the output dimension {output_dim}"
            )

    def __call__(self, x: Float[Array, " channels"]) -> Float[Array, " channels"]:

        (linear_output,) = self.linear(x)
        activated = self.config.activation(linear_output)
        return x + activated

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            linear=self.linear.export_weights(),
        )