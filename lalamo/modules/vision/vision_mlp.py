from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.activations import Activation
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.utils import call_vmapped
from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "VisionMLP",
    "VisionMLPConfig",
]


@dataclass(frozen=True)
class VisionMLPConfig(LalamoConfig):
    """Plain two-layer perceptron, ``down(act(up(x)))``, without a gate.

    The decoder's ``DenseMLP`` always has a gated (SwiGLU-style) up projection, so it cannot
    express this. The same module serves both the MLP inside each vision block and the
    projection head of the patch merger; only the widths and the activation differ.
    """

    linear_config: LinearConfig
    activation: Activation
    has_biases: bool

    def init(self, initializer: Initializer, input_dim: int, hidden_dim: int, output_dim: int) -> "VisionMLP":
        return VisionMLP(
            config=self,
            sharding_config=initializer.sharding_config,
            up_projection=self.linear_config.init(initializer, input_dim, (hidden_dim,), has_biases=self.has_biases),
            down_projection=self.linear_config.init(
                initializer,
                hidden_dim,
                (output_dim,),
                has_biases=self.has_biases,
            ),
        )


class VisionMLP(LalamoModule[VisionMLPConfig]):
    up_projection: Linear
    down_projection: Linear

    @property
    def input_dim(self) -> int:
        return self.up_projection.input_dim

    @property
    def output_dim(self) -> int:
        (result,) = self.down_projection.output_dims
        return result

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "tokens in_channels"],
        forward_pass_config: MatmulConfig = MatmulConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens out_channels"]:
        up_keychain, down_keychain = keychain.split()
        (hidden,) = call_vmapped(
            self.up_projection,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=up_keychain,
        )
        (result,) = call_vmapped(
            self.down_projection,
            self.config.activation(hidden),
            forward_pass_config=forward_pass_config,
            keychain=down_keychain,
        )
        return result
