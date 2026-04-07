from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.common import ParameterTree, require_mapping, require_tree
from lalamo.modules.common import Initializer, PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.utils import vmap_with_key

from .common import MixerForwardPassConfig, TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .state import ShortConvStateLayer

__all__ = [
    "ShortConv",
    "ShortConvConfig",
    "ShortConvResult",
]


ShortConvResult = TokenMixerResult[ShortConvStateLayer]


@dataclass(frozen=True)
class ShortConvConfig(TokenMixerConfigBase):
    in_projection_config: LinearConfig
    conv_config: SeparableCausalConvConfig
    out_projection_config: LinearConfig

    kernel_size: int

    @property
    def rope_dim(self) -> None:
        return None

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "ShortConv":
        in_projection = self.in_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(model_dim,) * 3,
            has_biases=False,
        )
        conv = self.conv_config.init(initializer, model_dim, self.kernel_size)
        out_projection = self.out_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        return ShortConv(
            config=self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
        )


class ShortConv(TokenMixerBase[ShortConvConfig, ShortConvStateLayer]):
    in_projection: LinearBase
    conv: SeparableCausalConv
    out_projection: Linear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_projection.input_dim

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: ShortConvStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),  # noqa: B008
        *,
        key: Key[Array, ""] | None,
    ) -> TokenMixerResult[ShortConvStateLayer]:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for ShortConv.")

        in_key, out_key = jax.random.split(key) if key is not None else (None, None)
        pre_conv_gate, post_conv_gate, x = vmap_with_key(
            partial(self.in_projection, forward_pass_config=forward_pass_config.arrays),
            inputs,
            key=in_key,
        )

        prev_conv_state = state.conv_state if state is not None else None
        conv_output = self.conv(x * pre_conv_gate, length_without_padding, prev_conv_state, return_updated_state)

        (outputs,) = vmap_with_key(
            partial(self.out_projection, forward_pass_config=forward_pass_config.arrays),
            conv_output.outputs * post_conv_gate,
            key=out_key,
        )
        updated_conv_state = conv_output.state

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = ShortConvStateLayer(updated_conv_state)
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int) -> ShortConvStateLayer:  # noqa: ARG002
        return ShortConvStateLayer.init(
            self.config.kernel_size,
            self.in_projection.input_dim,
            self.in_projection.activation_precision,
        )
