from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.common import PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .mamba import SeparableCausalConv, SeparableCausalConvConfig
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

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "ShortConv":
        in_projection = self.in_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim,)*3,
            has_biases=False,
            key=key,
        )

        conv = self.conv_config.random_init(model_dim, self.kernel_size, key=key)

        out_projection = self.out_projection_config.random_init(
            input_dim=model_dim,
            output_dims=(model_dim,),
            has_biases=False,
            key=key,
        )

        return ShortConv(
            self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "ShortConv":
        in_projection = self.in_projection_config.empty(
            input_dim=model_dim,
            output_dims=(model_dim,)*3,
            has_biases=False,
        )

        conv = self.conv_config.empty(model_dim, self.kernel_size)

        out_projection = self.out_projection_config.empty(
            input_dim=model_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )

        return ShortConv(
            self,
            in_projection=in_projection,
            conv=conv,
            out_projection=out_projection,
        )


class ShortConv(TokenMixerBase[ShortConvConfig, ShortConvStateLayer]):
    in_projection: LinearBase
    conv: SeparableCausalConv
    out_projection: LinearBase

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_projection.input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: ShortConvStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> TokenMixerResult[ShortConvStateLayer]:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for ShortConv.")

        pre_conv_gate, post_conv_gate, x = vmap(self.in_projection)(inputs)

        prev_conv_state = state.conv_state if state is not None else None
        conv_output = self.conv(x * pre_conv_gate, length_without_padding, prev_conv_state, return_updated_state)

        (outputs,) = vmap(self.out_projection)(conv_output.outputs * post_conv_gate)
        updated_conv_state = conv_output.state

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = ShortConvStateLayer(updated_conv_state)
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int) -> ShortConvStateLayer: # noqa: ARG002
        return ShortConvStateLayer.init(
            self.config.kernel_size,
            self.in_projection.input_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        return {
            "in_projection": self.in_projection.export_weights(),
            "conv": self.conv.export_weights(),
            "out_projection": self.out_projection.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            in_projection=self.in_projection.import_weights(require_tree(weights["in_projection"])),
            conv=self.conv.import_weights(require_tree(weights["conv"])),
            out_projection=self.out_projection.import_weights(require_tree(weights["out_projection"])),
        )
