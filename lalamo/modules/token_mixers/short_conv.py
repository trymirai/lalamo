from dataclasses import dataclass
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from lalamo.modules.utils import call_vmapped

__all__ = [
    "ShortConv",
    "ShortConvConfig",
    "ShortConvResult",
    "ShortConvStateLayer",
]


class ShortConvStateLayer(StateLayerBase):
    conv_state: Float[Array, "*batch tokens conv_channels"]

    def __post_init__(self) -> None:
        if self.conv_state.ndim not in (2, 3):
            raise ValueError(
                f"Conv state must have 2 or 3 dimensions: [batch], tokens, conv_channels,"
                f" got shape {self.conv_state.shape}",
            )

    @classmethod
    def init(
        cls,
        kernel_size: int,
        model_dim: int,
        dtype: DTypeLike,
    ) -> Self:
        return cls(conv_state=jnp.zeros((kernel_size - 1, model_dim), dtype=dtype))


ShortConvResult = TokenMixerResult[ShortConvStateLayer]


@dataclass(frozen=True)
class ShortConvConfig(TokenMixerConfig):
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
    in_projection: Linear
    conv: SeparableCausalConv
    out_projection: Linear

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
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[ShortConvStateLayer]:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for ShortConv.")
        if attention_parent_indices is not None:
            raise ValueError("Attention parent indices are not supported for ShortConv.")

        in_keychain, out_keychain = keychain.split()
        pre_conv_gate, post_conv_gate, x = call_vmapped(
            self.in_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=in_keychain,
        )

        prev_conv_state = state.conv_state if state is not None else None
        conv_output = self.conv(x * pre_conv_gate, length_without_padding, prev_conv_state, return_updated_state)

        (outputs,) = call_vmapped(
            self.out_projection,
            conv_output.outputs * post_conv_gate,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
        )
        updated_conv_state = conv_output.state

        if return_updated_state:
            assert updated_conv_state is not None
            updated_state = ShortConvStateLayer(updated_conv_state)
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> ShortConvStateLayer:  # noqa: ARG002
        return ShortConvStateLayer.init(self.config.kernel_size, self.in_projection.input_dim, dtype)
