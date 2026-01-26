from dataclasses import dataclass

import equinox as eqx
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int

from .common import ForwardPassMode, Initializer, LalamoConfig, LalamoModule
from .embedding import EmbeddingBase, EmbeddingConfigBase
from .rope import PositionalEmbeddings
from .token_mixers import State
from .transformer import (
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
)
from .transformer_layer import TransformerLayerResult
from .utils import vmap_twice

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderForwardPassConfig",
    "DecoderResult",
]


type DecoderForwardPassConfig = TransformerForwardPassConfig


class DecoderActivationTrace(eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None

    local_positional_embeddings: PositionalEmbeddings | None
    global_positional_embeddings: PositionalEmbeddings | None

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]


class DecoderResult(eqx.Module):
    logits: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None


@dataclass(frozen=True)
class DecoderConfig(LalamoConfig):
    embedding_config: EmbeddingConfigBase
    transformer_config: TransformerConfig

    vocab_size: int

    def init(self, initializer: Initializer) -> "Decoder":
        embedding = self.embedding_config.init(
            initializer,
            vocab_size=self.vocab_size,
            model_dim=self.transformer_config.model_dim,
        )
        transformer = self.transformer_config.init(initializer)

        return Decoder(
            embedding=embedding,
            transformer=transformer,
        )


class Decoder(LalamoModule):
    embedding: EmbeddingBase
    transformer: Transformer

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        num_suffix_tokens_to_return: int | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: DecoderForwardPassConfig | None = None,
    ) -> DecoderResult:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be a 2D array of size (batch_size, sequence_length), got {token_ids.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )

        inner_features = vmap(self.embedding.embed)(token_ids)

        transformer_result = self.transformer(
            inner_features=inner_features,
            token_positions=token_positions,
            state=state,
            return_updated_state=return_updated_state,
            return_activation_trace=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            num_suffix_tokens_to_return=num_suffix_tokens_to_return,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )

        logits = vmap_twice(self.embedding.readout)(transformer_result.outputs)

        if return_activation_trace:
            assert transformer_result.layer_results is not None

            activation_trace = DecoderActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                state=state,
                global_positional_embeddings=transformer_result.global_positional_embeddings,
                local_positional_embeddings=transformer_result.local_positional_embeddings,
                layer_results=transformer_result.layer_results,
                output_norm=transformer_result.outputs,
            )
        else:
            activation_trace = None

        return DecoderResult(
            logits=logits,
            updated_state=transformer_result.updated_state,
            activation_trace=activation_trace,
        )

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return self.transformer.init_static_state(batch_size, capacity)
