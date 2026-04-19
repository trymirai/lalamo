from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, LalamoConfig, LalamoModule

from .embedding import EmbeddingBase, EmbeddingConfig
from .linear import LinearBase, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixer import State
from .transformer import (
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
    TransformerLayerResult,
)
from .utils import vmap_twice_with_dequant_key

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderForwardPassConfig",
    "DecoderResult",
    "PLEModelConfig",
    "PerLayerEmbedding",
]


@dataclass(frozen=True)
class DecoderForwardPassConfig:
    transformer: TransformerForwardPassConfig = dataclass_field(default_factory=TransformerForwardPassConfig)


class DecoderActivationTrace(Exportable, eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None

    rope_embeddings: tuple[PositionalEmbeddings, ...] | None

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]


class DecoderResult(Exportable, eqx.Module):
    logits: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None


@dataclass(frozen=True)
class DecoderConfig(LalamoConfig):
    embedding_config: EmbeddingConfig
    transformer_config: TransformerConfig

    vocab_size: int
    pard_token: int | None = None
    ple_model_config: PLEModelConfig | None = None

    def init(self, initializer: Initializer) -> "Decoder":
        embedding = self.embedding_config.init(
            initializer,
            vocab_size=self.vocab_size,
            model_dim=self.transformer_config.model_dim,
        )
        transformer = self.transformer_config.init(initializer)

        return Decoder(
            config=self,
            embedding=embedding,
            transformer=transformer,
            per_layer_embedding=per_layer_embedding,
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    transformer: Transformer
    per_layer_embedding: PerLayerEmbedding | None

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        dequant_key: Key[Array, ""],
    ) -> DecoderResult:
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig()
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be a 2D array of size (batch_size, sequence_length), got {token_ids.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )
        embedding_dequant_key, transformer_dequant_key, readout_dequant_key = jax.random.split(dequant_key, 3)
        inner_features = vmap_twice_with_dequant_key(
            self.embedding.embed,
            token_ids,
            dequant_key=embedding_dequant_key,
        )

        if self.per_layer_embedding is not None:
            per_layer_inputs = self.per_layer_embedding(token_ids, inner_features)
        else:
            per_layer_inputs = None

        transformer_result = self.transformer(
            inner_features=inner_features,
            token_positions=token_positions,
            state=state,
            return_updated_state=return_updated_state,
            return_layer_results=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config.transformer,
            dequant_key=transformer_dequant_key,
        )

        logits = vmap_twice_with_dequant_key(
            self.embedding.readout,
            transformer_result.outputs,
            dequant_key=readout_dequant_key,
        )

        if return_activation_trace:
            assert transformer_result.layer_results is not None

            activation_trace = DecoderActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                state=state,
                rope_embeddings=transformer_result.rope_embeddings,
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

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return self.transformer.init_static_state(batch_size, capacity, dtype)
