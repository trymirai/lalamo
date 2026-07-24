import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.sharding import lookup_sharded_indices
from lalamo.weight_matrix import GradientEstimator

from .embedding import EmbeddingBase, EmbeddingConfig, EmbeddingForwardPassConfig
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixer import State
from .transformer import (
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
    TransformerLayerResult,
)
from .utils import call_vmapped_twice

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
    embedding_forward_pass_config: EmbeddingForwardPassConfig = dataclass_field(
        default_factory=EmbeddingForwardPassConfig,
    )
    transformer_forward_pass_config: TransformerForwardPassConfig = dataclass_field(
        default_factory=TransformerForwardPassConfig,
    )

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_tracer_tests(),
            transformer_forward_pass_config=TransformerForwardPassConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_inference(precision),
            transformer_forward_pass_config=TransformerForwardPassConfig.for_inference(mode, precision),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_training(gradient_estimator, precision),
            transformer_forward_pass_config=TransformerForwardPassConfig.for_training(gradient_estimator, precision),
        )


class DecoderActivationTrace(Exportable, eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None

    rope_embeddings: tuple[PositionalEmbeddings, ...] | None

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]


class DecoderResult(Exportable, eqx.Module):
    logits: Float[Array, "batch suffix_tokens vocabulary"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None


@dataclass(frozen=True)
class PLEModelConfig(LalamoConfig):
    ple_dim: int
    num_layers: int
    ple_vocab_size: int
    ple_embed_scale: float
    model_projection_scale: float
    input_scale: float
    linear_config: LinearConfig
    norm_config: NormalizationConfig


class PerLayerEmbedding(LalamoModule[PLEModelConfig]):
    token_embedding: Float[Array, "vocab ple_total_dim"]
    model_projection: Linear
    projection_norm: Normalization

    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        inner_features: Float[Array, "batch suffix_tokens channels"],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens ple_dim"], ...]:
        config = self.config
        token_ple = lookup_sharded_indices(self.token_embedding, token_ids) * config.ple_embed_scale
        token_ple = rearrange(
            token_ple,
            "batch tokens (layers ple_dim) -> batch tokens layers ple_dim",
            layers=config.num_layers,
            ple_dim=config.ple_dim,
        )
        (model_ple,) = call_vmapped_twice(
            self.model_projection,
            inner_features,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        model_ple = model_ple * config.model_projection_scale
        model_ple = rearrange(
            model_ple,
            "batch tokens (layers ple_dim) -> batch tokens layers ple_dim",
            layers=config.num_layers,
            ple_dim=config.ple_dim,
        )
        model_ple = jax.vmap(jax.vmap(jax.vmap(self.projection_norm)))(model_ple)
        combined = (model_ple + token_ple) * config.input_scale
        return tuple(combined[:, :, layer_index, :] for layer_index in range(config.num_layers))


@dataclass(frozen=True)
class DecoderConfig(LalamoConfig):
    embedding_config: EmbeddingConfig
    transformer_config: TransformerConfig

    vocab_size: int
    ple_model_config: PLEModelConfig | None = None

    def init(self, initializer: Initializer) -> "Decoder":
        embedding = self.embedding_config.init(
            initializer,
            model_dim=self.transformer_config.model_dim,
            vocab_size=self.vocab_size,
        )
        transformer = self.transformer_config.init(initializer)
        if self.ple_model_config is not None:
            config = self.ple_model_config
            total_ple_dim = config.num_layers * config.ple_dim
            per_layer_embedding = PerLayerEmbedding(
                config=config,
                sharding_config=initializer.sharding_config,
                token_embedding=initializer.normal(
                    1 / math.sqrt(config.ple_dim),
                    (config.ple_vocab_size, total_ple_dim),
                ),
                model_projection=config.linear_config.init(
                    initializer,
                    input_dim=self.transformer_config.model_dim,
                    output_dims=(total_ple_dim,),
                    has_biases=False,
                ),
                projection_norm=config.norm_config.init(initializer, config.ple_dim),
            )
        else:
            per_layer_embedding = None

        return Decoder(
            config=self,
            sharding_config=initializer.sharding_config,
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
        forward_pass_config: DecoderForwardPassConfig = DecoderForwardPassConfig(),
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
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
        if return_suffix_tokens is not None:
            _, sequence_length = token_ids.shape
            if not 1 <= return_suffix_tokens <= sequence_length:
                raise ValueError(
                    f"return_suffix_tokens must be between 1 and the sequence length {sequence_length},"
                    f" got {return_suffix_tokens}",
                )
            if return_activation_trace:
                raise ValueError("return_suffix_tokens cannot be combined with return_activation_trace.")
        embedding_keychain, ple_keychain, transformer_keychain, readout_keychain = keychain.split(4)
        inner_features = self.embedding.embed(
            token_ids,
            forward_pass_config=forward_pass_config.embedding_forward_pass_config,
            keychain=embedding_keychain,
        )

        if self.per_layer_embedding is not None:
            per_layer_inputs = self.per_layer_embedding(token_ids, inner_features, keychain=ple_keychain)
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
            forward_pass_config=forward_pass_config.transformer_forward_pass_config,
            per_layer_inputs=per_layer_inputs,
            attention_parent_indices=attention_parent_indices,
            return_suffix_tokens=return_suffix_tokens,
            keychain=transformer_keychain,
        )

        logits = call_vmapped_twice(
            self.embedding.readout,
            transformer_result.outputs,
            forward_pass_config=forward_pass_config.embedding_forward_pass_config,
            keychain=readout_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
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
