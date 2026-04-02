from dataclasses import dataclass

import equinox as eqx
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterTree

from .common import (
    ForwardPassMode,
    Initializer,
    LalamoModule,
)
from .embedding import EmbeddingBase, EmbeddingConfig
from .linear import LinearBase, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixers import State
from .transformer import (
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
    TransformerLayerResult,
)
from .utils import vmap_twice

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderForwardPassConfig",
    "DecoderResult",
    "PLEModelConfig",
    "PerLayerEmbedding",
]


type DecoderForwardPassConfig = TransformerForwardPassConfig


class DecoderActivationTrace(eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None

    rope_embeddings: tuple[PositionalEmbeddings, ...] | None

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
        )
        if self.state is not None:
            result["state"] = [state_layer.export() for state_layer in self.state]
        if self.rope_embeddings is not None:
            result["rope_embeddings"] = [emb.export() for emb in self.rope_embeddings]
        return result


class DecoderResult(eqx.Module):
    logits: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            logits=self.logits,
        )
        if self.updated_state is not None:
            result["updated_state"] = [state_layer.export() for state_layer in self.updated_state]
        if self.activation_trace is not None:
            result["activation_trace"] = self.activation_trace.export()
        return result


@dataclass(frozen=True)
class PLEModelConfig:
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
    model_projection: LinearBase
    projection_norm: Normalization

    @property
    def activation_precision(self) -> DTypeLike:
        return self.model_projection.activation_precision

    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        inner_features: Float[Array, "batch suffix_tokens channels"],
    ) -> tuple[Float[Array, "batch suffix_tokens ple_dim"], ...]:
        cfg = self.config
        token_ple = self.token_embedding[token_ids] * cfg.ple_embed_scale
        token_ple = rearrange(
            token_ple,
            "batch seq (layers ple_dim) -> batch seq layers ple_dim",
            layers=cfg.num_layers,
            ple_dim=cfg.ple_dim,
        )
        (model_ple,) = vmap(vmap(self.model_projection))(inner_features)
        model_ple = model_ple * cfg.model_projection_scale
        model_ple = rearrange(
            model_ple,
            "batch seq (layers ple_dim) -> batch seq layers ple_dim",
            layers=cfg.num_layers,
            ple_dim=cfg.ple_dim,
        )
        model_ple = vmap(vmap(vmap(self.projection_norm)))(model_ple)
        combined = (model_ple + token_ple) * cfg.input_scale
        return tuple(combined[:, :, i, :] for i in range(cfg.num_layers))

    def export_weights(self) -> ParameterTree:
        return {
            "token_embedding": self.token_embedding,
            "model_projection": self.model_projection.export_weights(),
            "projection_norm": self.projection_norm.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            token_embedding=require_array(weights["token_embedding"]),
            model_projection=self.model_projection.import_weights(require_tree(weights["model_projection"])),
            projection_norm=self.projection_norm.import_weights(require_tree(weights["projection_norm"])),
        )


@dataclass(frozen=True)
class DecoderConfig:
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
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: DecoderForwardPassConfig | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
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
            attention_parent_indices=attention_parent_indices,
            forward_pass_config=forward_pass_config,
            per_layer_inputs=per_layer_inputs,
        )

        logits = vmap_twice(self.embedding.readout)(transformer_result.outputs)

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

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return self.transformer.init_static_state(batch_size, capacity)
