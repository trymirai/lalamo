from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_mapping, require_tree

from .common import (
    ForwardPassMode,
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

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
        )
        if self.state is not None:
            result["state"] = [state_layer.export() for state_layer in self.state]
        if self.local_positional_embeddings is not None:
            result["local_positional_embeddings"] = self.local_positional_embeddings.export()
        if self.global_positional_embeddings is not None:
            result["global_positional_embeddings"] = self.global_positional_embeddings.export()
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


@dataclass(frozen=True)
class DecoderConfig:
    embedding_config: EmbeddingConfig
    transformer_config: TransformerConfig

    vocab_size: int
    pard_token: int | None = None
    ple_model_config: PLEModelConfig | None = None

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "Decoder":
        embedding_key, transformer_key = jax.random.split(key)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.transformer_config.model_dim,
            key=embedding_key,
        )
        transformer = self.transformer_config.random_init(key=transformer_key)

        return Decoder(
            config=self,
            embedding=embedding,
            transformer=transformer,
        )

    def empty(self) -> "Decoder":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.transformer_config.model_dim,
        )
        transformer = self.transformer_config.empty()
        ple_token_embedding = None
        ple_model_projection = None
        ple_projection_norm = None
        if self.ple_model_config is not None:
            cfg = self.ple_model_config
            total_ple_dim = cfg.num_layers * cfg.ple_dim
            ple_token_embedding = dummy_array(
                (cfg.ple_vocab_size, total_ple_dim),
                jnp.bfloat16,
            )
            ple_model_projection = cfg.linear_config.empty(
                self.transformer_config.model_dim,
                (total_ple_dim,),
                has_biases=False,
            )
            ple_projection_norm = cfg.norm_config.empty(cfg.ple_dim)

        return Decoder(
            config=self,
            embedding=embedding,
            transformer=transformer,
            ple_token_embedding=ple_token_embedding,
            ple_model_projection=ple_model_projection,
            ple_projection_norm=ple_projection_norm,
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    transformer: Transformer
    ple_token_embedding: Float[Array, "vocab ple_total_dim"] | None = None
    ple_model_projection: LinearBase | None = None
    ple_projection_norm: Normalization | None = None

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    def _compute_per_layer_inputs(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        inner_features: Float[Array, "batch suffix_tokens channels"],
    ) -> tuple[Float[Array, "batch suffix_tokens ple_dim"], ...] | None:
        if self.ple_token_embedding is None:
            return None
        assert self.ple_model_projection is not None
        assert self.ple_projection_norm is not None
        assert self.config.ple_model_config is not None
        cfg = self.config.ple_model_config

        # Token PLE: lookup and scale
        token_ple = self.ple_token_embedding[token_ids] * cfg.ple_embed_scale
        token_ple = rearrange(
            token_ple,
            "batch seq (layers ple_dim) -> batch seq layers ple_dim",
            layers=cfg.num_layers,
            ple_dim=cfg.ple_dim,
        )

        # Model projection PLE: vmap over batch and seq dims since linear expects 1D input
        (model_ple,) = vmap(vmap(self.ple_model_projection))(inner_features)
        model_ple = model_ple * cfg.model_projection_scale
        model_ple = rearrange(
            model_ple,
            "batch seq (layers ple_dim) -> batch seq layers ple_dim",
            layers=cfg.num_layers,
            ple_dim=cfg.ple_dim,
        )
        model_ple = vmap(vmap(vmap(self.ple_projection_norm)))(model_ple)

        # Combine
        combined = (model_ple + token_ple) * cfg.input_scale
        return tuple(combined[:, :, i, :] for i in range(cfg.num_layers))

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

        per_layer_inputs = self._compute_per_layer_inputs(token_ids, inner_features)

        transformer_result = self.transformer(
            inner_features=inner_features,
            token_positions=token_positions,
            state=state,
            return_updated_state=return_updated_state,
            return_layer_results=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
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

    def export_weights(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            embedding=self.embedding.export_weights(),
            transformer=self.transformer.export_weights(),
        )
        if self.ple_token_embedding is not None:
            result["ple_token_embedding"] = self.ple_token_embedding
        if self.ple_model_projection is not None:
            result["ple_model_projection"] = self.ple_model_projection.export_weights()
        if self.ple_projection_norm is not None:
            result["ple_projection_norm"] = self.ple_projection_norm.export_weights()
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        ple_token_embedding = None
        if self.ple_token_embedding is not None:
            ple_token_embedding = weights.get("ple_token_embedding")
        ple_model_projection = None
        if self.ple_model_projection is not None:
            ple_model_projection = self.ple_model_projection.import_weights(
                require_tree(weights["ple_model_projection"])
            )
        ple_projection_norm = None
        if self.ple_projection_norm is not None:
            ple_projection_norm = self.ple_projection_norm.import_weights(require_tree(weights["ple_projection_norm"]))
        return replace(
            self,
            embedding=self.embedding.import_weights(require_tree(weights["embedding"])),
            transformer=self.transformer.import_weights(require_tree(weights["transformer"])),
            ple_token_embedding=ple_token_embedding,
            ple_model_projection=ple_model_projection,
            ple_projection_norm=ple_projection_norm,
        )
