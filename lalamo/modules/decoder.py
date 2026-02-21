from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree

from .common import ForwardPassMode, LalamoModule, ShardingConfig, apply_data_sharding, get_default_sharding_config
from .embedding import EmbeddingBase, EmbeddingConfig
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
class DecoderConfig:
    embedding_config: EmbeddingConfig
    transformer_config: TransformerConfig

    vocab_size: int

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

        return Decoder(
            config=self,
            embedding=embedding,
            transformer=transformer,
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    transformer: Transformer
    sharding_config: ShardingConfig | None = eqx.field(static=True, default=None)

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @property
    def resolved_sharding_config(self) -> ShardingConfig | None:
        if self.sharding_config is not None:
            return self.sharding_config
        return get_default_sharding_config()

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
        sharding_config = self.resolved_sharding_config
        token_ids = apply_data_sharding(token_ids, sharding_config)
        token_positions = apply_data_sharding(token_positions, sharding_config)
        state = apply_data_sharding(state, sharding_config)
        lengths_without_padding = apply_data_sharding(lengths_without_padding, sharding_config)

        inner_features = vmap(self.embedding.embed)(token_ids)
        inner_features = apply_data_sharding(inner_features, sharding_config)

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
        )

        logits = vmap_twice(self.embedding.readout)(transformer_result.outputs)
        logits = apply_data_sharding(logits, sharding_config)

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
        state = self.transformer.init_static_state(batch_size, capacity)
        return apply_data_sharding(state, self.resolved_sharding_config)

    def export_weights(self) -> ParameterTree:
        return dict(
            embedding=self.embedding.export_weights(),
            transformer=self.transformer.export_weights(),
        )

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            embedding=self.embedding.import_weights(require_tree(weights["embedding"])),
            transformer=self.transformer.import_weights(require_tree(weights["transformer"])),
        )
