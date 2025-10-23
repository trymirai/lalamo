from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree

from .common import ForwardPassMode, LalamoModule
from .decoder_layer import TransformerLayerResult
from .embedding import EmbeddingBase, EmbeddingConfig
from .kv_cache import KVCache
from .rope import PositionalEmbeddings
from .transformer import Transformer, TransformerConfig, TransformerForwardPassConfig
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
    kv_cache: KVCache | None

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]

    def export(self) -> ParameterTree:
        result = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            local_positional_embeddings=self.local_positional_embeddings.export(),
            global_positional_embeddings=self.global_positional_embeddings.export(),
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
        )
        if self.kv_cache is not None:
            result["kv_cache"] = [kv_cache_layer_slice.export() for kv_cache_layer_slice in self.kv_cache]
        return result


class DecoderResult(eqx.Module):
    logits: Float[Array, "batch suffix_tokens channels"]
    updated_kv_cache: KVCache | None = None
    activation_trace: DecoderActivationTrace | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            logits=self.logits,
        )
        if self.updated_kv_cache is not None:
            result["updated_kv_cache"] = [
                kv_cache_layer_slice.export() for kv_cache_layer_slice in self.updated_kv_cache
            ]
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

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        token_positions: Int[Array, "batch suffix_tokens"],
        kv_cache: KVCache | None = None,
        return_updated_kv_cache: bool = False,
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

        transformer_result = self.transformer(
            inner_features=inner_features,
            token_positions=token_positions,
            kv_cache=kv_cache,
            return_updated_kv_cache=return_updated_kv_cache,
            return_layer_results=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )

        # Project to vocabulary
        logits = vmap_twice(self.embedding.readout)(transformer_result.outputs)

        # Build activation trace if requested
        if return_activation_trace:
            assert transformer_result.layer_results is not None
            assert transformer_result.global_positional_embeddings is not None
            assert transformer_result.local_positional_embeddings is not None

            activation_trace = DecoderActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                kv_cache=kv_cache,
                global_positional_embeddings=transformer_result.global_positional_embeddings,
                local_positional_embeddings=transformer_result.local_positional_embeddings,
                layer_results=transformer_result.layer_results,
                output_norm=transformer_result.outputs,
            )
        else:
            activation_trace = None

        return DecoderResult(
            logits=logits,
            updated_kv_cache=transformer_result.updated_kv_cache,
            activation_trace=activation_trace,
        )

    def init_static_kv_cache(self, batch_size: int, capacity: int) -> KVCache:
        return self.transformer.init_static_kv_cache(batch_size, capacity)

    def export_weights(self) -> ParameterTree:
        return dict(
            embedding=self.embedding.export_weights(),
            transformer=self.transformer.export_weights(),
        )

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["embedding"], Mapping)
        assert isinstance(weights["transformer"], Mapping)

        return replace(
            self,
            embedding=self.embedding.import_weights(weights["embedding"]),
            transformer=self.transformer.import_weights(weights["transformer"]),
        )
