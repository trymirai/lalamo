from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jax.random import PRNGKey
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.transformer import TransformerConfig, Transformer

from .common import AttentionType, LalamoModule
from .decoder_layer import DecoderLayer, DecoderLayerConfig, DecoderLayerResult
from .linear import FullPrecisionLinear, FullPrecisionLinearConfig
from .embedding import EmbeddingBase, EmbeddingConfig
from .kv_cache import KVCache
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .bert_heads import ModernBertPredictionHead, ModernBertPredictionHeadConfig

__all__ = [
    "Classifier",
    "ClassifierActivationTrace",
    "ClassifierConfig",
    "ClassifierResult",
]


class ClassifierActivationTrace(eqx.Module):
    token_ids: Int[Array, " suffix_tokens"]
    token_positions: Int[Array, " suffix_tokens"]
    kv_cache: KVCache | None

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    layer_results: tuple[DecoderLayerResult, ...]

    output_norm: Float[Array, "suffix_tokens channels"]

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


class ClassifierResult(eqx.Module):
    logits: Float[Array, "suffix_tokens channels"]
    updated_kv_cache: KVCache | None = None
    activation_trace: ClassifierActivationTrace | None = None

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
class ClassifierConfig:
    embedding_config: EmbeddingConfig
    transformer_config: TransformerConfig
    prediction_head_config: ModernBertPredictionHeadConfig
    classifier_config: FullPrecisionLinearConfig

    vocab_size: int
    model_dim: int
    hidden_dim: int
    num_heads: int
    # num_groups: int  NOTE: this one seem to be not used in ModertBert attention
    head_dim: int
    attention_scale: float | None
    num_layers: int
    sliding_window_sizes: tuple[int | None, ...] | None
    context_length: int
    num_labels: int

    def __post_init__(self) -> None:
        self.transformer_config.__post_init__()

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "Classifier":
        embedding_key, transformer_key = jax.random.split(key)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        # global_rope = self.global_rope_config.init(
        #     head_dim=self.head_dim,
        #     num_timesteps=self.context_length,
        # )

        # if self.local_rope_config:
        #     assert self.sliding_window_sizes is not None
        #     max_sliding_window_size = max(
        #         window_size for window_size in self.sliding_window_sizes if window_size is not None
        #     )
        #     local_rope = self.local_rope_config.init(
        #         head_dim=self.head_dim,
        #         num_timesteps=max(max_sliding_window_size, self.context_length),
        #     )
        # else:
        #     local_rope = None

        # if self.local_rope_config:
        #     assert self.sliding_window_sizes is not None
        #     max_sliding_window_size = max(
        #         window_size for window_size in self.sliding_window_sizes if window_size is not None
        #     )
        #     local_rope = self.local_rope_config.init(
        #         head_dim=self.head_dim,
        #         num_timesteps=max(max_sliding_window_size, self.context_length),
        #     )
        # else:
        #     local_rope = None

        # if self.sliding_window_sizes is None:
        #     sliding_window_sizes = [None] * self.num_layers
        # else:
        #     sliding_window_sizes = self.sliding_window_sizes
        # layers_keys = jax.random.split(layers_key, self.num_layers)
        # layers = tuple(
        #     self.layer_config.random_init(
        #         model_dim=self.model_dim,
        #         hidden_dim=self.hidden_dim,
        #         num_heads=self.num_heads,
        #         num_groups=1,
        #         head_dim=self.head_dim,
        #         attention_scale=self.attention_scale,
        #         sliding_window_size=sliding_window_size,
        #         key=key,
        #     )
        #     for sliding_window_size, key in zip(sliding_window_sizes, layers_keys, strict=True)
        # )
        # output_norm = self.output_norm_config.init(self.model_dim)
        transformer = self.transformer_config.random_init(
            key=transformer_key
        )
        classifier = self.classifier_config.random_init(
            self.hidden_dim,
            (self.num_labels,),
            has_biases=True,
            key=PRNGKey(123)
        )
        return Classifier(
            self,
            embedding=embedding,
            transformer=transformer,
            classifier=classifier
        )

    def empty(
        self,
    ) -> "Classifier":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )
        # global_rope = self.global_rope_config.init(
        #     head_dim=self.head_dim,
        #     num_timesteps=self.context_length,
        # )

        # if self.local_rope_config:
        #     assert self.sliding_window_sizes is not None
        #     max_sliding_window_size = max(
        #         window_size for window_size in self.sliding_window_sizes if window_size is not None
        #     )
        #     local_rope = self.local_rope_config.init(
        #         head_dim=self.head_dim,
        #         num_timesteps=max(max_sliding_window_size, self.context_length),
        #     )
        # else:
        #     local_rope = None

        # if self.sliding_window_sizes is None:
        #     sliding_window_sizes = [None] * self.num_layers
        # else:
        #     sliding_window_sizes = self.sliding_window_sizes
        # layers = tuple(
        #     self.layer_config.empty(
        #         model_dim=self.model_dim,
        #         hidden_dim=self.hidden_dim,
        #         num_heads=self.num_heads,
        #         num_groups=1,
        #         head_dim=self.head_dim,
        #         attention_scale=self.attention_scale,
        #         sliding_window_size=sliding_window_size,
        #     )
        #     for sliding_window_size in sliding_window_sizes
        # )
        # output_norm = self.output_norm_config.empty(self.model_dim)
        transformer= self.transformer_config.empty()
        classifier = self.classifier_config.empty(self.hidden_dim, (self.num_labels,), True)
        return Classifier(
            self,
            embedding=embedding,
            transformer=transformer,
            classifier=classifier,
        )


class Classifier(LalamoModule[ClassifierConfig]):
    embedding: EmbeddingBase
    transformer: Transformer
    classifier: FullPrecisionLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: KVCache | None = None,
        return_updated_kv_cache: bool = False,
        return_activation_trace: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> ClassifierResult:
        maybe_kv_cache = kv_cache or ([None] * len(self.layers))
        inner_features = self.embedding.embed(token_ids)

        global_positional_embeddings = self.global_rope(token_positions)
        if self.local_rope is not None:
            local_positional_embeddings = self.local_rope(token_positions)
        else:
            local_positional_embeddings = global_positional_embeddings

        updated_kv_cache_layers = []
        layer_results = []
        for layer, kv_cache_slice in zip(self.layers, maybe_kv_cache, strict=True):
            if layer.attention_type == AttentionType.SLIDING_WINDOW:
                positional_embeddings_to_use = local_positional_embeddings
            else:
                positional_embeddings_to_use = global_positional_embeddings

            layer_result = layer(
                inner_features,
                positional_embeddings_to_use,
                kv_cache=kv_cache_slice,
                return_updated_kv_cache=return_updated_kv_cache,
                return_activation_trace=return_activation_trace,
                length_without_padding=length_without_padding,
            )
            inner_features = layer_result.outputs
            layer_results.append(layer_result)
            updated_kv_cache_layers.append(layer_result.updated_kv_cache)

        normalized_outputs = vmap(self.output_norm, in_axes=0)(inner_features)
        logits = vmap(self.embedding.readout, in_axes=0)(normalized_outputs)

        if return_activation_trace:
            activation_trace = ClassifierActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                kv_cache=kv_cache,
                global_positional_embeddings=global_positional_embeddings,
                local_positional_embeddings=local_positional_embeddings,
                layer_results=tuple(layer_results),
                output_norm=normalized_outputs,
            )
        else:
            activation_trace = None

        if return_updated_kv_cache:
            updated_kv_cache = KVCache(updated_kv_cache_layers)
        else:
            updated_kv_cache = None

        return ClassifierResult(
            logits=logits,
            updated_kv_cache=updated_kv_cache,
            activation_trace=activation_trace,
        )

    def init_static_kv_cache(self, capacity: int) -> KVCache:
        return KVCache(layer.init_static_kv_cache(capacity) for layer in self.layers)

    def export_weights(self) -> ParameterTree:
        result = dict(
            embedding=self.embedding.export_weights(),
            transformer=self.transformer.export_weights(),
            classifier=self.classifier.export_weights()
        )
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["embedding"], Mapping)
        assert isinstance(weights["transformer"], Mapping)
        assert isinstance(weights["classifier"], Mapping)
        return replace(
            self,
            embedding=self.embedding.import_weights(weights["embedding"]),
            transformer=self.transformer.import_weights(weights["transformer"]),
            classifier=self.classifier.import_weights(weights["classifier"])
        )
