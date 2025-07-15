from dataclasses import dataclass

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterDict

from .common import AttentionType, LalamoModule, WeightLayout
from .decoder_layer import DecoderLayer, DecoderLayerConfig, DecoderLayerResult
from .embedding import EmbeddingBase, EmbeddingConfig
from .kv_cache import KVCache
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderResult",
]


class DecoderActivationTrace(eqx.Module):
    token_ids: Int[Array, " suffix_tokens"]
    token_positions: Int[Array, " suffix_tokens"]
    kv_cache: KVCache | None

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    layer_results: tuple[DecoderLayerResult, ...]

    output_norm: Float[Array, "suffix_tokens channels"]

    def export(self) -> ParameterDict:
        result = ParameterDict(
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
    logits: Float[Array, "suffix_tokens channels"]
    updated_kv_cache: KVCache | None = None
    activation_trace: DecoderActivationTrace | None = None

    def export(self) -> ParameterDict:
        result = ParameterDict(
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
    global_rope_config: RoPEConfig
    local_rope_config: RoPEConfig | None
    layer_config: DecoderLayerConfig
    output_norm_config: RMSNormConfig

    vocab_size: int
    model_dim: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    head_dim: int
    attention_scale: float | None
    num_layers: int
    sliding_window_sizes: tuple[int | None, ...] | None
    context_length: int

    def __post_init__(self) -> None:
        if self.local_rope_config is not None and self.sliding_window_sizes is None:
            raise ValueError("Sliding window sizes must be provided when using local RoPE")
        if self.sliding_window_sizes is None:
            return
        if len(self.sliding_window_sizes) != self.num_layers:
            raise ValueError(
                f"Number of sliding window sizes {len(self.sliding_window_sizes)} does not match"
                f" the number of layers {self.num_layers}",
            )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "Decoder":
        embedding_key, layers_key = jax.random.split(key)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        global_rope = self.global_rope_config.init(
            head_dim=self.head_dim,
            num_timesteps=self.context_length,
        )

        if self.local_rope_config:
            assert self.sliding_window_sizes is not None
            max_sliding_window_size = max(
                window_size for window_size in self.sliding_window_sizes if window_size is not None
            )
            local_rope = self.local_rope_config.init(
                head_dim=self.head_dim,
                num_timesteps=max(max_sliding_window_size, self.context_length),
            )
        else:
            local_rope = None

        if self.sliding_window_sizes is None:
            sliding_window_sizes = [None] * self.num_layers
        else:
            sliding_window_sizes = self.sliding_window_sizes
        layers_keys = jax.random.split(layers_key, self.num_layers)
        layers = tuple(
            self.layer_config.random_init(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                head_dim=self.head_dim,
                attention_scale=self.attention_scale,
                sliding_window_size=sliding_window_size,
                key=key,
            )
            for sliding_window_size, key in zip(sliding_window_sizes, layers_keys, strict=True)
        )
        output_norm = self.output_norm_config.init(self.model_dim)
        return Decoder(
            self,
            embedding=embedding,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    global_rope: RoPE
    local_rope: RoPE | None
    layers: tuple[DecoderLayer, ...]
    output_norm: RMSNorm

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: KVCache | None = None,
        return_updated_kv_cache: bool = False,
        return_activation_trace: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
    ) -> DecoderResult:
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
            activation_trace = DecoderActivationTrace(
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

        return DecoderResult(
            logits=logits,
            updated_kv_cache=updated_kv_cache,
            activation_trace=activation_trace,
        )

    def init_static_kv_cache(self, capacity: int) -> KVCache:
        return KVCache(layer.init_static_kv_cache(capacity) for layer in self.layers)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        result = ParameterDict(
            embedding=self.embedding.export_weights(weight_layout),
            global_rope=self.global_rope.export_weights(weight_layout),
            layers=[layer.export_weights(weight_layout) for layer in self.layers],
            output_norm=self.output_norm.export_weights(weight_layout),
        )
        if self.local_rope:
            result["local_rope"] = self.local_rope.export_weights(weight_layout)
        return result
