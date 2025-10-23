from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.utils import vmap_twice

from .common import AttentionType, ForwardPassMode, LalamoModule
from .kv_cache import KVCache
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .transformer_layer import (
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerForwardPassConfig,
    TransformerLayerResult,
)

__all__ = [
    "Transformer",
    "TransformerConfig",
    "TransformerResult",
]


type TransformerForwardPassConfig = TransformerLayerForwardPassConfig


class TransformerResult(eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_kv_cache: KVCache | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    global_positional_embeddings: PositionalEmbeddings | None = None
    local_positional_embeddings: PositionalEmbeddings | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            outputs=self.outputs,
        )
        if self.updated_kv_cache is not None:
            result["updated_kv_cache"] = [
                kv_cache_layer_slice.export() for kv_cache_layer_slice in self.updated_kv_cache
            ]
        if self.layer_results is not None:
            result["layer_results"] = [layer_result.export() for layer_result in self.layer_results]
        if self.global_positional_embeddings is not None:
            result["global_positional_embeddings"] = self.global_positional_embeddings.export()
        if self.local_positional_embeddings is not None:
            result["local_positional_embeddings"] = self.local_positional_embeddings.export()
        return result


@dataclass(frozen=True)
class TransformerConfig:
    global_rope_config: RoPEConfig
    local_rope_config: RoPEConfig | None
    layer_config: TransformerLayerConfig
    output_norm_config: NormalizationConfig

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
        is_causal: bool,
        *,
        key: PRNGKeyArray,
    ) -> "Transformer":
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

        layers_keys = jax.random.split(key, self.num_layers)
        layers = tuple(
            self.layer_config.random_init(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                head_dim=self.head_dim,
                attention_scale=self.attention_scale,
                sliding_window_size=sliding_window_size,
                key=layer_key,
                is_causal=is_causal,
            )
            for sliding_window_size, layer_key in zip(sliding_window_sizes, layers_keys, strict=True)
        )
        output_norm = self.output_norm_config.init(self.model_dim)

        return Transformer(
            config=self,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )

    def empty(self, is_causal:bool) -> "Transformer":
        global_rope = self.global_rope_config.init(
            head_dim=self.head_dim,
            num_timesteps=self.context_length,
        )

        if self.local_rope_config:
            local_rope = self.local_rope_config.init(
                head_dim=self.head_dim,
                num_timesteps=self.context_length,
            )
        else:
            local_rope = None

        if self.sliding_window_sizes is None:
            sliding_window_sizes = [None] * self.num_layers
        else:
            sliding_window_sizes = self.sliding_window_sizes

        layers = tuple(
            self.layer_config.empty(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                head_dim=self.head_dim,
                attention_scale=self.attention_scale,
                sliding_window_size=sliding_window_size,
                is_causal=is_causal,
            )
            for sliding_window_size in sliding_window_sizes
        )
        output_norm = self.output_norm_config.empty(self.model_dim)

        return Transformer(
            config=self,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    global_rope: RoPE
    local_rope: RoPE | None
    layers: tuple[TransformerLayer, ...]
    output_norm: Normalization

    @property
    def activation_precision(self) -> DTypeLike:
        return self.layers[0].activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        inner_features: Float[Array, "batch suffix_tokens channels"],
        token_positions: Int[Array, "batch suffix_tokens"],
        kv_cache: KVCache | None,
        return_updated_kv_cache: bool,
        return_layer_results: bool,
        return_positional_embeddings: bool,
        lengths_without_padding: Int[Array, " batch"] | None,
        forward_pass_mode: ForwardPassMode,
        forward_pass_config: TransformerForwardPassConfig | None,
    ) -> TransformerResult:
        if inner_features.ndim != 3:
            raise ValueError(
                f"inner_features must be a 3D array of size (batch_size, sequence_length, hidden_dim), got {inner_features.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )

        maybe_kv_cache = kv_cache or ([None] * len(self.layers))

        global_positional_embeddings = vmap(self.global_rope)(token_positions)
        if self.local_rope is not None:
            local_positional_embeddings = vmap(self.local_rope)(token_positions)
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
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_mode=forward_pass_mode,
                forward_pass_config=forward_pass_config,
            )
            inner_features = layer_result.outputs
            layer_results.append(layer_result)
            updated_kv_cache_layers.append(layer_result.updated_kv_cache)

        normalized_outputs = vmap_twice(self.output_norm)(inner_features)

        return TransformerResult(
            outputs=normalized_outputs,
            updated_kv_cache=KVCache(updated_kv_cache_layers) if return_updated_kv_cache else None,
            layer_results=tuple(layer_results) if return_layer_results else None,
            global_positional_embeddings=global_positional_embeddings if return_positional_embeddings else None,
            local_positional_embeddings=local_positional_embeddings if return_positional_embeddings else None,
        )

    def init_static_kv_cache(self, batch_size: int, capacity: int) -> KVCache:
        return KVCache(layer.init_static_kv_cache(batch_size, capacity) for layer in self.layers)

    def export_weights(self) -> ParameterTree:
        result = dict(
            global_rope=self.global_rope.export_weights(),
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )
        if self.local_rope:
            result["local_rope"] = self.local_rope.export_weights()
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["global_rope"], Mapping)
        assert isinstance(weights["layers"], Sequence)
        assert isinstance(weights["output_norm"], Mapping)

        if self.local_rope:
            assert isinstance(weights["local_rope"], Mapping)
            local_rope = self.local_rope.import_weights(weights["local_rope"])
        else:
            local_rope = None

        layers = []
        for layer, layer_weights in zip(self.layers, weights["layers"], strict=True):
            assert isinstance(layer_weights, Mapping)
            layers.append(layer.import_weights(layer_weights))

        return replace(
            self,
            global_rope=self.global_rope.import_weights(weights["global_rope"]),
            layers=tuple(layers),
            output_norm=self.output_norm.import_weights(weights["output_norm"]),
            local_rope=local_rope,
        )
