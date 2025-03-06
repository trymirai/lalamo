from dataclasses import dataclass
from typing import NamedTuple, Any

import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import ParameterDict
from fartsovka.modules.medusa import MedusaConfig

from .common import FartsovkaModule
from .decoder_layer import DecoderLayer, DecoderLayerConfig
from .embedding import EmbeddingBase, EmbeddingConfig
from .kv_cache import KVCacheLayerSlice
from .normalization import RMSNorm, RMSNormConfig
from .rope import RoPE, RoPEConfig

__all__ = [
    "Decoder",
    "DecoderConfig",
    "DecoderOutput",
]


class DecoderOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: list[KVCacheLayerSlice] | None = None
    medusa_output: Float[Array, "num_heads"] | None = None


@dataclass
class DecoderConfig:
    embedding_config: EmbeddingConfig
    rope_config: RoPEConfig
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
    medusa_config: "MedusaConfig | None" = None

    def __post_init__(self) -> None:
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
        
        if self.medusa_config is not None:
            keys = jax.random.split(key, 3)
            embedding_key, layers_key, medusa_key = keys
        else:
            keys = jax.random.split(key, 2)
            embedding_key, layers_key = keys
            medusa_key = None
            
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        
        rope = self.rope_config.init(
            head_dim=self.head_dim,
            num_timesteps=self.context_length,
        )

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
        
        medusa = None
        if self.medusa_config is not None and medusa_key is not None:
            from fartsovka.modules.medusa import Medusa
            medusa = self.medusa_config.random_init(
                hidden_size=self.model_dim,
                key=medusa_key,
            )
            
        return Decoder(
            self,
            embedding=embedding,
            rope=rope,
            layers=layers,
            output_norm=output_norm,
            medusa=medusa,
        )


class Decoder(FartsovkaModule[DecoderConfig]):
    embedding: EmbeddingBase
    rope: RoPE
    layers: tuple[DecoderLayer, ...]
    output_norm: RMSNorm
    medusa: Any | None = None
    
    def __init__(
        self,
        config: DecoderConfig,
        embedding: EmbeddingBase,
        rope: RoPE,
        layers: tuple[DecoderLayer, ...],
        output_norm: RMSNorm,
        medusa: Any | None = None,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.rope = rope
        self.layers = layers
        self.output_norm = output_norm
        self.medusa = medusa

    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: list[KVCacheLayerSlice] | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderOutput:
        maybe_kv_cache = kv_cache or ([None] * len(self.layers))
        x = self.embedding.embed(token_ids)
        positional_embeddings = self.rope(token_positions)
        updated_kv_cache = []
        for layer, kv_cache_slice in zip(self.layers, maybe_kv_cache, strict=True):
            decoder_layer_output = layer(
                x,
                positional_embeddings,
                kv_cache_slice,
                mask,
                return_updated_kv_cache,
            )
            x = decoder_layer_output.output
            updated_kv_cache.append(decoder_layer_output.kv_cache)
        x = vmap(self.output_norm, in_axes=0)(x)
        result = vmap(self.embedding.readout, in_axes=0)(x)
        
        medusa_output = None
        if self.medusa is not None:
            medusa_states = self.medusa(x)
            medusa_output = jax.vmap(
                lambda x: jax.vmap(self.embedding.readout, in_axes=0)(x),
                in_axes=0
            )(medusa_states)
        return DecoderOutput(
            output=result, 
            kv_cache=updated_kv_cache or None,
            medusa_output=medusa_output
        )

    def export_weights(self) -> ParameterDict:
        weights = ParameterDict(
            embedding=self.embedding.export_weights(),
            rope=self.rope.export_weights(),
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )
        
        if self.medusa is not None:
            weights["medusa"] = self.medusa.export_weights()
            
        return weights
