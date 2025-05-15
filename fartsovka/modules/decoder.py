from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import ParameterDict
from fartsovka.samples import ModuleSample, DecoderSamplesContext

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
        embedding_key, layers_key = jax.random.split(key)
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
        return Decoder(
            self,
            embedding=embedding,
            rope=rope,
            layers=layers,
            output_norm=output_norm,
        )


class Decoder(FartsovkaModule[DecoderConfig]):
    embedding: EmbeddingBase
    rope: RoPE
    layers: tuple[DecoderLayer, ...]
    output_norm: RMSNorm

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
        return DecoderOutput(output=result, kv_cache=updated_kv_cache or None)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            embedding=self.embedding.export_weights(),
            rope=self.rope.export_weights(),
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )

    def export_samples(self, context: DecoderSamplesContext, key: PRNGKeyArray) -> ParameterDict:
        embedding_key, rope_key, layers_all_key, decoder_key, output_norm_key = jax.random.split(key, num=5)
        layers_keys = jax.random.split(layers_all_key, len(self.layers))

        embedding_x = context.token_ids
        embedding_y = self.embedding.embed(embedding_x)
        embedding_sample = ModuleSample(inputs=(embedding_x,), outputs=(embedding_y,))

        rope_x = context.token_positions
        rope_y = self.rope(rope_x)
        rope_sample = ModuleSample(inputs=(rope_x,), outputs=(rope_y.sines, rope_y.cosines))

        layers_samples: [ParameterDict] = []
        for index, layer in enumerate(self.layers):
            key = layers_keys[index]
            parameters = layer.export_samples(context, key)
            layers_samples.append(parameters)

        context_parameters = ParameterDict(
            token_ids=context.token_ids,
            token_positions=context.token_positions,
            mask=context.mask,
        )

        x = context.token_ids
        y = self(x, context.token_positions, None, context.mask, False)
        sample = ModuleSample(inputs=(x.astype(jnp.int64),), outputs=(y.output,))

        return ParameterDict(
            embedding=embedding_sample.export(),
            rope=rope_sample.export(),
            layers=layers_samples,
            output_norm=self.output_norm.export_samples(context, output_norm_key),
            context=context_parameters,
            value=sample.export(),
        )
