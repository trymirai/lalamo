import math
from dataclasses import dataclass, replace
from functools import cached_property

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.weight_matrix import EmbeddingMatrix

from .activations import SiLU
from .flow import Context, StateLayerBase, TokenMixerBase, TokenMixerConfig, TokenMixerResult
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped_twice

__all__ = ["NGramConfig", "NGramInjection", "NGramState"]


class NGramState(StateLayerBase):
    tokens: Int[Array, "*batch context"]
    convolution: Float[Array, "*batch history wide_channels"]


def _is_prime(value: int) -> bool:
    return value >= 2 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


@dataclass(frozen=True)
class NGramConfig(TokenMixerConfig):
    """Qwen3.8 hashed N-gram embeddings gated by, and injected into, every residual stream."""

    vocab_size: int
    seed: int
    eos_token_id: int
    ngram_size: int
    heads_per_ngram: int
    vocab_size_base: int
    vocab_size_divisor: int
    embedding_dim: int
    kernel_size: int
    norm_config: NormalizationConfig
    layer_index: int
    linear_config: LinearConfig
    streams: int

    @cached_property
    def head_vocab_sizes(self) -> tuple[int, ...]:
        count = (self.ngram_size - 1) * self.heads_per_ngram
        prime = self.vocab_size_base - 1
        sizes = []
        for head in range((self.layer_index + 1) * count):
            prime += 1
            while not _is_prime(prime):
                prime += 1
            if head >= self.layer_index * count:
                sizes.append(prime)
        return tuple(sizes)

    @cached_property
    def multipliers(self) -> tuple[int, ...]:
        # Qwen's N-gram hashes use SplitMix64 with one layer-specific seed.
        mask = (1 << 64) - 1
        gamma = 0x9E3779B97F4A7C15
        bound = max(1, ((1 << 63) - 1) // self.vocab_size // 2)
        multipliers = []
        for index in range(self.ngram_size):
            value = (self.seed + 10007 * self.layer_index + gamma * (index + 2)) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            value ^= value >> 31
            multipliers.append(2 * (value % bound) + 1)
        return tuple(multipliers)

    def init(self, initializer: Initializer, model_dim: int) -> "NGramInjection":
        heads = len(self.head_vocab_sizes)
        if self.embedding_dim % heads:
            raise ValueError("N-gram embedding dimension must be divisible by its hash head count.")
        rows = math.ceil(sum(self.head_vocab_sizes) / self.vocab_size_divisor) * self.vocab_size_divisor
        width = model_dim * self.streams
        norm_config = replace(self.norm_config, group_size=model_dim)
        return NGramInjection(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=initializer.embedding_matrix(rows, self.embedding_dim // heads, is_sharded=True),
            key=self.linear_config.init(initializer, self.embedding_dim, (width,), has_biases=False),
            value=self.linear_config.init(initializer, self.embedding_dim, (model_dim,), has_biases=False),
            key_norm=norm_config.init(initializer, width),
            query_norm=norm_config.init(initializer, width),
            conv_norm=norm_config.init(initializer, width),
            conv_weights=initializer.normal(1 / math.sqrt(width * self.kernel_size), (width, self.kernel_size)),
        )


class NGramInjection(TokenMixerBase[NGramConfig, NGramState]):
    embedding: EmbeddingMatrix
    key: Linear
    value: Linear
    key_norm: Normalization
    query_norm: Normalization
    conv_norm: Normalization
    conv_weights: Float[Array, "wide_channels kernel"]

    def hash_ids(self, tokens: Int[Array, "batch tokens"]) -> Int[Array, "batch tokens heads"]:
        shifted = []
        for shift in range(self.config.ngram_size):
            previous = jnp.pad(tokens, ((0, 0), (shift, 0)), constant_values=self.config.eos_token_id)[
                :, : tokens.shape[1]
            ]
            reset = jnp.zeros_like(tokens, dtype=jnp.bool_)
            for offset in range(1, shift + 1):
                reset |= jnp.pad(tokens == self.config.eos_token_id, ((0, 0), (offset, 0)), constant_values=True)[
                    :, : tokens.shape[1]
                ]
            shifted.append(jnp.where(reset, self.config.eos_token_id, previous))
        # Hash arithmetic is signed 64-bit upstream; return int32 indices before leaving this boundary.
        with jax.enable_x64():
            mixed = shifted[0].astype(jnp.int64) * self.config.multipliers[0]
            blocks = []
            offset = 0
            for order in range(2, self.config.ngram_size + 1):
                mixed ^= shifted[order - 1].astype(jnp.int64) * self.config.multipliers[order - 1]
                for head in range(self.config.heads_per_ngram):
                    size = self.config.head_vocab_sizes[(order - 2) * self.config.heads_per_ngram + head]
                    blocks.append((mixed % size + offset).astype(jnp.int32))
                    offset += size
            return jnp.stack(blocks, axis=-1)

    @property
    def model_dim(self) -> int:
        return self.value.output_dims[0]

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> NGramState:  # noqa: ARG002
        history = (self.config.kernel_size - 1) * self.config.ngram_size
        return NGramState(
            jnp.full((self.config.ngram_size - 1,), self.config.eos_token_id, dtype=jnp.int32),
            jnp.zeros((history, self.key.output_dims[0]), dtype=dtype),
        )

    def forward(
        self,
        x: Float[Array, "batch tokens wide_channels"],
        *,
        state: NGramState | None,
        ctx: Context,
        keychain: Keychain,
    ) -> TokenMixerResult[NGramState]:
        if ctx.parents is not None or ctx.token_ids is None:
            raise ValueError("N-gram injection requires raw token IDs and a causal prefix.")
        streams, token_ids, lengths_without_padding = x, ctx.token_ids, ctx.lengths
        forward_pass_config = ctx.forward_pass_config.matmul_config
        batch, length, width = streams.shape
        if state is None:
            state = jax.tree.map(
                lambda leaf: jnp.repeat(leaf[None], batch, axis=0), self.init_static_state(0, streams.dtype)
            )
        token_history = jnp.concatenate((state.tokens, token_ids), axis=1)
        embed_key, key_key, value_key = keychain.split(3)
        ids = self.hash_ids(token_history)[:, -length:]
        embeddings = self.embedding.lookup_embedding(
            ids, dtype=streams.dtype, keychain=embed_key, forward_pass_config=forward_pass_config
        )
        embeddings = embeddings.reshape(batch, length, self.config.embedding_dim)
        axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        (keys,) = call_vmapped_twice(
            self.key, embeddings, keychain=key_key, forward_pass_config=forward_pass_config, added_sharding_axes=axes
        )
        (values,) = call_vmapped_twice(
            self.value,
            embeddings,
            keychain=value_key,
            forward_pass_config=forward_pass_config,
            added_sharding_axes=axes,
        )
        channels = values.shape[-1]
        normalized_keys = call_vmapped_twice(self.key_norm, keys).reshape(batch, length, -1, channels)
        queries = call_vmapped_twice(self.query_norm, streams).reshape(batch, length, -1, channels)
        gate = jnp.sum(normalized_keys * queries, axis=-1, keepdims=True) / math.sqrt(channels)
        gate = jnp.sign(gate) * jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6))
        gated = (jax.nn.sigmoid(gate) * values[..., None, :]).reshape(batch, length, width)
        normalized = call_vmapped_twice(self.conv_norm, gated)
        lengths = (
            jnp.full((batch,), length, dtype=jnp.int32) if lengths_without_padding is None else lengths_without_padding
        )
        valid = jnp.arange(length)[None, :, None] < lengths[:, None, None]
        gated = jnp.where(valid, gated, 0)
        normalized = jnp.where(valid, normalized, 0)
        context = jnp.concatenate((state.convolution, normalized), axis=1)
        windows = jnp.arange(length)[:, None] + jnp.arange(self.config.kernel_size)[None, :] * self.config.ngram_size
        convolution = jnp.einsum("btkc,ck->btc", context[:, windows], self.conv_weights.astype(streams.dtype))
        history = state.convolution.shape[1]
        updated_convolution = jax.vmap(lambda row, count: jax.lax.dynamic_slice(row, (count, 0), (history, width)))(
            context, lengths
        )
        token_context = self.config.ngram_size - 1
        updated_tokens = jax.vmap(lambda row, count: jax.lax.dynamic_slice(row, (count,), (token_context,)))(
            token_history, lengths
        )
        return TokenMixerResult(gated + SiLU()(convolution), NGramState(updated_tokens, updated_convolution))
