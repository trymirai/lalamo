import math
from dataclasses import dataclass
from itertools import accumulate

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, UInt

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LogicalAxis
from lalamo.weight_matrix import EmbeddingMatrix

from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .residual import GatedResidualConfig, ResidualConfig, normalize_streams
from .stream_ops import StreamOpBase, StreamOpConfig, StreamOpContext
from .token_mixer import StateLayerBase
from .token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "NGramHashConfig",
    "NGramInjection",
    "NGramInjectionConfig",
    "NGramInjectionState",
    "multiply_limbs",
    "remainder_of_limbs",
]


def multiply_limbs(
    values: UInt[Array, "*shape"],
    constant: int,
    limb_bits: int,
    num_limbs: int,
) -> UInt[Array, "*shape limbs"]:
    limb_mask = (1 << limb_bits) - 1
    limbs = []
    carry = jnp.zeros_like(values)
    for limb_index in range(num_limbs):
        partial_product = values * ((constant >> (limb_bits * limb_index)) & limb_mask) + carry
        limbs.append(partial_product & limb_mask)
        carry = partial_product >> limb_bits
    return jnp.stack(limbs, axis=-1)


def remainder_of_limbs(
    limbs: UInt[Array, "*shape limbs"],
    divisors: UInt[Array, " divisors"],
    limb_bits: int,
    chunk_bits: int,
) -> UInt[Array, "*shape divisors"]:
    *shape, num_limbs = limbs.shape
    remainders = jnp.zeros((*shape, *divisors.shape), dtype=jnp.uint32)
    for limb_index in reversed(range(num_limbs)):
        limb = limbs[..., limb_index, None]
        for low_bit in reversed(range(0, limb_bits, chunk_bits)):
            width = min(chunk_bits, limb_bits - low_bit)
            chunk = (limb >> low_bit) & ((1 << width) - 1)
            remainders = ((remainders << width) | chunk) % divisors
    return remainders


@dataclass(frozen=True)
class NGramHashConfig(LalamoConfig):
    token_vocab_size: int
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    multipliers: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.ngram_size < 2:
            raise ValueError(f"N-gram hashing requires ngram_size >= 2, got {self.ngram_size}.")
        if len(self.multipliers) != self.ngram_size:
            raise ValueError(f"Expected {self.ngram_size} multipliers, got {len(self.multipliers)}.")
        if len(self.head_vocab_sizes) != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} head vocabulary sizes, got {len(self.head_vocab_sizes)}.")
        if not 0 <= self.eos_token_id < self.token_vocab_size:
            raise ValueError(f"eos_token_id {self.eos_token_id} is outside the vocabulary.")
        if (self.token_vocab_size - 1).bit_length() >= 32:
            raise ValueError(f"Token vocabulary of size {self.token_vocab_size} does not fit uint32 limb arithmetic.")
        if any(multiplier <= 0 or self.max_product_bits(multiplier) > 63 for multiplier in self.multipliers):
            raise ValueError("Every token-multiplier product must be a non-negative int64.")
        if min(self.head_vocab_sizes) < 2 or max(self.head_vocab_sizes).bit_length() >= 32:
            raise ValueError("Head vocabulary sizes must be in [2, 2**31).")
        if self.num_rows >= 2**31:
            raise ValueError(f"N-gram table of {self.num_rows} rows does not fit int32 indices.")

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def num_rows(self) -> int:
        return sum(self.head_vocab_sizes)

    @property
    def head_offsets(self) -> tuple[int, ...]:
        *offsets, _ = accumulate(self.head_vocab_sizes, initial=0)
        return tuple(offsets)

    def max_product_bits(self, multiplier: int) -> int:
        return ((self.token_vocab_size - 1) * multiplier).bit_length()

    def ngram_ids(self, tokens: Int[Array, " tokens"]) -> Int[Array, "tokens heads"]:
        (num_tokens,) = tokens.shape
        limb_bits = 32 - (self.token_vocab_size - 1).bit_length()
        num_limbs = math.ceil(max(map(self.max_product_bits, self.multipliers)) / limb_bits)
        chunk_bits = 32 - max(self.head_vocab_sizes).bit_length()

        positions = jnp.arange(num_tokens, dtype=jnp.int32)
        eos_positions = jnp.where(tokens == self.eos_token_id, positions, -1)
        previous_eos_positions = jnp.concatenate(
            [jnp.full((1,), -1, dtype=jnp.int32), jax.lax.cummax(eos_positions)[:-1]],
        )
        positions_in_segment = positions - previous_eos_positions - 1

        product_limbs = []
        for shift, multiplier in enumerate(self.multipliers):
            source_positions = positions - shift
            shifted_tokens = jnp.where(
                (positions_in_segment >= shift) & (source_positions >= 0),
                tokens[jnp.maximum(source_positions, 0)],
                self.eos_token_id,
            )
            product_limbs.append(multiply_limbs(shifted_tokens.astype(jnp.uint32), multiplier, limb_bits, num_limbs))
        _, *mixed_limbs_per_order = accumulate(product_limbs, jnp.bitwise_xor)

        head_ids = []
        for order_index, mixed_limbs in enumerate(mixed_limbs_per_order):
            heads = slice(order_index * self.heads_per_ngram, (order_index + 1) * self.heads_per_ngram)
            remainders = remainder_of_limbs(
                mixed_limbs,
                jnp.asarray(self.head_vocab_sizes[heads], dtype=jnp.uint32),
                limb_bits,
                chunk_bits,
            )
            head_ids.append(remainders.astype(jnp.int32) + jnp.asarray(self.head_offsets[heads], dtype=jnp.int32))
        return jnp.concatenate(head_ids, axis=-1)


class NGramInjectionState(StateLayerBase):
    token_history: Int[Array, "*batch context_tokens"]
    conv_state: Float[Array, "*batch conv_span streams_channels"]


@dataclass(frozen=True)
class NGramInjectionConfig(StreamOpConfig):
    hash_config: NGramHashConfig
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    conv_config: SeparableCausalConvConfig
    embedding_dim: int
    conv_kernel_size: int

    def __post_init__(self) -> None:
        if self.embedding_dim % self.hash_config.num_heads != 0:
            raise ValueError(
                f"embedding_dim {self.embedding_dim} is not divisible by {self.hash_config.num_heads} n-gram heads.",
            )

    @property
    def is_token_local(self) -> bool:
        return False

    @property
    def is_stateful(self) -> bool:
        return True

    def init(
        self,
        initializer: Initializer,
        residual_config: ResidualConfig,
        model_dim: int,
    ) -> "NGramInjection":
        if not isinstance(residual_config, GatedResidualConfig):
            raise TypeError(f"N-gram injection requires a gated residual, got {type(residual_config).__name__}.")
        stream_count = residual_config.stream_count
        return NGramInjection(
            config=self,
            sharding_config=initializer.sharding_config,
            ngram_embedding=initializer.embedding_matrix(
                self.hash_config.num_rows,
                self.embedding_dim // self.hash_config.num_heads,
            ),
            key_projection=self.linear_config.init(
                initializer,
                input_dim=self.embedding_dim,
                output_dims=(stream_count * model_dim,),
                has_biases=False,
            ),
            value_projection=self.linear_config.init(
                initializer,
                input_dim=self.embedding_dim,
                output_dims=(model_dim,),
                has_biases=False,
            ),
            key_norm=self.norm_config.init_mixture(initializer, stream_count, model_dim),
            query_norm=self.norm_config.init_mixture(initializer, stream_count, model_dim),
            conv_norm=self.norm_config.init_mixture(initializer, stream_count, model_dim),
            conv=self.conv_config.init(initializer, stream_count * model_dim, self.conv_kernel_size),
        )


class NGramInjection(
    StreamOpBase[NGramInjectionConfig, Float[Array, "batch suffix_tokens streams channels"], NGramInjectionState],
):
    ngram_embedding: EmbeddingMatrix
    key_projection: Linear
    value_projection: Linear
    key_norm: Normalization
    query_norm: Normalization
    conv_norm: Normalization
    conv: SeparableCausalConv

    def __call__(
        self,
        stream: Float[Array, "batch suffix_tokens streams channels"],
        context: StreamOpContext,
        state: NGramInjectionState | None,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens streams channels"], NGramInjectionState]:
        if context.token_ids is None:
            raise ValueError("N-gram injection requires token_ids.")
        hash_config = self.config.hash_config
        _, num_tokens, stream_count, model_dim = stream.shape
        num_context_tokens = hash_config.ngram_size - 1
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        embedding_keychain, key_keychain, value_keychain = keychain.split(3)

        def hash_row(
            token_ids_row: Int[Array, " suffix_tokens"],
            token_history_row: Int[Array, " context_tokens"] | None,
            length_without_padding: Int[Array, ""] | None,
        ) -> tuple[Int[Array, "suffix_tokens heads"], Int[Array, " context_tokens"]]:
            if token_history_row is None:
                token_history_row = jnp.full((num_context_tokens,), hash_config.eos_token_id, dtype=jnp.int32)
            if length_without_padding is None:
                length_without_padding = jnp.asarray(num_tokens, dtype=jnp.int32)
            tokens_with_history = jnp.concatenate([token_history_row, token_ids_row.astype(jnp.int32)])
            updated_token_history_row = jax.lax.dynamic_slice_in_dim(
                tokens_with_history,
                length_without_padding,
                num_context_tokens,
            )
            return hash_config.ngram_ids(tokens_with_history)[num_context_tokens:], updated_token_history_row

        ngram_ids, updated_token_history = call_vmapped(
            hash_row,
            context.token_ids,
            state.token_history if state is not None else None,
            context.lengths_without_padding,
            added_sharding_axis=batch_axis,
        )
        embeddings = rearrange(
            self.ngram_embedding.lookup_embedding(
                ngram_ids,
                dtype=stream.dtype,
                keychain=embedding_keychain,
            ),
            "batch tokens heads head_channels -> batch tokens (heads head_channels)",
        )
        (flat_keys,) = call_vmapped_twice(
            self.key_projection,
            embeddings,
            keychain=key_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        (values,) = call_vmapped_twice(
            self.value_projection,
            embeddings,
            keychain=value_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        keys = normalize_streams(
            self.key_norm,
            rearrange(
                flat_keys,
                "batch tokens (streams channels) -> batch tokens streams channels",
                streams=stream_count,
            ),
        )
        queries = normalize_streams(self.query_norm, stream)
        gate_logits = jnp.sum(keys * queries, axis=-1) / math.sqrt(model_dim)
        gate_logits = jnp.sign(gate_logits) * jnp.sqrt(jnp.maximum(jnp.abs(gate_logits), 1e-6))
        gated_values = rearrange(
            jax.nn.sigmoid(gate_logits),
            "batch tokens streams -> batch tokens streams 1",
        ) * rearrange(values, "batch tokens channels -> batch tokens 1 channels")
        conv_outputs, updated_conv_state = call_vmapped(
            lambda conv_inputs, length_without_padding, conv_state_row: self.conv(
                conv_inputs,
                length_without_padding,
                conv_state_row,
                return_updated_state=True,
            ),
            rearrange(
                normalize_streams(self.conv_norm, gated_values),
                "batch tokens streams channels -> batch tokens (streams channels)",
            ),
            context.lengths_without_padding,
            state.conv_state if state is not None else None,
            added_sharding_axis=batch_axis,
        )
        assert updated_conv_state is not None
        injection = rearrange(gated_values, "batch tokens streams channels -> batch tokens (streams channels)")
        injection = injection + jax.nn.silu(conv_outputs)
        updated_stream = stream + rearrange(
            injection,
            "batch tokens (streams channels) -> batch tokens streams channels",
            streams=stream_count,
        )
        return updated_stream, NGramInjectionState(
            token_history=updated_token_history,
            conv_state=updated_conv_state,
        )

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> NGramInjectionState:  # noqa: ARG002
        hash_config = self.config.hash_config
        return NGramInjectionState(
            token_history=jnp.full((hash_config.ngram_size - 1,), hash_config.eos_token_id, dtype=jnp.int32),
            conv_state=jnp.zeros((self.conv.kernel_size - 1, self.conv.input_dim), dtype=dtype),
        )
