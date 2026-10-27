import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.utils import call_vmapped

from .forward_pass_config import VisionAttentionImplementation, VisionForwardPassConfig
from .image_grid import ImageGrid

__all__ = [
    "VisionAttention",
    "VisionAttentionConfig",
]


def _attend(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens heads head_channels"],
    values: Float[Array, "src_tokens heads head_channels"],
    implementation: VisionAttentionImplementation,
) -> Float[Array, "dst_tokens heads head_channels"]:
    # Inputs share one dtype; "standard" promotion only matters for the internal float32 softmax.
    with jax.numpy_dtype_promotion("standard"):
        match implementation:
            case VisionAttentionImplementation.XLA:
                return jax.nn.dot_product_attention(queries, keys, values, implementation="xla")
            case VisionAttentionImplementation.CUDNN:
                return jax.nn.dot_product_attention(queries, keys, values, implementation="cudnn")


def _bidirectional_attention(
    queries: Float[Array, "tokens heads head_channels"],
    keys: Float[Array, "tokens heads head_channels"],
    values: Float[Array, "tokens heads head_channels"],
    forward_pass_config: VisionForwardPassConfig,
) -> Float[Array, "tokens heads head_channels"]:
    implementation = forward_pass_config.attention_implementation
    chunk_size = forward_pass_config.attention_query_chunk_size
    num_tokens, _, _ = queries.shape

    # Flash attention is already memory efficient, and short sequences fit comfortably.
    if implementation == VisionAttentionImplementation.CUDNN or chunk_size is None or num_tokens <= chunk_size:
        return _attend(queries, keys, values, implementation)

    # Softmax runs over keys only, so query rows are independent and can be processed in slices.
    # Padding queries (never keys) keeps every slice the same shape, which lets lax.map compile
    # a single attention kernel and reuse its buffers across slices.
    num_chunks = math.ceil(num_tokens / chunk_size)
    padded_queries = jnp.pad(queries, ((0, num_chunks * chunk_size - num_tokens), (0, 0), (0, 0)))
    query_chunks = rearrange(
        padded_queries,
        "(chunks chunk_tokens) heads head_channels -> chunks chunk_tokens heads head_channels",
        chunks=num_chunks,
    )
    output_chunks = jax.lax.map(lambda chunk: _attend(chunk, keys, values, implementation), query_chunks)
    outputs = rearrange(
        output_chunks,
        "chunks chunk_tokens heads head_channels -> (chunks chunk_tokens) heads head_channels",
    )
    return outputs[:num_tokens]


@dataclass(frozen=True)
class VisionAttentionConfig(LalamoConfig):
    """Bidirectional multi-head self-attention with 2D rotary embeddings.

    Differences from the decoder attention: no causal mask, no KV cache, no grouped heads, and
    attention never crosses an image boundary. HF builds one packed sequence for all images and
    restricts attention with ``cu_seqlens``, with one segment per frame. Because frame sizes are
    static here, the packed sequence is cut into those segments with plain slices. Each segment
    gets its own small attention call instead of one big call with a block-diagonal mask, which
    avoids building a ``tokens x tokens`` mask and wastes no FLOPs on masked-out blocks.
    """

    qkv_projection_config: LinearConfig
    out_projection_config: LinearConfig
    num_heads: int
    head_dim: int
    has_biases: bool

    @property
    def attention_dim(self) -> int:
        return self.num_heads * self.head_dim

    def init(self, initializer: Initializer, model_dim: int) -> "VisionAttention":
        return VisionAttention(
            config=self,
            sharding_config=initializer.sharding_config,
            qkv_projection=self.qkv_projection_config.init(
                initializer,
                model_dim,
                (self.attention_dim,) * 3,
                has_biases=self.has_biases,
            ),
            out_projection=self.out_projection_config.init(
                initializer,
                self.attention_dim,
                (model_dim,),
                has_biases=self.has_biases,
            ),
        )


class VisionAttention(LalamoModule[VisionAttentionConfig]):
    qkv_projection: Linear
    out_projection: Linear

    def _split_heads(self, projection: Float[Array, "tokens channels"]) -> Float[Array, "tokens heads head_channels"]:
        return rearrange(
            projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=self.config.num_heads,
        )

    @staticmethod
    def _rotate(
        heads: Float[Array, "tokens heads head_channels"],
        positional_embeddings: PositionalEmbeddings,
    ) -> Float[Array, "tokens heads head_channels"]:
        # The rotation runs in float32 and is cast back, the same as HF does.
        rotated = call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)
        return rotated.astype(heads.dtype)

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        grids: Sequence[ImageGrid],
        forward_pass_config: VisionForwardPassConfig = VisionForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        qkv_keychain, out_keychain = keychain.split()
        queries, keys, values = (
            self._split_heads(projection)
            for projection in call_vmapped(
                self.qkv_projection,
                inputs,
                forward_pass_config=forward_pass_config.matmul_config,
                keychain=qkv_keychain,
            )
        )
        queries = self._rotate(queries, positional_embeddings)
        keys = self._rotate(keys, positional_embeddings)

        frame_lengths = [grid.patches_per_frame for grid in grids for _ in range(grid.num_frames)]
        frame_bounds = list(accumulate(frame_lengths, initial=0))
        attended = jnp.concatenate(
            [
                _bidirectional_attention(
                    queries[start:end],
                    keys[start:end],
                    values[start:end],
                    forward_pass_config,
                )
                for start, end in pairwise(frame_bounds)
            ],
            axis=0,
        )

        (result,) = call_vmapped(
            self.out_projection,
            rearrange(attended, "tokens heads head_channels -> tokens (heads head_channels)"),
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
        )
        return result
