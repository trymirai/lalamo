import math
from collections.abc import Sequence
from itertools import accumulate, pairwise
from typing import Literal

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float

from lalamo.module import Keychain
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.utils import call_vmapped

from .forward_pass_config import VisionForwardPassConfig
from .image_grid import ImageGrid

__all__ = ["vision_attention"]


def _is_cudnn_supported(queries: Float[Array, "tokens heads head_channels"]) -> bool:
    device = jax.typeof(queries).sharding.mesh.abstract_device
    platform = device.platform if device is not None else jax.default_backend()
    _, _, head_dim = queries.shape
    # 128 is the head size limit on pre-Hopper GPUs; Hopper and later allow 256.
    return platform == "gpu" and queries.dtype in (jnp.float16, jnp.bfloat16) and head_dim % 8 == 0 and head_dim <= 128


def _attend(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens heads head_channels"],
    values: Float[Array, "src_tokens heads head_channels"],
    implementation: Literal["cudnn", "xla"],
) -> Float[Array, "dst_tokens heads head_channels"]:
    # Inputs share one dtype; "standard" promotion only matters for the internal float32 softmax.
    with jax.numpy_dtype_promotion("standard"):
        return jax.nn.dot_product_attention(queries, keys, values, implementation=implementation)


def _bidirectional_attention(
    queries: Float[Array, "tokens heads head_channels"],
    keys: Float[Array, "tokens heads head_channels"],
    values: Float[Array, "tokens heads head_channels"],
    forward_pass_config: VisionForwardPassConfig,
) -> Float[Array, "tokens heads head_channels"]:
    if forward_pass_config.attention_implementation == "cudnn" and _is_cudnn_supported(
        queries,
    ):
        return _attend(queries, keys, values, "cudnn")

    # Exact attention, computed for a chunk of queries at a time: softmax runs over keys only, so query rows
    # are independent. The last chunk is padded with dummy queries whose outputs are dropped.
    num_tokens, _, _ = queries.shape
    chunk_size = min(forward_pass_config.attention_query_chunk_size, num_tokens)
    num_chunks = math.ceil(num_tokens / chunk_size)
    padded_queries = jnp.pad(queries, ((0, num_chunks * chunk_size - num_tokens), (0, 0), (0, 0)))
    query_chunks = rearrange(
        padded_queries,
        "(chunks chunk_tokens) heads head_channels -> chunks chunk_tokens heads head_channels",
        chunks=num_chunks,
    )
    output_chunks = jax.lax.map(
        lambda chunk: _attend(chunk, keys, values, "xla"),
        query_chunks,
    )
    outputs = rearrange(
        output_chunks,
        "chunks chunk_tokens heads head_channels -> (chunks chunk_tokens) heads head_channels",
    )
    return outputs[:num_tokens]


@eqx.filter_jit
def vision_attention(
    attention: Attention,
    inputs: Float[Array, "tokens channels"],
    positional_embeddings: PositionalEmbeddings,
    grids: Sequence[ImageGrid],
    forward_pass_config: VisionForwardPassConfig = VisionForwardPassConfig(),
    *,
    keychain: Keychain,
) -> Float[Array, "tokens channels"]:
    qkv_keychain, out_keychain = keychain.split()
    queries, keys, values = (
        rearrange(
            projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=attention.config.num_heads,
        )
        for projection in call_vmapped(
            attention.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=qkv_keychain,
        )
    )
    # The rotation runs in float32 and is cast back, the same as HF does.
    queries, keys = (
        call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1).astype(heads.dtype)
        for heads in (queries, keys)
    )

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
        attention.out_projection,
        rearrange(attended, "tokens heads head_channels -> tokens (heads head_channels)"),
        forward_pass_config=forward_pass_config.matmul_config,
        keychain=out_keychain,
    )
    return result
