import jax.numpy as jnp
from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha
from jaxtyping import Array, Float

PALLAS_TRAINING_BLOCKS = BlockSizes(
    block_q=64,
    block_k=64,
    block_q_dkv=32,
    block_kv_dkv=32,
    block_q_dq=32,
    block_kv_dq=32,
)


def pallas_training_attention(
    queries: Float[Array, "tokens heads head_channels"],
    keys: Float[Array, "tokens groups head_channels"],
    values: Float[Array, "tokens groups head_channels"],
    scale: float | None,
) -> Float[Array, "tokens heads head_channels"]:
    sequence_length = queries.shape[0]
    heads_per_group = queries.shape[1] // keys.shape[1]
    keys = jnp.repeat(keys, heads_per_group, axis=1)
    values = jnp.repeat(values, heads_per_group, axis=1)

    padding = (-sequence_length) % PALLAS_TRAINING_BLOCKS.block_q
    pad_width = ((0, padding), (0, 0), (0, 0))
    queries = jnp.pad(queries, pad_width)
    keys = jnp.pad(keys, pad_width)
    values = jnp.pad(values, pad_width)

    outputs = mha(
        queries[None],
        keys[None],
        values[None],
        segment_ids=None,
        sm_scale=queries.shape[-1] ** -0.5 if scale is None else scale,
        causal=True,
        block_sizes=PALLAS_TRAINING_BLOCKS,
    )
    return outputs[0, :sequence_length]
