import functools

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.modules.decoder import DecoderActivationTrace
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import StaticKVCacheLayer


def extract_hiddens(
    trace: DecoderActivationTrace,
    batch: int,
    token: int,
) -> tuple[jnp.ndarray, ...]:
    """Extract per-layer hidden states + output norm at a single position."""
    return tuple(lr.outputs[batch, token] for lr in trace.layer_results) + (trace.output_norm[batch, token],)


def top_k_from_logits(logits: jnp.ndarray, k: int) -> list[int]:
    arr = np.asarray(logits)
    if k >= arr.shape[0]:
        return list(np.argsort(arr)[::-1])
    indices = np.argpartition(arr, -k)[-k:]
    return list(indices[np.argsort(arr[indices])[::-1]])


@functools.partial(jax.jit, static_argnames=("max_slots",))
def compact_kv_cache(
    state: State,
    cache_len: jnp.ndarray,
    accepted_indices: jnp.ndarray,
    num_accepted: jnp.ndarray,
    max_slots: int,
) -> State:
    """Keep accepted KV entries, discard rejected draft tokens.

    JIT-compiled: all layers x 2 scatter ops fuse into a single XLA program.
    """
    dst = jnp.arange(max_slots, dtype=jnp.int32) + cache_len
    src = cache_len + accepted_indices
    valid = jnp.arange(max_slots) < num_accepted
    src = jnp.where(valid, src, dst)
    new_length = cache_len + num_accepted

    def compact_layer(layer: StaticKVCacheLayer) -> StaticKVCacheLayer:
        return StaticKVCacheLayer(
            has_sinks=layer.has_sinks,
            keys=layer.keys.at[:, dst].set(layer.keys[:, src]),
            values=layer.values.at[:, dst].set(layer.values[:, src]),
            current_length=jnp.full_like(layer.current_length, new_length),
        )

    return State(compact_layer(layer) for layer in state)
