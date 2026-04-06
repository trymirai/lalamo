from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from lalamo.safetensors import safe_read, safe_write


@dataclass(frozen=True)
class LalamoCompletion:
    prefix_token_ids: list[int]
    completion_token_ids: list[int]
    top_k_ids: Int[Array, "tokens k"]
    top_k_logits: Float[Array, "tokens k"]
    logsumexp: Float[Array, " tokens"]
    activation_output: Float[Array, "tokens hidden"]
    layer_indices: tuple[int, ...]
    layer_output: Float[Array, "n_layers tokens hidden"] | None

    def __post_init__(self) -> None:
        n = len(self.completion_token_ids)
        if self.top_k_ids.shape[0] != n:
            raise ValueError("top_k_ids first dim must match completion length.")
        if self.top_k_logits.shape != self.top_k_ids.shape:
            raise ValueError("top_k_logits shape must match top_k_ids shape.")
        if self.activation_output.shape[0] != n:
            raise ValueError("activation_output first dim must match completion length.")
        if self.layer_output is not None:
            expected = (len(self.layer_indices), n, self.activation_output.shape[-1])
            if self.layer_output.shape != expected:
                raise ValueError(f"layer_output shape {self.layer_output.shape} != expected {expected}.")

    @property
    def completion_token_logits(self) -> list[dict[int, float]]:
        ids = np.asarray(jax.device_get(self.top_k_ids), dtype=np.int32)
        vals = np.asarray(jax.device_get(self.top_k_logits), dtype=np.float32)
        return [dict(zip(row_ids.tolist(), row_vals.tolist(), strict=True)) for row_ids, row_vals in zip(ids, vals)]


def _pack_ragged(sequences: list[list[int]]) -> tuple[Array, Array]:
    offsets = np.cumsum([0, *(len(s) for s in sequences)], dtype=np.int64)
    flat = [t for s in sequences for t in s]
    return jnp.asarray(offsets), jnp.asarray(flat, dtype=jnp.int32)


def _unpack_ragged(offsets: Array, flat: Array) -> list[list[int]]:
    off = np.asarray(jax.device_get(offsets))
    arr = np.asarray(jax.device_get(flat), dtype=np.int32)
    return [arr[off[i] : off[i + 1]].tolist() for i in range(len(off) - 1)]


def save_completions(path: Path, completions: list[LalamoCompletion]) -> None:
    if not completions:
        return
    prefix_off, prefix_flat = _pack_ragged([c.prefix_token_ids for c in completions])
    comp_off, comp_flat = _pack_ragged([c.completion_token_ids for c in completions])
    tensors: dict[str, Array] = {
        "prefix_offsets": prefix_off,
        "prefix_tokens": prefix_flat,
        "completion_offsets": comp_off,
        "completion_tokens": comp_flat,
        "top_k_ids": jnp.concatenate([c.top_k_ids for c in completions], axis=0),
        "top_k_logits": jnp.concatenate([c.top_k_logits for c in completions], axis=0),
        "logsumexp": jnp.concatenate([c.logsumexp for c in completions], axis=0),
        "activation_output": jnp.concatenate([c.activation_output for c in completions], axis=0),
        "layer_indices": jnp.asarray(completions[0].layer_indices, dtype=jnp.int32),
    }
    if completions[0].layer_output is not None:
        tensors["layer_output"] = jnp.concatenate(
            [c.layer_output for c in completions if c.layer_output is not None], axis=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fd:
        safe_write(fd, tensors)


def load_completions(path: Path) -> list[LalamoCompletion]:
    with path.open("rb") as fd:
        _, lazy = safe_read(fd)
        tensors = {k: lazy[k] for k in lazy}  # eagerly materialize before fd closes
    prefixes = _unpack_ragged(tensors["prefix_offsets"], tensors["prefix_tokens"])
    completions_tok = _unpack_ragged(tensors["completion_offsets"], tensors["completion_tokens"])
    layer_indices = tuple(np.asarray(jax.device_get(tensors["layer_indices"]), dtype=np.int32).tolist())
    has_layers = "layer_output" in tensors
    comp_off = np.asarray(jax.device_get(tensors["completion_offsets"]))
    return [
        LalamoCompletion(
            prefix_token_ids=prefix,
            completion_token_ids=comp_tok,
            top_k_ids=tensors["top_k_ids"][int(comp_off[i]) : int(comp_off[i + 1])],
            top_k_logits=tensors["top_k_logits"][int(comp_off[i]) : int(comp_off[i + 1])],
            logsumexp=tensors["logsumexp"][int(comp_off[i]) : int(comp_off[i + 1])],
            activation_output=tensors["activation_output"][int(comp_off[i]) : int(comp_off[i + 1])],
            layer_indices=layer_indices,
            layer_output=tensors["layer_output"][:, int(comp_off[i]) : int(comp_off[i + 1]), :] if has_layers else None,
        )
        for i, (prefix, comp_tok) in enumerate(zip(prefixes, completions_tok, strict=True))
    ]


def iter_completions(trace_dir: Path) -> Iterator[LalamoCompletion]:
    shard_paths = sorted(trace_dir.glob("part-*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No trace shards found in {trace_dir}.")
    for shard_path in shard_paths:
        yield from load_completions(shard_path)
