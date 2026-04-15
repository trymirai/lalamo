from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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
    logsumexp: Float[Array, " tokens"] | None = None
    activation_output: Float[Array, "tokens hidden"] | None = None
    layer_indices: tuple[int, ...] = ()
    layer_output: Float[Array, "n_layers tokens hidden"] | None = None

    def __post_init__(self) -> None:
        n = len(self.completion_token_ids)
        if self.top_k_ids.shape[0] != n:
            raise ValueError("top_k_ids first dim must match completion length.")
        if self.top_k_logits.shape != self.top_k_ids.shape:
            raise ValueError("top_k_logits shape must match top_k_ids shape.")
        if self.logsumexp is not None and self.logsumexp.shape[0] != n:
            raise ValueError("logsumexp first dim must match completion length.")
        if self.activation_output is not None and self.activation_output.shape[0] != n:
            raise ValueError("activation_output first dim must match completion length.")
        if self.layer_output is None and self.layer_indices:
            raise ValueError("layer_indices must be empty when layer_output is None.")
        if self.layer_output is not None and self.activation_output is not None:
            expected = (len(self.layer_indices), n, self.activation_output.shape[-1])
            if self.layer_output.shape != expected:
                raise ValueError(f"layer_output shape {self.layer_output.shape} != expected {expected}.")

    @property
    def completion_token_logits(self) -> list[dict[int, float]]:
        ids = np.asarray(self.top_k_ids, dtype=np.int32)
        vals = np.asarray(self.top_k_logits, dtype=np.float32)
        return [
            dict(zip(row_ids.tolist(), row_vals.tolist(), strict=True))
            for row_ids, row_vals in zip(ids, vals, strict=True)
        ]


def pack_ragged(sequences: list[list[int]]) -> tuple[Array, Array]:
    offsets = np.cumsum([0, *(len(s) for s in sequences)], dtype=np.int64)
    flat = [t for s in sequences for t in s]
    return jnp.asarray(offsets), jnp.asarray(flat, dtype=jnp.int32)


def unpack_ragged(offsets: Array, flat: Array) -> list[list[int]]:
    offset_array = np.asarray(offsets, dtype=np.int64)
    flat_array = np.asarray(flat, dtype=np.int32)
    return [flat_array[offset_array[i] : offset_array[i + 1]].tolist() for i in range(len(offset_array) - 1)]


def validate_shared_trace_layout(
    completions: list[LalamoCompletion],
) -> tuple[tuple[int, ...], bool, bool, bool]:
    first_completion, *remaining_completions = completions
    layer_indices = first_completion.layer_indices
    has_logsumexp = first_completion.logsumexp is not None
    has_activation_output = first_completion.activation_output is not None
    has_layer_output = first_completion.layer_output is not None

    for completion in remaining_completions:
        if completion.layer_indices != layer_indices:
            raise ValueError("All completions must use the same layer_indices.")
        if (completion.logsumexp is not None) != has_logsumexp:
            raise ValueError("All completions must either include logsumexp or omit it.")
        if (completion.activation_output is not None) != has_activation_output:
            raise ValueError("All completions must either include activation_output or omit it.")
        if (completion.layer_output is not None) != has_layer_output:
            raise ValueError("All completions must either include layer_output or omit it.")

    return layer_indices, has_logsumexp, has_activation_output, has_layer_output


def save_completions(path: Path, completions: list[LalamoCompletion]) -> None:
    if not completions:
        return
    layer_indices, has_logsumexp, has_activation_output, has_layer_output = validate_shared_trace_layout(completions)
    prefix_off, prefix_flat = pack_ragged([c.prefix_token_ids for c in completions])
    comp_off, comp_flat = pack_ragged([c.completion_token_ids for c in completions])
    arrays: dict[str, Array] = {
        "prefix_offsets": prefix_off,
        "prefix_tokens": prefix_flat,
        "completion_offsets": comp_off,
        "completion_tokens": comp_flat,
        "top_k_ids": jnp.concatenate([c.top_k_ids for c in completions], axis=0),
        "top_k_logits": jnp.concatenate([c.top_k_logits for c in completions], axis=0),
        "layer_indices": jnp.asarray(layer_indices, dtype=jnp.int32),
    }
    if has_logsumexp:
        arrays["logsumexp"] = jnp.concatenate(
            [c.logsumexp for c in completions if c.logsumexp is not None],
            axis=0,
        )
    if has_activation_output:
        arrays["activation_output"] = jnp.concatenate(
            [c.activation_output for c in completions if c.activation_output is not None],
            axis=0,
        )
    if has_layer_output:
        arrays["layer_output"] = jnp.concatenate(
            [c.layer_output for c in completions if c.layer_output is not None],
            axis=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fd:
        safe_write(fd, arrays)


def load_completions(path: Path, exclude: set[str] | None = None) -> list[LalamoCompletion]:
    with path.open("rb") as fd:
        _, lazy = safe_read(fd)
        arrays = {k: lazy[k] for k in lazy if not exclude or k not in exclude}
    prefixes = unpack_ragged(arrays["prefix_offsets"], arrays["prefix_tokens"])
    completions = unpack_ragged(arrays["completion_offsets"], arrays["completion_tokens"])
    offsets = np.asarray(arrays["completion_offsets"], dtype=np.int64)
    layer_indices_array = arrays.get("layer_indices")
    layer_indices = (
        tuple(np.asarray(layer_indices_array, dtype=np.int32).tolist()) if layer_indices_array is not None else ()
    )

    def sliced(key: str, token_slice: slice) -> Array | None:
        a = arrays.get(key)
        return a[token_slice] if a is not None else None

    loaded: list[LalamoCompletion] = []
    for i, (prefix, completion) in enumerate(zip(prefixes, completions, strict=True)):
        s = slice(int(offsets[i]), int(offsets[i + 1]))
        layer_output_array = arrays.get("layer_output")
        loaded.append(
            LalamoCompletion(
                prefix_token_ids=prefix,
                completion_token_ids=completion,
                top_k_ids=arrays["top_k_ids"][s],
                top_k_logits=arrays["top_k_logits"][s],
                logsumexp=sliced("logsumexp", s),
                activation_output=sliced("activation_output", s),
                layer_indices=layer_indices if layer_output_array is not None else (),
                layer_output=layer_output_array[:, s, :] if layer_output_array is not None else None,
            )
        )
    return loaded


def iter_completions(trace_dir: Path, exclude: set[str] | None = None) -> Iterator[LalamoCompletion]:
    shard_paths = sorted(trace_dir.glob("part-*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No trace shards found in {trace_dir}.")
    for shard_path in shard_paths:
        yield from load_completions(shard_path, exclude=exclude)
