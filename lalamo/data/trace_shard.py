from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from lalamo.safetensors import safe_read, safe_write

TRACE_SHARD_PATTERN = "part-*.safetensors"
DEFAULT_TRACE_SHARD_SIZE_BYTES = 512 * 1024 * 1024
_PER_COMPLETION_TOKEN_FIELDS = (
    "raw_topk_token_ids",
    "raw_topk_token_logits",
    "output_norm_hidden",
)


@dataclass(frozen=True)
class RaggedTokens:
    offsets: Int[Array, " items_plus_one"]
    token_ids: Int[Array, " total_tokens"]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def token_range(self, index: int) -> tuple[int, int]:
        return tuple(map(int, self.offsets[index : index + 2].tolist()))

    def tokens_at(self, index: int) -> list[int]:
        start, end = self.token_range(index)
        return np.asarray(jax.device_get(self.token_ids[start:end]), dtype=np.int32).tolist()

    def to_tensors(self, prefix: str) -> dict[str, Array]:
        return {
            f"{prefix}_offsets": self.offsets,
            f"{prefix}_token_ids": self.token_ids,
        }

    @staticmethod
    def from_tensors(tensors: dict[str, Array], prefix: str) -> "RaggedTokens":
        return RaggedTokens(
            offsets=tensors[f"{prefix}_offsets"],
            token_ids=tensors[f"{prefix}_token_ids"],
        )

    @staticmethod
    def pack(sequences: Sequence[Sequence[int]]) -> "RaggedTokens":
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(sequence) for sequence in sequences], dtype=np.int64)
        return RaggedTokens(
            offsets=jnp.asarray(offsets),
            token_ids=jnp.asarray([token for sequence in sequences for token in sequence], dtype=jnp.int32),
        )


@dataclass(frozen=True)
class TraceCompletionRecord:
    prefix_token_ids: list[int]
    completion_token_ids: list[int]
    raw_topk_token_ids: Int[Array, "completion_tokens k"]
    raw_topk_token_logits: Float[Array, "completion_tokens k"]
    output_norm_hidden: Float[Array, "completion_tokens hidden"]
    layer_indices: tuple[int, ...] = tuple()
    layer_hidden_states: Float[Array, "traced_layers completion_tokens hidden"] | None = None

    def __post_init__(self) -> None:
        completion_tokens = len(self.completion_token_ids)
        if self.raw_topk_token_ids.shape[0] != completion_tokens:
            raise ValueError("raw_topk_token_ids length must match completion_token_ids length.")
        if self.raw_topk_token_logits.shape != self.raw_topk_token_ids.shape:
            raise ValueError("raw_topk_token_logits shape must match raw_topk_token_ids shape.")
        if self.output_norm_hidden.shape[0] != completion_tokens:
            raise ValueError("output_norm_hidden length must match completion_token_ids length.")
        if self.layer_hidden_states is None:
            if self.layer_indices:
                raise ValueError("layer_hidden_states must be provided when layer_indices are non-empty.")
        else:
            expected_shape = (len(self.layer_indices), completion_tokens, self.output_norm_hidden.shape[-1])
            if self.layer_hidden_states.shape != expected_shape:
                raise ValueError(
                    "layer_hidden_states shape must be (num_layers, completion_tokens, hidden_dim).",
                )

    @property
    def estimated_size_bytes(self) -> int:
        total = (len(self.prefix_token_ids) + len(self.completion_token_ids)) * np.dtype(np.int32).itemsize
        total += sum(
            getattr(self, field_name).size * getattr(self, field_name).dtype.itemsize
            for field_name in _PER_COMPLETION_TOKEN_FIELDS
        )
        total += len(self.layer_indices) * np.dtype(np.int32).itemsize
        if self.layer_hidden_states is not None:
            total += self.layer_hidden_states.size * self.layer_hidden_states.dtype.itemsize
        return total


@dataclass(frozen=True)
class TraceShard:
    prefix: RaggedTokens
    completion: RaggedTokens
    raw_topk_token_ids: Int[Array, " total_completion_tokens k"]
    raw_topk_token_logits: Float[Array, " total_completion_tokens k"]
    output_norm_hidden: Float[Array, " total_completion_tokens hidden"]
    layer_indices: Int[Array, " traced_layers"]
    layer_hidden_states: Float[Array, " traced_layers total_completion_tokens hidden"] | None = None

    @classmethod
    def from_path(cls, path: Path) -> "TraceShard":
        with path.open("rb") as fd:
            _, tensors = safe_read(fd)
            loaded_tensors = {
                "prefix_offsets": tensors["prefix_offsets"],
                "prefix_token_ids": tensors["prefix_token_ids"],
                "completion_offsets": tensors["completion_offsets"],
                "completion_token_ids": tensors["completion_token_ids"],
                **{field_name: tensors[field_name] for field_name in _PER_COMPLETION_TOKEN_FIELDS},
            }
            if "layer_indices" in tensors:
                loaded_tensors["layer_indices"] = tensors["layer_indices"]
            if "layer_hidden_states" in tensors:
                loaded_tensors["layer_hidden_states"] = tensors["layer_hidden_states"]
        return cls(
            prefix=RaggedTokens.from_tensors(loaded_tensors, "prefix"),
            completion=RaggedTokens.from_tensors(loaded_tensors, "completion"),
            layer_indices=loaded_tensors.get("layer_indices", jnp.zeros((0,), dtype=jnp.int32)),
            layer_hidden_states=loaded_tensors.get("layer_hidden_states"),
            **{field_name: loaded_tensors[field_name] for field_name in _PER_COMPLETION_TOKEN_FIELDS},
        )

    @classmethod
    def iter_paths(cls, trace_dir: Path) -> Iterator[Path]:
        shard_paths = sorted(trace_dir.glob(TRACE_SHARD_PATTERN))
        if not shard_paths:
            raise RuntimeError(f"No trace shards found in {trace_dir}.")
        yield from shard_paths

    @classmethod
    def iter_records(cls, trace_dir: Path) -> Iterator[TraceCompletionRecord]:
        for shard_path in cls.iter_paths(trace_dir):
            yield from cls.from_path(shard_path).iter_completions()

    def iter_completions(self) -> Iterator[TraceCompletionRecord]:
        layer_indices = tuple(np.asarray(jax.device_get(self.layer_indices), dtype=np.int32).tolist())
        for index in range(len(self.prefix)):
            completion_start, completion_end = self.completion.token_range(index)
            yield TraceCompletionRecord(
                prefix_token_ids=self.prefix.tokens_at(index),
                completion_token_ids=self.completion.tokens_at(index),
                layer_indices=layer_indices,
                layer_hidden_states=(
                    self.layer_hidden_states[:, completion_start:completion_end, :]
                    if self.layer_hidden_states is not None
                    else None
                ),
                **{
                    field_name: getattr(self, field_name)[completion_start:completion_end]
                    for field_name in _PER_COMPLETION_TOKEN_FIELDS
                },
            )


@dataclass
class TraceShardWriter:
    output_dir: Path
    target_shard_size_bytes: int = DEFAULT_TRACE_SHARD_SIZE_BYTES
    shard_index: int = 0
    _records: list[TraceCompletionRecord] = field(default_factory=list, repr=False)
    _estimated_size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.target_shard_size_bytes <= 0:
            raise ValueError("target_shard_size_bytes must be positive.")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise RuntimeError(f"{self.output_dir} exists and is not a directory.")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise RuntimeError(f"{self.output_dir} must be empty for collect-traces output.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: TraceCompletionRecord) -> None:
        if self._records and record.layer_indices != self._records[0].layer_indices:
            raise ValueError("All records in a trace shard must use the same layer_indices.")
        if self._records and self._estimated_size_bytes + record.estimated_size_bytes > self.target_shard_size_bytes:
            self.flush()
        self._records.append(record)
        self._estimated_size_bytes += record.estimated_size_bytes

    def flush(self) -> None:
        if not self._records:
            return

        tensors: dict[str, Array] = {
            **RaggedTokens.pack([record.prefix_token_ids for record in self._records]).to_tensors("prefix"),
            **RaggedTokens.pack([record.completion_token_ids for record in self._records]).to_tensors("completion"),
            **{
                field_name: jnp.concatenate([getattr(record, field_name) for record in self._records], axis=0)
                for field_name in _PER_COMPLETION_TOKEN_FIELDS
            },
            "layer_indices": jnp.asarray(self._records[0].layer_indices, dtype=jnp.int32),
        }

        if self._records[0].layer_hidden_states is not None:
            tensors["layer_hidden_states"] = jnp.concatenate(
                [record.layer_hidden_states for record in self._records if record.layer_hidden_states is not None],
                axis=1,
            )

        shard_path = self.output_dir / f"part-{self.shard_index:05d}.safetensors"
        with shard_path.open("wb") as fd:
            safe_write(fd, tensors)

        self.shard_index += 1
        self._records.clear()
        self._estimated_size_bytes = 0

    def finish(self) -> None:
        self.flush()
