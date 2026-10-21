import json
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lalamo.utils.lazy_collections import LazyDict

from .origins import FileSpec, Origin, StatusEvent, WeightShard, WeightShardContext

__all__ = ["VocabularyOrderedOrigin", "frequency_order"]


def frequency_order(counts_path: Path, vocab_size: int) -> np.ndarray:
    """New id -> old id: most counted first, ties in the old order. `counts_path` holds [token_id, count] pairs."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for token_id, count in json.loads(counts_path.read_text()):
        counts[token_id] += count
    return np.argsort(-counts, kind="stable")


def _rename_ids(value: object, new_ids: np.ndarray, is_id: bool = False) -> object:
    """Every integer under an `id`, `ids`, `*token_id` or `*token_ids` key is a token id."""
    match value:
        case dict():
            return {
                key: _rename_ids(item, new_ids, key in ("id", "ids") or key.endswith(("token_id", "token_ids")))
                for key, item in value.items()
            }
        case list():
            return [_rename_ids(item, new_ids, is_id) for item in value]
        case int() if is_id:
            return int(new_ids[value])
        case _:
            return value


@dataclass(frozen=True, eq=False)
class VocabularyOrderedOrigin(Origin):
    """`origin` with its vocabulary renumbered by `order` (new id -> old id). Vocabulary rows of the weights are
    permuted and every token id in the JSON files is rewritten, so the model computes the same function."""

    origin: Origin
    order: np.ndarray

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path:
        path = self.origin.resolve_file(file_spec, progress_callback)
        if path.suffix != ".json":
            return path
        new_ids = np.argsort(self.order)
        original = json.loads(path.read_text())
        content = _rename_ids(original, new_ids)
        assert isinstance(content, dict)
        if "added_tokens_decoder" in content:
            decoder = content["added_tokens_decoder"]
            content["added_tokens_decoder"] = {str(new_ids[int(i)]): token for i, token in decoder.items()}
        if "added_tokens" in content:
            # `tokenizers` numbers added tokens after the model vocabulary unless the vocabulary lists them.
            vocab = {piece: int(new_ids[i]) for piece, i in original["model"]["vocab"].items()}
            content["model"]["vocab"] = vocab | {token["content"]: token["id"] for token in content["added_tokens"]}
        renamed = Path(tempfile.mkdtemp()) / path.name
        renamed.write_text(json.dumps(content))
        return renamed

    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShardContext]:
        return tuple(self._renumbered(shard) for shard in self.origin.get_weights(progress_callback))

    @contextmanager
    def _renumbered(self, shard: WeightShardContext) -> Generator[WeightShard]:
        with shard as (weights, metadata):

            def get(name: str) -> jnp.ndarray:
                tensor = weights[name]
                return tensor[self.order] if tensor.ndim and tensor.shape[0] == len(self.order) else tensor

            yield LazyDict(set(weights), get), metadata

    @property
    def description(self) -> str:
        return self.origin.description
