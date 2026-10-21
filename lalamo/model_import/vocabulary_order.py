"""Renumber a model's vocabulary so the most frequent tokens get the smallest ids.

A speculator can then draft over a prefix of the target readout: its K most frequent tokens are exactly rows 0..K-1,
for every K, with no gather and no id table at runtime. The renumbering is applied at the source, before the model is
built: vocabulary rows of the weights are permuted, and every token id in the tokenizer and the JSON configs is
rewritten. The model computes the same function; only the labels of its tokens change.
"""

import json
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lalamo.utils.lazy_collections import LazyDict

from .origins import FileSpec, Origin, StatusEvent, WeightShard, WeightShardContext

__all__ = ["VocabularyOrderedOrigin", "frequency_order", "vocabulary_size"]


def frequency_order(counts_path: Path, vocab_size: int) -> np.ndarray:
    """new_id -> old_id: counted tokens by descending count (ties by id), then every uncounted id in order.
    The counts file is a JSON list of [token_id, count]."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for token_id, count in json.loads(counts_path.read_text()):
        assert 0 <= token_id < vocab_size, f"token id {token_id} outside the vocabulary of {vocab_size}"
        counts[token_id] += count
    return np.lexsort((np.arange(vocab_size), -counts))


def vocabulary_size(config: dict) -> int:
    """The single `vocab_size` in a model config (top level or nested, e.g. a VL model's text_config)."""
    sizes = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            sizes.update(item for key, item in value.items() if key == "vocab_size")
            for item in value.values():
                collect(item)

    collect(config)
    assert len(sizes) == 1, f"expected one vocab_size in the model config, found {sorted(sizes)}"
    (size,) = sizes
    return size


def _rename_token_ids(value: object, new_ids: np.ndarray) -> object:
    """Rewrite every value under a key ending in `token_id` or `token_ids` (an id or a list of ids), recursively."""
    match value:
        case dict():
            renamed = {}
            for key, item in value.items():
                if key.endswith(("token_id", "token_ids")) and item is not None:
                    renamed[key] = [int(new_ids[i]) for i in item] if isinstance(item, list) else int(new_ids[item])
                else:
                    renamed[key] = _rename_token_ids(item, new_ids)
            return renamed
        case list():
            return [_rename_token_ids(item, new_ids) for item in value]
        case _:
            return value


def _rename_tokenizer(tokenizer: dict, new_ids: np.ndarray) -> dict:
    model = tokenizer["model"]
    assert isinstance(model.get("vocab"), dict), f"only id-keyed vocabularies are supported, got {model['type']}"
    pieces = {piece: int(new_ids[i]) for piece, i in model["vocab"].items()}
    added_tokens = [{**added, "id": int(new_ids[added["id"]])} for added in tokenizer["added_tokens"]]
    assert not any(added["content"] in pieces for added in added_tokens), "an added token is also a vocabulary piece"
    # `tokenizers` numbers added tokens after the model vocabulary on load unless the model vocabulary has them.
    vocab = pieces | {added["content"]: added["id"] for added in added_tokens}
    return {
        **tokenizer,
        "model": {**model, "vocab": vocab},
        "added_tokens": added_tokens,
        "post_processor": _rename_post_processor_ids(tokenizer["post_processor"], new_ids),
    }


def _rename_post_processor_ids(value: object, new_ids: np.ndarray) -> object:
    """Template post-processors list their special tokens' ids under `ids`."""
    match value:
        case dict():
            return {
                key: [int(new_ids[i]) for i in item] if key == "ids" else _rename_post_processor_ids(item, new_ids)
                for key, item in value.items()
            }
        case list():
            return [_rename_post_processor_ids(item, new_ids) for item in value]
        case _:
            return value


def _rename_json(name: str, content: dict, new_ids: np.ndarray) -> object:
    if name == "tokenizer.json":
        return _rename_tokenizer(content, new_ids)
    if name == "tokenizer_config.json" and "added_tokens_decoder" in content:
        decoder = content["added_tokens_decoder"]
        content = {**content, "added_tokens_decoder": {str(int(new_ids[int(i)])): t for i, t in decoder.items()}}
    return _rename_token_ids(content, new_ids)


@dataclass(frozen=True, eq=False)
class VocabularyOrderedOrigin(Origin):
    """`origin` renumbered by `order` (new_id -> old_id over the model's whole vocabulary)."""

    origin: Origin
    order: np.ndarray
    _directory: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="lalamo-vocabulary-")))

    @cached_property
    def new_ids(self) -> np.ndarray:
        return np.argsort(self.order)

    def resolve_file(
        self,
        file_spec: FileSpec,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Path:
        path = self.origin.resolve_file(file_spec, progress_callback)
        if path.suffix != ".json":
            return path
        renamed = self._directory / path.name
        renamed.write_text(json.dumps(_rename_json(path.name, json.loads(path.read_text()), self.new_ids)))
        return renamed

    def get_weights(
        self,
        progress_callback: Callable[[StatusEvent], None] | None = None,
    ) -> Sequence[WeightShardContext]:
        return tuple(self._renumbered(shard) for shard in self.origin.get_weights(progress_callback))

    @contextmanager
    def _renumbered(self, shard: WeightShardContext) -> Generator[WeightShard]:
        """Vocabulary rows (tensors whose first axis is the vocabulary) are read in the new order."""
        vocab_size = len(self.order)
        with shard as (weights, metadata):

            def get(name: str) -> jnp.ndarray:
                tensor = weights[name]
                return tensor[self.order] if tensor.ndim >= 1 and tensor.shape[0] == vocab_size else tensor

            yield LazyDict(set(weights), get), metadata

    @property
    def description(self) -> str:
        return f"{self.origin.description} (frequency-ordered vocabulary)"
