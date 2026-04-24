import dataclasses
import random
import struct
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain, repeat, tee
from math import exp
from pathlib import Path
from typing import Annotated, ClassVar, Self

import xxhash
from typer import Argument, Option, Typer

from lalamo.data.lalamo_completions import LalamoCompletion, iter_completions
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.speculator.common import LMState, SamplerConfig
from lalamo.speculator.sampler import GumbelSeed, gumbel_rank_from_probs
from lalamo.speculator.trie import TreeSpeculator, TreeVerifyResult, TrieNode


def padded_sliding_window(seq: Iterable[int], size: int, pad: int) -> Iterable[tuple[int, ...]]:
    seqs = tee(seq, size)
    pads = tuple(repeat(pad, size - i) for i in range(size))
    padded_seqs = tuple(chain(pad, seq) for pad, seq in zip(pads, seqs, strict=True))
    return zip(*padded_seqs, strict=False)


def softmax(logits: Iterable[float]) -> list[float]:
    logits = list(logits)
    log_max = max(logits)
    exp_logs = [exp(logit - log_max) for logit in logits]
    exp_log_sum = sum(exp_logs)
    return [exp_log / exp_log_sum for exp_log in exp_logs]


def mean_update(probs: dict[int, float], sample: dict[int, float], new_count: int, top_k: int) -> dict[int, float]:
    decay = (new_count - 1) / new_count
    inv_n = 1.0 / new_count
    merged = {k: v * decay for k, v in probs.items()}
    for k, v in sample.items():
        merged[k] = merged.get(k, 0.0) + v * inv_n
    top_items = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    total = sum(v for _, v in top_items)
    if total > 0:
        return {k: v / total for k, v in top_items}
    return dict(top_items)


def seqhash(tokens: Iterable[int], size: int) -> tuple[int, int]:
    tokens = list(tokens)
    packed = struct.pack("<" + "I" * len(tokens), *tokens) if tokens else b""
    h = xxhash.xxh3_64_intdigest(packed)
    return h % size, h // size


def _discount_and_interpolate(
    dist: dict[int, float],
    lower_dist: dict[int, float],
    discount: float,
) -> dict[int, float]:
    num_nonzero = sum(1 for v in dist.values() if v > 0)
    freed_mass = min(discount * num_nonzero, 0.5)
    smoothed = {k: max(v - discount, 0.0) for k, v in dist.items()}
    for token, prob in lower_dist.items():
        smoothed[token] = smoothed.get(token, 0.0) + freed_mass * prob
    return smoothed


def _write_top_k(dist: dict[int, float], keys: array, values: array, idx: int, ngram_k: int) -> None:
    top = sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:ngram_k]
    total = sum(v for _, v in top)
    if total <= 0:
        return
    new_keys = array("I", repeat(0, ngram_k))
    new_values = array("f", repeat(0.0, ngram_k))
    for i, (token_id, prob) in enumerate(top):
        new_keys[i] = token_id
        new_values[i] = prob / total
    k_start = idx * ngram_k
    memoryview(keys)[k_start : k_start + ngram_k] = new_keys
    memoryview(values)[k_start : k_start + ngram_k] = new_values


class ExactBucket:
    __slots__ = ("count", "probs")

    def __init__(self) -> None:
        self.count: int = 0
        self.probs: dict[int, float] = {}


@dataclass(eq=False)
class TaggedNGramTable:
    hashtable_size: int
    ngram_k: int
    ngram_n: int
    ngram_pad: int

    _exact_data: dict[tuple[int, int], ExactBucket] = field(default_factory=dict, repr=False)
    _context_tokens: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict, repr=False)

    _tags: array = field(default_factory=lambda: array("Q"), repr=False)
    _keys: array = field(default_factory=lambda: array("I"), repr=False)
    _values: array = field(default_factory=lambda: array("f"), repr=False)
    _counts: array = field(default_factory=lambda: array("I"), repr=False)
    _compressed: bool = field(default=False, repr=False)

    @classmethod
    def init(cls, hashtable_size: int, ngram_k: int, ngram_n: int, ngram_pad: int = 2**32 - 1) -> Self:
        return cls(
            hashtable_size=hashtable_size,
            ngram_k=ngram_k,
            ngram_n=ngram_n,
            ngram_pad=ngram_pad,
        )

    def _context_for(self, seq: Iterable[int]) -> list[int]:
        seq = list(seq)
        ngram_ctx = self.ngram_n - 1
        if ngram_ctx > 0:
            return [*repeat(self.ngram_pad, max(ngram_ctx - len(seq), 0)), *seq[-ngram_ctx:]]
        return []

    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        ngram_ctx = self.ngram_n - 1
        if ngram_ctx > 0:
            contexts = padded_sliding_window(token_ids, ngram_ctx, self.ngram_pad)
        else:
            contexts = repeat(())

        for ctx, cur_logits in zip(contexts, token_logits, strict=False):
            ctx_tuple = tuple(ctx)
            idx, tag = seqhash(ctx_tuple, self.hashtable_size)
            key = (idx, tag)
            if key not in self._context_tokens:
                self._context_tokens[key] = ctx_tuple
            bucket = self._exact_data.get(key)

            if bucket is None:
                bucket = ExactBucket()
                self._exact_data[key] = bucket

            bucket.count += 1
            sample = dict(zip(cur_logits.keys(), softmax(cur_logits.values()), strict=True))
            bucket.probs = mean_update(bucket.probs, sample, bucket.count, self.ngram_k)

    def compress(self) -> None:
        tags = array("Q", repeat(0, self.hashtable_size))
        keys = array("I", repeat(0, self.hashtable_size * self.ngram_k))
        values = array("f", repeat(0.0, self.hashtable_size * self.ngram_k))
        bucket_counts: array = array("I", repeat(0, self.hashtable_size))

        for (idx, tag), bucket in self._exact_data.items():
            if bucket.count > bucket_counts[idx]:
                bucket_counts[idx] = bucket.count
                tags[idx] = tag
                _write_top_k(bucket.probs, keys, values, idx, self.ngram_k)

        self._tags[:] = tags
        self._keys[:] = keys
        self._values[:] = values
        self._counts[:] = bucket_counts
        self._compressed = True

    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        assert self._compressed, "call compress() before probs()"
        ctx = self._context_for(seq)
        idx, tag = seqhash(ctx, self.hashtable_size)

        if self._tags[idx] != tag:
            return {}

        k_start = idx * self.ngram_k
        k_end = k_start + self.ngram_k
        return {k: v for k, v in zip(self._keys[k_start:k_end], self._values[k_start:k_end], strict=True) if v > 0.0}

    def exact_dist_for(self, seq: Iterable[int]) -> dict[int, float] | None:
        key = seqhash(self._context_for(seq), self.hashtable_size)
        bucket = self._exact_data.get(key)
        return dict(bucket.probs) if bucket is not None else None

    def apply_smoothing(
        self,
        discount: float,
        lower_order_fn: Callable[[tuple[int, ...]], dict[int, float]],
    ) -> None:
        for (idx, tag), bucket in self._exact_data.items():
            if self._tags[idx] != tag:
                continue
            ctx_tokens = self._context_tokens.get((idx, tag))
            if ctx_tokens is None:
                continue
            lower_dist = lower_order_fn(ctx_tokens)
            smoothed = _discount_and_interpolate(bucket.probs, lower_dist, discount)
            _write_top_k(smoothed, self._keys, self._values, idx, self.ngram_k)

    def serialize(self) -> bytes:
        assert self._compressed, "call compress() before serialize()"
        hdr = struct.pack("<4I", self.hashtable_size, self.ngram_k, self.ngram_n, self.ngram_pad)
        return hdr + bytes(self._tags) + bytes(self._keys) + bytes(self._values) + bytes(self._counts)

    @classmethod
    def deserialize(cls, blob: bytes) -> Self:
        offset = 16
        hashtable_size, ngram_k, ngram_n, ngram_pad = struct.unpack("<4I", blob[:offset])

        tag_len = 8 * hashtable_size
        tags = array("Q", blob[offset : offset + tag_len])
        offset += tag_len

        kv_len = 4 * ngram_k * hashtable_size
        keys = array("I", blob[offset : offset + kv_len])
        offset += kv_len

        values = array("f", blob[offset : offset + kv_len])
        offset += kv_len

        count_len = 4 * hashtable_size
        counts = array("I", blob[offset : offset + count_len])
        offset += count_len

        instance = cls(
            hashtable_size=hashtable_size,
            ngram_k=ngram_k,
            ngram_n=ngram_n,
            ngram_pad=ngram_pad,
        )
        instance._tags[:] = tags
        instance._keys[:] = keys
        instance._values[:] = values
        instance._counts[:] = counts
        instance._compressed = True
        return instance


def _deserialize_tagged_tables(blob: bytes, offset: int, max_order: int) -> tuple[list[TaggedNGramTable], int]:
    tables: list[TaggedNGramTable] = []
    for _ in range(max_order):
        (length,) = struct.unpack("<Q", blob[offset : offset + 8])
        offset += 8
        tables.append(TaggedNGramTable.deserialize(blob[offset : offset + length]))
        offset += length
    return tables, offset


@dataclass(frozen=True, eq=False)
class NGramModel:
    max_order: int
    tables: tuple[TaggedNGramTable, ...]
    discount: float = 0.002

    @classmethod
    def init(cls, hashtable_size: int, ngram_k: int, max_order: int = 4, discount: float = 0.002) -> Self:
        tables = tuple(TaggedNGramTable.init(hashtable_size, ngram_k, n) for n in range(1, max_order + 1))
        return cls(max_order, tables, discount)

    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        ids = list(token_ids)
        logits = list(token_logits)
        for table in self.tables:
            table.train(ids, logits)

    def _get_best_lower_order_dist(self, ctx_tokens: tuple[int, ...], max_lower_order: int) -> dict[int, float]:
        for order in range(max_lower_order, 0, -1):
            table = self.tables[order - 1]
            shorter_ctx = ctx_tokens[-(order - 1) :] if order > 1 else ()
            dist = table.exact_dist_for(shorter_ctx)
            if dist is not None:
                return dist
        return {}

    def compress(self) -> None:
        for table in self.tables:
            table.compress()

        for order_idx in range(1, len(self.tables)):
            table = self.tables[order_idx]
            table.apply_smoothing(
                self.discount,
                lambda ctx, oi=order_idx: self._get_best_lower_order_dist(ctx, oi),
            )

    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        seq = list(seq)
        for table in reversed(self.tables):
            result = table.probs(seq)
            if result:
                return result
        return {}

    def serialize(self) -> bytes:
        parts = [struct.pack("<If", self.max_order, self.discount)]
        for table in self.tables:
            table_bytes = table.serialize()
            parts.append(struct.pack("<Q", len(table_bytes)))
            parts.append(table_bytes)
        return b"".join(parts)

    @classmethod
    def deserialize(cls, blob: bytes) -> Self:
        offset = 0
        max_order, discount = struct.unpack("<If", blob[offset : offset + 8])
        offset += 8
        tables, _ = _deserialize_tagged_tables(blob, offset, max_order)
        return cls(max_order, tuple(tables), discount)


@dataclass(frozen=True, kw_only=True)
class NGramSpeculator(TreeSpeculator):
    name: ClassVar[str] = "ngram"

    model: NGramModel
    width: int = 4
    depth: int = 8
    context: tuple[int, ...] = ()

    def draft(self, lm: LMState) -> TrieNode:
        gseed = self.seed
        root = TrieNode(token=lm.bonus, seed=gseed.derive(1).value)

        (vocab_size,) = lm.logits.shape
        ctx = [*self.context, lm.bonus]
        node_offset = 2
        chain_node = root
        for _k in range(self.depth):
            probs = self.model.probs(ctx)
            if not probs:
                break
            parent_seed = GumbelSeed(chain_node.seed)
            next_candidates = gumbel_rank_from_probs(parent_seed, probs, vocab_size, self.width)
            for i, tok in enumerate(next_candidates):
                chain_node.add_child(tok, seed=gseed.derive(node_offset + i).value)
            node_offset += self.width
            chain_node = chain_node.get_child(next_candidates[0])
            ctx.append(next_candidates[0])

        return root

    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]:
        _, lm_state = super().prefill(prompt_ids)
        max_ctx = self.model.max_order - 1
        context = tuple(prompt_ids[-max_ctx:]) if max_ctx > 0 else ()
        return dataclasses.replace(self, context=context), lm_state

    def build_next_state(
        self,
        lm: LMState,
        fwd: DecoderResult,
        result: TreeVerifyResult,
    ) -> tuple[Self, LMState]:
        parent_self, new_lm = super().build_next_state(lm, fwd, result)
        new_context = self.context + (lm.bonus,) + tuple(result.accepted_tokens)
        max_ctx = self.model.max_order - 1
        trimmed = new_context[-max_ctx:] if max_ctx > 0 else ()
        final = dataclasses.replace(self, seed=parent_self.seed, context=trimmed)
        return final, new_lm

    def serialize(self) -> bytes:
        return self.model.serialize()

    @classmethod
    def deserialize_impl(
        cls,
        data: bytes,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        width: int = 4,
        depth: int = 8,
        **_: object,
    ) -> Self:
        return cls(
            decoder=decoder,
            config=config,
            eos_set=eos_set,
            seed=GumbelSeed(config.seed),
            model=NGramModel.deserialize(data),
            width=width,
            depth=depth,
        )

    def sample(self, context: Iterable[int] = (), max_length: int = 32) -> list[int]:
        seq = list(context)
        for _ in range(max_length):
            probs = self.model.probs(seq)
            if not probs or sum(probs.values()) == 0:
                break
            seq.append(random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0])
        return seq

    @staticmethod
    def train_command(app: Typer, callbacks_type: type) -> None:
        @app.command(name="train-ngram", help="Train an n-gram speculator from inference traces")
        def train_ngram_cmd(
            trace_path: Annotated[Path, Argument(help="Trace directory", metavar="TRACE_PATH")],
            output_path: Annotated[Path, Option(help="Output file for trained speculator")],
            hashtable_size: Annotated[int, Option(help="Size of ngram hashtable")] = 65536,
            num_logits_per_token: Annotated[int, Option(help="Top K tokens per ngram bucket")] = 8,
            max_order: Annotated[int, Option(help="Maximum n-gram order")] = 4,
            discount: Annotated[float, Option(help="Kneser-Ney discount")] = 0.002,
            subsample_size: Annotated[int | None, Option(help="Stop after this many tokens")] = None,
        ) -> None:
            callbacks = callbacks_type(output_path=output_path, subsample_size=subsample_size)
            traces = iter_completions(trace_path, exclude={"activation_output", "layer_output", "logsumexp"})
            model = train_ngram(
                traces,
                hashtable_size=hashtable_size,
                num_logits_per_token=num_logits_per_token,
                max_order=max_order,
                discount=discount,
                tokens_to_train=subsample_size,
                progress_callback=lambda e: callbacks.training_progress(e.trained_tokens),
            )
            callbacks.finished_training()
            callbacks.save_drafter(model.serialize())


@dataclass(frozen=True)
class NGramTrainingEvent:
    trained_sequences: int
    trained_tokens: int


def train_ngram(
    traces: Iterable[LalamoCompletion],
    hashtable_size: int = 65536,
    num_logits_per_token: int = 8,
    max_order: int = 4,
    discount: float = 0.002,
    tokens_to_train: int | None = None,
    progress_callback: Callable[[NGramTrainingEvent], None] | None = None,
) -> NGramModel:
    model = NGramModel.init(hashtable_size, num_logits_per_token, max_order, discount)
    trained_tokens = 0

    for trained_sequences, trace in enumerate(traces, start=1):
        if tokens_to_train is not None and trained_tokens + len(trace.completion_token_ids) > tokens_to_train:
            end = tokens_to_train - trained_tokens
        else:
            end = None
        token_ids = trace.completion_token_ids[:end]
        token_logits = trace.completion_token_logits[:end]

        model.train(token_ids, token_logits)
        trained_tokens += len(token_ids)

        if progress_callback is not None:
            progress_callback(NGramTrainingEvent(trained_sequences, trained_tokens))

        if tokens_to_train is not None and trained_tokens >= tokens_to_train:
            break

    model.compress()
    return model
