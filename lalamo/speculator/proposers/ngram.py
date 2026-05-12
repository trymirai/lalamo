from __future__ import annotations

import struct
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain, repeat, tee
from math import exp
from pathlib import Path
from typing import Annotated, Any, Self

import jax.numpy as jnp
import xxhash
from typer import Option

from lalamo.data.completion_features import FeatureRequest, LalamoCompletionFeatures
from lalamo.data.lalamo_completions import LalamoCompletion  # noqa: TC001
from lalamo.modules.decoder import Decoder  # noqa: TC001
from lalamo.speculator.common import (
    Speculator,
    SpeculatorBackend,
    SpeculatorState,
    write_speculator_artifact,
)
from lalamo.speculator.proposal import TrieProposal  # noqa: TC001
from lalamo.speculator.state import LMState, StateRequest
from lalamo.speculator.training import (
    SpeculatorBatchResult,
    SpeculatorTrainer,
    SpeculatorTrainingConfig,
    SpeculatorTrainingContext,
    SpeculatorTrainingEvent,
)

__all__ = [
    "NGramBackend",
    "NGramConfig",
    "NGramModel",
    "NGramSpeculator",
    "NGramTrainer",
    "NGramTrainingState",
    "TaggedNGramTable",
]


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


def update_probs(probs: dict[int, float], sample: dict[int, float], new_count: int, top_k: int) -> None:
    decay = (new_count - 1) / new_count
    inv_n = 1.0 / new_count
    for token_id in probs:
        probs[token_id] *= decay
    for token_id, probability in sample.items():
        probs[token_id] = probs.get(token_id, 0.0) + probability * inv_n
    if len(probs) > top_k:
        keep = dict(sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:top_k])
        probs.clear()
        probs.update(keep)
    total = sum(probs.values())
    if total > 0:
        inv_total = 1.0 / total
        for token_id in probs:
            probs[token_id] *= inv_total


def seqhash(tokens: Iterable[int], size: int) -> tuple[int, int]:
    tokens = list(tokens)
    packed = struct.pack("<" + "I" * len(tokens), *tokens) if tokens else b""
    digest = xxhash.xxh3_64_intdigest(packed)
    return digest % size, digest // size


def discount_and_interpolate(
    dist: dict[int, float],
    lower_dist: dict[int, float],
    discount: float,
) -> dict[int, float]:
    num_nonzero = sum(1 for value in dist.values() if value > 0)
    freed_mass = min(discount * num_nonzero, 0.5)
    smoothed = {token_id: max(probability - discount, 0.0) for token_id, probability in dist.items()}
    for token_id, probability in lower_dist.items():
        smoothed[token_id] = smoothed.get(token_id, 0.0) + freed_mass * probability
    return smoothed


def write_top_k(dist: dict[int, float], keys: array, values: array, index: int, ngram_k: int) -> None:
    top = sorted(dist.items(), key=lambda item: (-item[1], item[0]))[:ngram_k]
    total = sum(probability for _, probability in top)
    if total <= 0:
        return

    new_keys = array("I", repeat(0, ngram_k))
    new_values = array("f", repeat(0.0, ngram_k))
    for slot, (token_id, probability) in enumerate(top):
        new_keys[slot] = token_id
        new_values[slot] = probability / total

    start = index * ngram_k
    memoryview(keys)[start : start + ngram_k] = new_keys
    memoryview(values)[start : start + ngram_k] = new_values


class ExactBucket:
    __slots__ = ("count", "probs")

    def __init__(self) -> None:
        self.count: int = 0
        self.probs: dict[int, float] = {}


@dataclass(frozen=True, eq=False)
class TaggedNGramTable:
    hashtable_size: int
    ngram_k: int
    ngram_n: int
    ngram_pad: int
    exact_data: dict[tuple[int, int], ExactBucket] = field(default_factory=dict, repr=False)
    context_tokens: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict, repr=False)
    tags: array = field(default_factory=lambda: array("Q"), repr=False)
    keys: array = field(default_factory=lambda: array("I"), repr=False)
    values: array = field(default_factory=lambda: array("f"), repr=False)
    counts: array = field(default_factory=lambda: array("I"), repr=False)
    compressed: list[bool] = field(default_factory=lambda: [False], repr=False)

    @classmethod
    def init(cls, hashtable_size: int, ngram_k: int, ngram_n: int, ngram_pad: int = 2**32 - 1) -> Self:
        return cls(
            hashtable_size=hashtable_size,
            ngram_k=ngram_k,
            ngram_n=ngram_n,
            ngram_pad=ngram_pad,
        )

    def context_for(self, seq: Iterable[int]) -> list[int]:
        seq = list(seq)
        context_length = self.ngram_n - 1
        if context_length > 0:
            return [*repeat(self.ngram_pad, max(context_length - len(seq), 0)), *seq[-context_length:]]
        return []

    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        context_length = self.ngram_n - 1
        contexts = (
            padded_sliding_window(token_ids, context_length, self.ngram_pad) if context_length > 0 else repeat(())
        )

        for context, current_logits in zip(contexts, token_logits, strict=False):
            context_tuple = tuple(context)
            index, tag = seqhash(context_tuple, self.hashtable_size)
            key = (index, tag)
            if key not in self.context_tokens:
                self.context_tokens[key] = context_tuple
            bucket = self.exact_data.get(key)

            if bucket is None:
                bucket = ExactBucket()
                self.exact_data[key] = bucket

            bucket.count += 1
            sample = dict(zip(current_logits.keys(), softmax(current_logits.values()), strict=True))
            update_probs(bucket.probs, sample, bucket.count, self.ngram_k)

    def compress(self) -> None:
        tags = array("Q", repeat(0, self.hashtable_size))
        keys = array("I", repeat(0, self.hashtable_size * self.ngram_k))
        values = array("f", repeat(0.0, self.hashtable_size * self.ngram_k))
        counts = array("I", repeat(0, self.hashtable_size))

        for (index, tag), bucket in self.exact_data.items():
            if bucket.count > counts[index]:
                counts[index] = bucket.count
                tags[index] = tag
                write_top_k(bucket.probs, keys, values, index, self.ngram_k)

        self.tags[:] = tags
        self.keys[:] = keys
        self.values[:] = values
        self.counts[:] = counts
        self.compressed[0] = True

    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        assert self.compressed[0], "call compress() before probs()"
        context = self.context_for(seq)
        index, tag = seqhash(context, self.hashtable_size)

        if self.tags[index] != tag:
            return {}

        start = index * self.ngram_k
        end = start + self.ngram_k
        return {
            token_id: probability
            for token_id, probability in zip(self.keys[start:end], self.values[start:end], strict=True)
            if probability > 0.0
        }

    def exact_dist_for(self, seq: Iterable[int]) -> dict[int, float] | None:
        key = seqhash(self.context_for(seq), self.hashtable_size)
        bucket = self.exact_data.get(key)
        return dict(bucket.probs) if bucket is not None else None

    def apply_smoothing(
        self,
        discount: float,
        lower_order_fn: Callable[[tuple[int, ...]], dict[int, float]],
    ) -> None:
        for (index, tag), bucket in self.exact_data.items():
            if self.tags[index] != tag:
                continue
            context = self.context_tokens.get((index, tag))
            if context is None:
                continue
            smoothed = discount_and_interpolate(bucket.probs, lower_order_fn(context), discount)
            write_top_k(smoothed, self.keys, self.values, index, self.ngram_k)

    def serialize(self) -> bytes:
        assert self.compressed[0], "call compress() before serialize()"
        header = struct.pack("<4I", self.hashtable_size, self.ngram_k, self.ngram_n, self.ngram_pad)
        return header + bytes(self.tags) + bytes(self.keys) + bytes(self.values) + bytes(self.counts)

    @classmethod
    def deserialize(cls, blob: bytes) -> Self:
        offset = 16
        hashtable_size, ngram_k, ngram_n, ngram_pad = struct.unpack("<4I", blob[:offset])

        tag_len = 8 * hashtable_size
        tags = array("Q", blob[offset : offset + tag_len])
        offset += tag_len

        key_value_len = 4 * ngram_k * hashtable_size
        keys = array("I", blob[offset : offset + key_value_len])
        offset += key_value_len

        values = array("f", blob[offset : offset + key_value_len])
        offset += key_value_len

        count_len = 4 * hashtable_size
        counts = array("I", blob[offset : offset + count_len])

        instance = cls(
            hashtable_size=hashtable_size,
            ngram_k=ngram_k,
            ngram_n=ngram_n,
            ngram_pad=ngram_pad,
        )
        instance.tags[:] = tags
        instance.keys[:] = keys
        instance.values[:] = values
        instance.counts[:] = counts
        instance.compressed[0] = True
        return instance


@dataclass(frozen=True, eq=False)
class NGramModel:
    max_order: int
    tables: tuple[TaggedNGramTable, ...]
    discount: float = 0.002

    @classmethod
    def init(cls, hashtable_size: int, ngram_k: int, max_order: int = 4, discount: float = 0.002) -> Self:
        tables = tuple(TaggedNGramTable.init(hashtable_size, ngram_k, order) for order in range(1, max_order + 1))
        return cls(max_order=max_order, tables=tables, discount=discount)

    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        ids = list(token_ids)
        logits = list(token_logits)
        for table in self.tables:
            table.train(ids, logits)

    def best_lower_order_dist(self, context_tokens: tuple[int, ...], max_lower_order: int) -> dict[int, float]:
        for order in range(max_lower_order, 0, -1):
            table = self.tables[order - 1]
            shorter_context = context_tokens[-(order - 1) :] if order > 1 else ()
            dist = table.exact_dist_for(shorter_context)
            if dist is not None:
                return dist
        return {}

    def compress(self) -> None:
        for table in self.tables:
            table.compress()

        for order_index in range(1, len(self.tables)):
            table = self.tables[order_index]
            table.apply_smoothing(
                self.discount,
                lambda context, oi=order_index: self.best_lower_order_dist(context, oi),
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
        tables: list[TaggedNGramTable] = []
        for _ in range(max_order):
            (length,) = struct.unpack("<Q", blob[offset : offset + 8])
            offset += 8
            tables.append(TaggedNGramTable.deserialize(blob[offset : offset + length]))
            offset += length
        return cls(max_order=max_order, tables=tuple(tables), discount=discount)


@dataclass(frozen=True, kw_only=True, eq=False)
class NGramSpeculator(Speculator):
    model: NGramModel
    width: int = 4
    depth: int = 8

    @classmethod
    def create(
        cls,
        model: NGramModel,
        width: int = 4,
        depth: int = 8,
    ) -> Self:
        return cls(
            model=model,
            width=width,
            depth=depth,
        )

    @property
    def state_request(self) -> StateRequest:
        return StateRequest(token_id_capacity=max(self.model.max_order - 1, 0))

    def draft(self, state: LMState, speculator_state: SpeculatorState) -> TrieProposal:
        del speculator_state
        proposal = state.create_root_proposal(budget=self.width * self.depth + 1)
        parent_indices = [0 for _ in range(state.root_bonus_id.shape[0])]
        contexts = self.contexts(state)

        for _ in range(self.depth):
            candidates = [self.rank_candidates(context) for context in contexts]
            if all(not row_candidates for row_candidates in candidates):
                break

            for rank in range(self.width):
                batch_indices = [
                    row_index for row_index, row_candidates in enumerate(candidates) if rank < len(row_candidates)
                ]
                if not batch_indices:
                    continue

                node_token_ids = [candidates[row_index][rank] for row_index in batch_indices]
                proposal, node_index = proposal.add_nodes(
                    batch_indices=jnp.asarray(batch_indices, dtype=jnp.int32),
                    parent_indices=jnp.asarray(
                        [parent_indices[row_index] for row_index in batch_indices], dtype=jnp.int32
                    ),
                    token_ids=jnp.asarray(node_token_ids, dtype=jnp.int32),
                )

                if rank == 0:
                    for row_index, token_id in zip(batch_indices, node_token_ids, strict=True):
                        parent_indices[row_index] = node_index
                        contexts[row_index].append(token_id)

        return proposal

    def contexts(self, state: LMState) -> list[list[int]]:
        root_bonus_ids = [int(token_id) for token_id in state.root_bonus_id.tolist()]
        context_length = max(self.model.max_order - 1, 0)
        if context_length == 0:
            return [[root_bonus_id] for root_bonus_id in root_bonus_ids]

        token_ids, token_mask = state.recent_token_ids(context_length)
        return [
            [int(token_id) for token_id, valid in zip(row_token_ids, row_mask, strict=True) if bool(valid)]
            + [root_bonus_id]
            for row_token_ids, row_mask, root_bonus_id in zip(
                token_ids.tolist(), token_mask.tolist(), root_bonus_ids, strict=True
            )
        ]

    def rank_candidates(self, context: Iterable[int]) -> tuple[int, ...]:
        probs = self.model.probs(context)
        return tuple(
            token_id for token_id, _ in sorted(probs.items(), key=lambda item: (-item[1], item[0]))[: self.width]
        )

    def serialize(self) -> bytes:
        return self.model.serialize()

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        width: int = 4,
        depth: int = 8,
    ) -> Self:
        return cls.create(
            model=NGramModel.deserialize(data),
            width=width,
            depth=depth,
        )


@dataclass(frozen=True)
class NGramConfig:
    hash_table_size: Annotated[
        int,
        Option("--hash_table_size", help="Number of buckets in each n-gram table."),
    ] = 65536
    num_logits_per_token: Annotated[
        int,
        Option("--num_logits_per_token", help="Number of target-logit candidates retained per trained token."),
    ] = 8
    max_order: Annotated[int, Option("--max_order", help="Maximum n-gram order used for backoff.")] = 4
    discount: Annotated[
        float,
        Option("--discount", help="Absolute discount used for lower-order interpolation."),
    ] = 0.002
    width: Annotated[int, Option("--width", help="Number of candidate tokens proposed at each draft step.")] = 4
    depth: Annotated[int, Option("--depth", help="Number of draft steps proposed from the root token.")] = 8


@dataclass(frozen=True)
class NGramTrainingState:
    model: NGramModel


@dataclass(frozen=True, kw_only=True)
class NGramTrainer(SpeculatorTrainer[NGramSpeculator, NGramTrainingState]):
    artifact_path: Path | str
    config: NGramConfig

    def make_feature_request(
        self,
        completions: tuple[LalamoCompletion, ...],
        config: SpeculatorTrainingConfig,
    ) -> FeatureRequest:
        return FeatureRequest(
            completions=completions,
            top_k_logits=config.top_k_logits,
        )

    def init_state(self) -> NGramTrainingState:
        return NGramTrainingState(
            model=NGramModel.init(
                hashtable_size=self.config.hash_table_size,
                ngram_k=self.config.num_logits_per_token,
                max_order=self.config.max_order,
                discount=self.config.discount,
            ),
        )

    def train(
        self,
        state: NGramTrainingState,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> tuple[NGramTrainingState, SpeculatorBatchResult]:
        del context
        for token_ids, token_logits in iter_ngram_feature_rows(features):
            state.model.train(token_ids, token_logits)
        return state, SpeculatorBatchResult()

    def evaluate(
        self,
        state: NGramTrainingState,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> SpeculatorBatchResult:
        del state, features, context
        return SpeculatorBatchResult()

    def finish(
        self,
        state: NGramTrainingState,
    ) -> NGramSpeculator:
        state.model.compress()
        return NGramSpeculator.create(
            model=state.model,
            width=self.config.width,
            depth=self.config.depth,
        )

    def save(
        self,
        state: NGramTrainingState,
        event: SpeculatorTrainingEvent,
    ) -> None:
        del event
        speculator = self.finish(state)
        write_speculator_artifact(
            self.artifact_path,
            NGramBackend,
            self.config.width,
            self.config.depth,
            speculator.serialize(),
        )


class NGramBackend(SpeculatorBackend[NGramConfig]):
    name = "ngram"
    config_type = NGramConfig

    @classmethod
    def create_trainer(
        cls,
        config: NGramConfig,
        artifact_path: Path,
        target_model: Decoder,
    ) -> NGramTrainer:
        del target_model
        return NGramTrainer(
            artifact_path=artifact_path,
            config=config,
        )

    @classmethod
    def deserialize(cls, fields: tuple[Any, ...], target_model: Decoder) -> NGramSpeculator:
        del target_model
        if len(fields) != 3:
            raise ValueError("ngram speculator artifact must contain width, depth, and model bytes.")
        width, depth, data = fields
        if not isinstance(width, int) or not isinstance(depth, int) or not isinstance(data, bytes):
            raise TypeError("ngram speculator artifact fields must be int, int, bytes.")
        return NGramSpeculator.deserialize(
            data,
            width=width,
            depth=depth,
        )


def iter_ngram_feature_rows(
    features: LalamoCompletionFeatures,
) -> Iterable[tuple[list[int], list[dict[int, float]]]]:
    target_top_k_ids = features.target_top_k_ids
    target_top_k_logits = features.target_top_k_logits
    assert target_top_k_ids is not None
    assert target_top_k_logits is not None

    for token_id_row, mask_row, top_id_row, top_logit_row in zip(
        features.completion_batch.target_token_ids.tolist(),
        features.completion_batch.target_mask.tolist(),
        target_top_k_ids.tolist(),
        target_top_k_logits.tolist(),
        strict=True,
    ):
        token_ids: list[int] = []
        token_logits: list[dict[int, float]] = []
        for token_id, valid, top_ids, top_logits in zip(
            token_id_row,
            mask_row,
            top_id_row,
            top_logit_row,
            strict=True,
        ):
            if not valid:
                continue
            token_ids.append(int(token_id))
            token_logits.append(
                {
                    int(candidate_id): float(candidate_logit)
                    for candidate_id, candidate_logit in zip(top_ids, top_logits, strict=True)
                },
            )
        yield token_ids, token_logits
