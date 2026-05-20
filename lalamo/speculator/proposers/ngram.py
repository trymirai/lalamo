# ruff: noqa: ANN401, TC002
from __future__ import annotations

import struct
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain, repeat, tee
from math import exp, log
from pathlib import Path
from typing import Annotated, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xxhash
from jaxtyping import Array, Bool, Float, Int, Key
from typer import Option

from lalamo.data.completion_features import FeatureRequest, LalamoCompletionFeatures
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.modules.decoder import Decoder
from lalamo.modules.token_mixer import State
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.common import Speculator, SpeculatorBackend, write_speculator_artifact
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal
from lalamo.speculator.state import LMState, PrefillResults
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


def ngram_nodes_from_context(
    model: NGramModel,
    recent_token_ids: Int[Array, "batch context"],
    recent_token_mask: Bool[Array, "batch context"],
    root_bonus_ids: Int[Array, " batch"],
    width: int,
    depth: int,
    vocabulary_size: int,
) -> tuple[
    Int[Array, "batch nodes"],
    Int[Array, "batch nodes"],
    Int[Array, "batch nodes"],
    Bool[Array, "batch nodes"],
]:
    batch_size = root_bonus_ids.shape[0]
    node_budget = width * depth
    output_shape = jax.ShapeDtypeStruct((batch_size, node_budget), jnp.int32)
    mask_shape = jax.ShapeDtypeStruct((batch_size, node_budget), jnp.bool)
    return jax.pure_callback(
        lambda token_ids, token_mask, bonus_ids: ngram_nodes_from_context_numpy(
            model,
            token_ids,
            token_mask,
            bonus_ids,
            width,
            depth,
            vocabulary_size,
        ),
        (output_shape, output_shape, output_shape, mask_shape),
        recent_token_ids,
        recent_token_mask,
        root_bonus_ids,
        vmap_method="sequential",
    )


def ngram_nodes_from_context_numpy(
    model: NGramModel,
    recent_token_ids: Any,
    recent_token_mask: Any,
    root_bonus_ids: Any,
    width: int,
    depth: int,
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recent_token_ids = np.asarray(recent_token_ids)
    recent_token_mask = np.asarray(recent_token_mask)
    root_bonus_ids = np.asarray(root_bonus_ids)
    batch_size = root_bonus_ids.shape[0]
    node_budget = width * depth
    token_ids = np.zeros((batch_size, node_budget), dtype=np.int32)
    parent_indices = np.full((batch_size, node_budget), -1, dtype=np.int32)
    depths = np.zeros((batch_size, node_budget), dtype=np.int32)
    node_mask = np.zeros((batch_size, node_budget), dtype=np.bool_)
    for batch_index in range(batch_size):
        context = [
            int(token_id)
            for token_id, valid in zip(
                recent_token_ids[batch_index].tolist(),
                recent_token_mask[batch_index].tolist(),
                strict=True,
            )
            if bool(valid)
        ]
        context.append(int(root_bonus_ids[batch_index]))
        row_token_ids, row_parent_indices, row_depths, row_mask = ngram_nodes_for_row_numpy(
            model,
            context,
            width,
            depth,
            vocabulary_size,
        )
        token_ids[batch_index] = row_token_ids
        parent_indices[batch_index] = row_parent_indices
        depths[batch_index] = row_depths
        node_mask[batch_index] = row_mask
    return token_ids, parent_indices, depths, node_mask


def ngram_nodes_for_row_numpy(
    model: NGramModel,
    root_context: list[int],
    width: int,
    depth: int,
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_budget = width * depth
    token_ids = np.zeros((node_budget,), dtype=np.int32)
    parent_indices = np.full((node_budget,), -1, dtype=np.int32)
    node_depths = np.zeros((node_budget,), dtype=np.int32)
    node_mask = np.zeros((node_budget,), dtype=np.bool_)
    frontier: list[tuple[int, list[int]]] = [(0, root_context)]
    node_count = 0
    for depth_index in range(1, depth + 1):
        next_frontier: list[tuple[int, list[int]]] = []
        for parent_index, context in frontier:
            if node_count >= node_budget:
                break
            probs = model.probs(context)
            candidates = sorted(
                (
                    (int(token_id), float(probability))
                    for token_id, probability in probs.items()
                    if int(token_id) < vocabulary_size and probability > 0.0
                ),
                key=lambda item: (-item[1], item[0]),
            )[:width]
            for token_id, _probability in candidates:
                if node_count >= node_budget:
                    break
                node_index = node_count + 1
                token_ids[node_count] = np.int32(token_id)
                parent_indices[node_count] = np.int32(parent_index)
                node_depths[node_count] = np.int32(depth_index)
                node_mask[node_count] = True
                next_context = [*context, token_id]
                context_length = max(model.max_order - 1, 0)
                if context_length > 0:
                    next_context = next_context[-context_length:]
                next_frontier.append((node_index, next_context))
                node_count += 1
        if not next_frontier or node_count >= node_budget:
            break
        frontier = next_frontier
    return token_ids, parent_indices, node_depths, node_mask


class NGramLMState(LMState):
    context_token_ids: Int[Array, "batch capacity"]
    context_lengths: Int[Array, " batch"]

    @classmethod
    def from_base(
        cls,
        state: LMState,
        prefill_results: PrefillResults,
    ) -> NGramLMState:
        return cls(
            kv_cache=state.kv_cache,
            next_token_position=state.next_token_position,
            root_bonus_id=state.root_bonus_id,
            root_sample_logits=state.root_sample_logits,
            sampling_policy=state.sampling_policy,
            gumbel_keys=state.gumbel_keys,
            output_lengths=state.output_lengths,
            stop_flags=state.stop_flags,
            context_token_ids=prefill_results.input_token_ids,
            context_lengths=prefill_results.input_lengths,
        )

    def update_status(
        self,
        kv_cache: State,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        accepted: AcceptedProposal,
        accepted_token_ids: Int[Array, "batch max_slots"],
        accepted_output_features: Float[Array, "batch max_slots channels"] | None,
        accepted_layer_outputs: tuple[Float[Array, "batch max_slots channels"], ...],
        stop_flags: Bool[Array, " batch"],
    ) -> NGramLMState:
        next_state = super().update_status(
            kv_cache,
            processed_tree_logits,
            accepted,
            accepted_token_ids,
            accepted_output_features,
            accepted_layer_outputs,
            stop_flags,
        )
        batch_size, capacity = self.context_token_ids.shape
        slots = jnp.arange(accepted_token_ids.shape[1], dtype=jnp.int32)[None, :]
        positions = self.context_lengths[:, None] + slots
        valid = slots < accepted.num_compact_indices[:, None]
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        context_token_ids = self.context_token_ids.at[batch_indices, positions].set(
            jnp.where(valid, accepted_token_ids, 0),
            mode="drop",
        )
        context_lengths = jnp.minimum(self.context_lengths + accepted.num_compact_indices, capacity)
        return eqx.tree_at(
            lambda state: (state.context_token_ids, state.context_lengths),
            next_state,
            (context_token_ids, context_lengths),
        )


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
    def max_step_tokens(self) -> int:
        return self.depth + 1

    def init_state(
        self,
        prefill_results: PrefillResults,
        next_token_position: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
    ) -> NGramLMState:
        state = LMState.from_prefill(
            prefill_results,
            next_token_position,
            sampling_policy,
            gumbel_keys,
        )
        return NGramLMState.from_base(state, prefill_results)

    def draft(self, state: LMState) -> TrieProposal:
        if not isinstance(state, NGramLMState):
            raise TypeError(f"NGram requires NGramLMState, got {type(state).__name__}")
        node_budget = self.width * self.depth
        proposal = state.create_root_proposal(budget=node_budget + 1)
        slots = jnp.arange(state.context_token_ids.shape[1], dtype=jnp.int32)[None, :]
        context_token_mask = slots < state.context_lengths[:, None]
        token_ids, parent_indices, depths, node_mask = ngram_nodes_from_context(
            self.model,
            state.context_token_ids,
            context_token_mask,
            state.root_bonus_id,
            self.width,
            self.depth,
            self.vocab_size(state),
        )
        return proposal.add_nodes(
            token_ids,
            parent_indices,
            depths,
            node_mask,
            self.depth,
        )

    def contexts(self, state: LMState) -> list[list[int]]:
        if not isinstance(state, NGramLMState):
            raise TypeError(f"NGram requires NGramLMState, got {type(state).__name__}")
        root_bonus_ids = [int(token_id) for token_id in state.root_bonus_id.tolist()]
        context_length = max(self.model.max_order - 1, 0)
        if context_length == 0:
            return [[root_bonus_id] for root_bonus_id in root_bonus_ids]

        token_mask = jnp.arange(state.context_token_ids.shape[1], dtype=jnp.int32)[None, :] < state.context_lengths[
            :, None
        ]
        return [
            [int(token_id) for token_id, valid in zip(row_token_ids, row_mask, strict=True) if bool(valid)]
            + [root_bonus_id]
            for row_token_ids, row_mask, root_bonus_id in zip(
                state.context_token_ids.tolist(),
                token_mask.tolist(),
                root_bonus_ids,
                strict=True,
            )
        ]

    def vocab_size(self, state: LMState) -> int:
        return state.root_sample_logits.shape[-1]

    def has_token(self, probs: dict[int, float], vocab_size: int) -> bool:
        return any(token_id < vocab_size for token_id in probs)

    def logits_for(self, rows: list[dict[int, float]], vocab_size: int) -> Float[Array, "batch vocabulary"]:
        logits = []
        for probs in rows:
            token_ids = [token_id for token_id in probs if token_id < vocab_size]
            row_logits = jnp.full((vocab_size,), -jnp.inf, dtype=jnp.float32)
            row_logits = row_logits.at[jnp.asarray(token_ids, dtype=jnp.int32)].set(
                jnp.asarray([log(probs[token_id]) for token_id in token_ids], dtype=jnp.float32),
            )
            logits.append(row_logits)
        return jnp.stack(logits)

    def logits_for_tree(
        self,
        rows: list[list[dict[int, float]]],
        vocab_size: int,
    ) -> Float[Array, "batch depth vocabulary"]:
        return jnp.stack([self.logits_for(row, vocab_size) for row in rows])

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
