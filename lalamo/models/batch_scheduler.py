from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import batched

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, Key, Shaped

from lalamo.common import get_usable_memory_from_bytes
from lalamo.message_processor import AssistantMessage, Message
from lalamo.modules import (
    ForwardPassMode,
    MLPForwardPassConfig,
    ShardingConfig,
    State,
    get_current_sharding_config,
)
from lalamo.sampling import SamplingPolicy

from .common import (
    BatchSizeInfo,
    BatchSizesComputedEvent,
    InferenceConfig,
)
from .language_model import (
    DecodingState,
    ForwardPassConfig,
    GenerationConfig,
    GenerationResults,
    GenerationTraceConfig,
    LanguageModel,
    PrefillResults,
)
from .lm_helpers import (
    bucket_by_length,
    bucket_sequences,
    decrease_batchsize_on_oom,
    merge_small_buckets,
    pad_sequences,
)


class PrefillBatch(eqx.Module):
    prefill_results: PrefillResults
    mask: Bool[Array, " prefill_bs"]
    keys: Key[Array, " prefill_bs"]
    sequence_ids: Int[Array, " prefill_bs"]

    def __check_init__(self) -> None:
        prefill_bs = self.prefill_results.last_token_indices.shape[0]
        if self.mask.shape != (prefill_bs,):
            raise ValueError(f"mask shape {self.mask.shape} != ({prefill_bs},)")
        if self.keys.shape != (prefill_bs,):
            raise ValueError(f"keys shape {self.keys.shape} != ({prefill_bs},)")
        if self.sequence_ids.shape != (prefill_bs,):
            raise ValueError(f"sequence_ids shape {self.sequence_ids.shape} != ({prefill_bs},)")
        if self.prefill_results.last_token_logits.shape[0] != prefill_bs:
            raise ValueError(
                f"results.last_token_logits batch {self.prefill_results.last_token_logits.shape[0]} != {prefill_bs}"
            )


@dataclass
class PrefillSource:
    tokenized_messages: list[list[int]]
    keys: Key[Array, " num_sequences"]
    prefill_batch_size: int
    padded_length: int
    state_capacity: int
    forward_pass_config: ForwardPassConfig
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return len(self.tokenized_messages) - self.cursor

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.tokenized_messages)

    def fill(
        self,
        decoder: "BlockContinuousDecoder",
        state: "BlockContinuousState",
        jitted_fill_lines: Callable[["BlockContinuousState", "PrefillBatch"], "BlockContinuousState"],
    ) -> tuple["PrefillSource", "BlockContinuousState"]:
        source = self
        # iterate until the decoder state is full (the state that participates in decode loop)
        while state.empty_lines():
            if source.exhausted:
                break

            # generate new partially prefilled batch - we prefill at most a small percentage of the maximum
            # batch size due to performance reasons, thus its partially prefilled (ie mostly masked)
            source, partially_prefilled_batch = source.request(len(state.empty_lines()), decoder.language_model)

            # update the state with the new samples
            state = jitted_fill_lines(state, partially_prefilled_batch)

            # Explicit deletion of batch is crucial. GC of Python doesn't collect it instantly,
            # which causes double allocation, spiking memory consumption.
            del partially_prefilled_batch
        return source, state

    def request(self, num_requested: int, model: LanguageModel) -> tuple["PrefillSource", PrefillBatch]:
        # how many samples should we provide? we cannot provide more than the prefill batch size,
        # and we cannot more provide more than the number of remaining samples
        num_designated = min(num_requested, self.prefill_batch_size, self.remaining)

        selected_indices = np.zeros(self.prefill_batch_size, dtype=np.int32)
        selected_indices[:num_designated] = np.arange(self.cursor, self.cursor + num_designated)

        sequences = self.tokenized_messages[self.cursor : self.cursor + num_designated]
        lengths_without_padding = np.zeros(self.prefill_batch_size, dtype=np.int32)
        lengths_without_padding[:num_designated] = [len(s) for s in sequences]
        padded_sequences = pad_sequences(sequences, (self.prefill_batch_size, self.padded_length), dtype=jnp.int32)

        mask = np.zeros(self.prefill_batch_size, dtype=bool)
        mask[:num_designated] = True

        prefilled_state_with_info: PrefillResults = model._prefill(  # noqa: SLF001
            token_ids=jnp.array(padded_sequences),
            lengths_without_padding=jnp.array(lengths_without_padding),
            state_capacity=self.state_capacity,
            forward_pass_config=self.forward_pass_config,
            chunk_size=128,
        )

        batch = PrefillBatch(
            prefill_results=prefilled_state_with_info,
            mask=jnp.asarray(mask),
            keys=self.keys[jnp.asarray(selected_indices)],
            sequence_ids=jnp.asarray(selected_indices),
        )
        return replace(self, cursor=self.cursor + num_designated), batch


def append_block_tokens(
    token_buffer: Int[Array, "num_lines max_output"],
    num_generated: Int[Array, " num_lines"],
    block_tokens: Int[Array, "block_size num_lines"],
) -> tuple[Int[Array, "num_lines max_output"], Int[Array, " num_lines"]]:
    num_lines, _ = token_buffer.shape
    block_size, _ = block_tokens.shape
    offsets = num_generated[:, None] + jnp.arange(block_size)
    new_buffer = token_buffer.at[jnp.arange(num_lines)[:, None], offsets].set(block_tokens.T, mode="drop")
    return new_buffer, num_generated + block_size


class BlockContinuousState(eqx.Module):
    last_token_logits: Float[Array, "num_lines vocab"]
    last_token_indices: Int[Array, " num_lines"]
    kv_state: State
    stop_flags: Bool[Array, " num_lines"]  # True = empty / just-finished; False = active
    token_buffer: Int[Array, "num_lines max_output"]
    num_generated: Int[Array, " num_lines"]
    keys: Key[Array, " num_lines"]
    in_batch_index_to_sequence_id: Int[Array, " num_lines"]

    def __check_init__(self) -> None:
        (num_lines,) = self.in_batch_index_to_sequence_id.shape
        for leaf in jax.tree.leaves(self):
            if leaf.ndim == 0:
                continue
            if leaf.shape[0] != num_lines:
                raise ValueError(f"BlockContinuousState leaf has first dim {leaf.shape[0]} != num_lines={num_lines}")

    @property
    def num_lines(self) -> int:
        return len(self.last_token_logits)

    @staticmethod
    def init(
        num_lines: int, max_output_length: int, state_capacity: int, model: LanguageModel
    ) -> "BlockContinuousState":
        return BlockContinuousState(
            last_token_logits=jnp.zeros((num_lines, model.model.vocab_size), dtype=jnp.float32),
            last_token_indices=jnp.zeros(num_lines, dtype=jnp.int32),
            kv_state=model.model.init_static_state(num_lines, state_capacity),
            stop_flags=jnp.ones(num_lines, dtype=bool),
            token_buffer=jnp.zeros((num_lines, max_output_length), dtype=jnp.int32),
            num_generated=jnp.zeros(num_lines, dtype=jnp.int32),
            keys=jax.random.split(jax.random.key(0), num_lines),
            in_batch_index_to_sequence_id=jnp.zeros(num_lines, dtype=jnp.int32),
        )

    def empty_lines(self) -> list[int]:
        stopped = np.asarray(self.stop_flags)
        (num_lines,) = self.in_batch_index_to_sequence_id.shape
        return [i for i in range(num_lines) if stopped[i]]

    def extract_completed_sequences(
        self, completed_mask: Bool[Array, " num_lines"]
    ) -> Iterator[tuple[int, GenerationResults]]:
        completed_np = np.asarray(completed_mask)
        in_batch_index_to_sequence_id_np = np.asarray(self.in_batch_index_to_sequence_id)
        for line in np.flatnonzero(completed_np):
            yield (
                int(in_batch_index_to_sequence_id_np[line]),
                GenerationResults(
                    token_ids=self.token_buffer[line],
                    top_k_token_ids=None,
                    top_k_token_logits=None,
                ),
            )

    def is_empty(self) -> bool:
        return bool(jnp.all(self.stop_flags))

    def attempt_halving(self, min_allowed_lines: int) -> "BlockContinuousState":
        active_mask = np.logical_not(np.asarray(self.stop_flags))
        active_count = int(active_mask.sum())
        candidate_num_lines = self.num_lines // 2

        if not (
            active_count > min_allowed_lines // 2
            and candidate_num_lines >= min_allowed_lines
            and candidate_num_lines >= active_count
        ):
            return self

        active_idx_np = np.flatnonzero(active_mask)
        stopped_idx_np = np.flatnonzero(np.logical_not(active_mask))
        pad = stopped_idx_np[: candidate_num_lines - active_count]
        gather_idx = jnp.asarray(np.concatenate([active_idx_np, pad]).astype(np.int32))
        # Donate the old (larger) state so its GPU buffers can be freed in-place
        # by XLA as the gather writes into the new (smaller) buffers, instead of
        # both states briefly coexisting on device.
        return _halve_state(self, gather_idx)


@eqx.filter_jit(donate="all")
def _halve_state(
    state: BlockContinuousState,
    gather_idx: Int[Array, " new_num_lines"],
) -> BlockContinuousState:
    return jax.tree.map(lambda x: x[gather_idx], state)


class BlockContinuousDecoder(eqx.Module):
    block_size: int
    max_output_length: int

    language_model: LanguageModel
    sampling_policy: SamplingPolicy
    forward_pass_config: ForwardPassConfig | None

    @classmethod
    def init(
        cls,
        max_output_length: int,
        block_size: int,
        model: LanguageModel,
        sampling_policy: SamplingPolicy,
        forward_pass_config: ForwardPassConfig | None,
    ) -> "BlockContinuousDecoder":
        return cls(
            block_size=block_size,
            max_output_length=max_output_length,
            language_model=model,
            sampling_policy=sampling_policy,
            forward_pass_config=forward_pass_config,
        )

    def _step(
        self,
        carry: tuple[DecodingState, Key[Array, " num_lines"]],
        eos_token_ids: Int[Array, " eos_tokens"],
    ) -> tuple[tuple[DecodingState, Key[Array, " num_lines"]], Int[Array, " num_lines"]]:
        decode_state, keys = carry

        keys = jax.vmap(jax.random.split)(keys)
        next_keys = keys[:, 0]
        sample_keys = keys[:, 1]

        logits = decode_state.last_token_logits.astype(jnp.float32)
        processed = jax.vmap(self.sampling_policy.process_logits)(logits)
        next_tokens = jax.vmap(jax.random.categorical)(sample_keys, processed)
        next_tokens = jnp.where(decode_state.stop_flags, 0, next_tokens)

        next_indices = decode_state.last_token_indices + 1
        stop_flags = jnp.logical_or(
            decode_state.stop_flags,
            jnp.any(next_tokens[:, None] == eos_token_ids[None, :], axis=-1),
        )

        outputs = self.language_model.model(
            next_tokens[:, None],
            next_indices[:, None],
            decode_state.state,
            return_updated_state=True,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=self.forward_pass_config,
        )
        assert outputs.updated_state is not None
        new_decode_state = DecodingState(
            rearrange(outputs.logits, "num_lines 1 vocab -> num_lines vocab").astype(jnp.float32),
            next_indices,
            outputs.updated_state,
            stop_flags,
        )
        return (new_decode_state, next_keys), next_tokens

    def fill_lines(self, state: BlockContinuousState, batch: PrefillBatch) -> BlockContinuousState:
        (prefill_bs,) = batch.prefill_results.last_token_indices.shape
        prefill_state = DecodingState(
            batch.prefill_results.last_token_logits,
            batch.prefill_results.last_token_indices,
            batch.prefill_results.state,
            jnp.zeros(prefill_bs, dtype=bool),
        )
        current_decode_state = DecodingState(
            state.last_token_logits,
            state.last_token_indices,
            state.kv_state,
            state.stop_flags,
        )

        empty_lines = state.stop_flags
        lines_passed = jnp.sum(batch.mask)
        source_idx = jnp.cumsum(empty_lines, dtype=jnp.int32) - 1
        need_updating = jnp.logical_and(empty_lines, source_idx < lines_passed)

        def apply_updates(
            new: Shaped[Array, "num_lines ..."],
            old: Shaped[Array, "num_lines ..."],
        ) -> Shaped[Array, "num_lines ..."]:
            return jax.vmap(jnp.where)(need_updating, new, old)

        gathered_prefill_state = jax.tree.map(lambda x: x[source_idx], prefill_state)
        new_decode_state = jax.tree.map(apply_updates, gathered_prefill_state, current_decode_state)

        return BlockContinuousState(
            last_token_logits=new_decode_state.last_token_logits,
            last_token_indices=new_decode_state.last_token_indices,
            kv_state=new_decode_state.state,
            stop_flags=new_decode_state.stop_flags,
            token_buffer=state.token_buffer,
            num_generated=apply_updates(jnp.zeros_like(state.num_generated), state.num_generated),
            keys=apply_updates(batch.keys[source_idx], state.keys),
            in_batch_index_to_sequence_id=apply_updates(
                batch.sequence_ids[source_idx], state.in_batch_index_to_sequence_id
            ),
        )

    def decode_block(self, state: BlockContinuousState) -> tuple[BlockContinuousState, Bool[Array, " num_lines"]]:
        eos_token_ids = jnp.asarray(self.language_model.stop_token_ids, dtype=jnp.int32)

        initial_decode_state = DecodingState(
            state.last_token_logits,
            state.last_token_indices,
            state.kv_state,
            state.stop_flags,
        )
        (new_decode_state, new_keys), block_tokens = jax.lax.scan(
            lambda carry, _: self._step(carry, eos_token_ids),
            (initial_decode_state, state.keys),
            None,
            length=self.block_size,
        )

        new_buffer, new_num_generated = append_block_tokens(state.token_buffer, state.num_generated, block_tokens)
        combined_stop = jnp.logical_or(new_decode_state.stop_flags, new_num_generated >= self.max_output_length)
        completed_mask = jnp.logical_and(jnp.logical_not(state.stop_flags), combined_stop)

        return BlockContinuousState(
            last_token_logits=new_decode_state.last_token_logits,
            last_token_indices=new_decode_state.last_token_indices,
            kv_state=new_decode_state.state,
            stop_flags=combined_stop,
            token_buffer=new_buffer,
            num_generated=new_num_generated,
            keys=new_keys,
            in_batch_index_to_sequence_id=state.in_batch_index_to_sequence_id,
        ), completed_mask


class SchedulerKind(StrEnum):
    FIXED = "fixed"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class BatchScheduler(ABC):
    model: LanguageModel

    @abstractmethod
    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig,
        *,
        fast_peak_memory: bool = False,
        generation_trace_config: GenerationTraceConfig | None = None,
        sharding_config: ShardingConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
    ) -> Iterator[tuple[int, GenerationResults]]: ...

    def reply_many(
        self,
        messages: Iterable[Iterable[Message]],
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        sharding_config: ShardingConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
        vram_bytes: int | None = None,
        batch_sizes_callback: Callable[[BatchSizesComputedEvent], None] | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        messages = list(messages)

        if sharding_config is None:
            sharding_config = get_current_sharding_config()
        if generation_config is None:
            generation_config = GenerationConfig()
        if inference_config is None:
            inference_config = InferenceConfig()
        if forward_pass_config is None:
            forward_pass_config = MLPForwardPassConfig()  # TODO(knyazer): fix when we get proper fp config setup

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(messages))

        if len(keys) != len(messages):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        if vram_bytes is not None and inference_config.batch_size is not None:
            raise ValueError("You have to specify only one of batch_size and vram_gb, not both.")

        if vram_bytes is None and inference_config.batch_size is None:
            raise ValueError("You have to specify either batch_size or vram_gb, but you provided neither.")

        tokenized: list[list[int]] = self.model.message_processor.tokenize_requests(messages)

        if inference_config.batch_size is not None:
            sequences_per_bucket = bucket_by_length(tokenized)
            batch_size_per_bucket = dict.fromkeys(sequences_per_bucket, inference_config.batch_size)
        else:
            assert vram_bytes is not None

            def memory_probe(batch_size: int, padded_length: int) -> None:
                iterator = self.generate_tokens_many(
                    [tokenized[0]] * batch_size,
                    generation_config=generation_config,
                    inference_config=replace(inference_config, batch_size=batch_size, padded_length=padded_length),
                    forward_pass_config=forward_pass_config,
                    sharding_config=sharding_config,
                    fast_peak_memory=True,
                )
                # "fast_peak_memory" runs exactly one decode block then returns
                first_result = next(iterator, None)
                if first_result is not None:
                    jax.block_until_ready(first_result)

            sequences_per_bucket, batch_size_per_bucket = bucket_sequences(
                tokenized, memory_probe, max_vram=get_usable_memory_from_bytes(vram_bytes), min_batches_per_bucket=10
            )

        buckets = merge_small_buckets(sequences_per_bucket, batch_size_per_bucket, min_batches=10)
        assert sum(len(bucket) for bucket in buckets.values()) == len(tokenized)

        if batch_sizes_callback is not None:
            batch_sizes = tuple(
                BatchSizeInfo(
                    prefix_length=padded_length,
                    num_elements=len(buckets[padded_length]),
                    batch_size=batch_size_per_bucket.get(padded_length, 1),
                )
                for padded_length in sorted(buckets.keys())
            )
            batch_sizes_callback(BatchSizesComputedEvent(batch_sizes=batch_sizes))

        for padded_length in sorted(buckets.keys(), reverse=True):
            sequence_ids, sequence_tokenized = zip(*buckets[padded_length], strict=True)
            sequence_ids = list(sequence_ids)
            batch_size = batch_size_per_bucket[padded_length]

            bucket_inference_config = replace(inference_config, batch_size=batch_size, padded_length=padded_length)

            all_results = self.generate_tokens_many(
                sequence_tokenized,
                generation_config=generation_config,
                inference_config=bucket_inference_config,
                fast_peak_memory=False,
                forward_pass_config=forward_pass_config,
                keys=keys[jnp.asarray(sequence_ids)],
                sharding_config=sharding_config,
            )

            for local_idx, result in all_results:
                idx = sequence_ids[local_idx]
                trimmed_ids = self.model.trim_at_eos(result.token_ids.tolist())
                response = self.model.message_processor.parse_tokenized_response(trimmed_ids)
                yield (idx, response)


@dataclass(frozen=True)
class FixedBatchScheduler(BatchScheduler):
    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig,
        *,
        fast_peak_memory: bool = False,
        generation_trace_config: GenerationTraceConfig | None = None,
        sharding_config: ShardingConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
    ) -> Iterator[tuple[int, GenerationResults]]:
        tokenized = list(tokenized)

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(tokenized))

        if len(keys) != len(tokenized):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        def process_batches(batch_size: int) -> Iterator[tuple[int, GenerationResults]]:
            new_inference_config = replace(inference_config, batch_size=batch_size)

            for batch_items in batched(enumerate(zip(tokenized, keys, strict=True)), batch_size):
                batch_with_ids = list(batch_items)
                batch_indices = [i for i, _ in batch_with_ids]
                real_batch = tuple(tokens for _, (tokens, _) in batch_with_ids)
                batch_keys = tuple(key for _, (_, key) in batch_with_ids)
                batch_results = self.model._generate_tokens_batch(  # noqa: SLF001
                    real_batch,
                    batch_keys,
                    generation_config=generation_config,
                    inference_config=new_inference_config,
                    forward_pass_config=forward_pass_config,
                    generation_trace_config=generation_trace_config,
                    sharding_config=sharding_config,
                )
                yield from zip(batch_indices, batch_results, strict=True)

        assert inference_config.batch_size is not None
        if fast_peak_memory:
            yield from process_batches(inference_config.batch_size)
            return

        yield from decrease_batchsize_on_oom(process_batches, starting_batch_size=inference_config.batch_size)


@dataclass(frozen=True)
class ContinuousBatchScheduler(BatchScheduler):
    """Block-continuous batching scheduler.

    Keeps a fixed-size batch of active decode lines. Every ``continuous_batching_block_size``
    decode steps, finished lines (those that emitted an EOS token or reached
    ``max_output_length``) are evicted and replaced with fresh sequences from the
    prefill generator. Replacement sequences are prefilled and scattered into the
    vacated KV-cache lines so all data stays on device except for a small
    ``[num_lines]`` boolean stop-flag read on the host for scheduling decisions.

    Heuristics: defaults were tuned on a ~12k-row workload and may underperform
    on small ones. prefill_batch_fraction encodes a guess at mean completion
    length as a fraction of block_size (0.2 ≈ 1280 tokens at block_size=256).
    """

    continuous_batching_block_size: int = 256
    prefill_batch_fraction: float = 0.2
    min_allowed_batch_fraction: float = 0.2

    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig,
        *,
        fast_peak_memory: bool = False,
        generation_trace_config: GenerationTraceConfig | None = None,
        sharding_config: ShardingConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
    ) -> Iterator[tuple[int, GenerationResults]]:
        if generation_trace_config is not None:
            raise NotImplementedError("generation_trace_config is not supported with ContinuousBatchScheduler")
        if inference_config.num_top_logits_to_return is not None:
            raise NotImplementedError("num_top_logits_to_return is not supported with ContinuousBatchScheduler")
        if sharding_config is not None:
            raise NotImplementedError(
                "Sharding was not tested for continuous batching; we preemptively disallow it since its likely "
                "going to be just silently ignored. Prefer running an instance per GPU - which is basically DDP."
            )

        tokenized = list(tokenized)
        if len(tokenized) == 0:
            return

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(tokenized))
        if len(keys) != len(tokenized):
            raise ValueError(
                f"Length of 'keys' should equal the number of sequences; got {len(keys)} vs {len(tokenized)}",
            )

        assert inference_config.batch_size is not None

        sampling_policy = (
            generation_config.default_policy()
            if generation_config is not None
            else self.model.default_sampling_policy()
        )

        batch_size = inference_config.batch_size
        max_output_length = inference_config.max_output_length
        padded_length = inference_config.padded_length

        block_size = min(self.continuous_batching_block_size, max_output_length)
        state_capacity = padded_length + max_output_length
        prefill_batch_size = max(1, min(int(batch_size * self.prefill_batch_fraction + 1), batch_size))
        min_allowed_lines = max(1, int(batch_size * self.min_allowed_batch_fraction))

        prefills = PrefillSource(
            tokenized_messages=tokenized,
            keys=keys,
            prefill_batch_size=prefill_batch_size,
            padded_length=padded_length,
            state_capacity=state_capacity,
            forward_pass_config=forward_pass_config,
        )
        decoder = BlockContinuousDecoder.init(
            max_output_length,
            block_size,
            self.model,
            sampling_policy,
            forward_pass_config,
        )
        state = BlockContinuousState.init(batch_size, max_output_length, state_capacity, self.model)

        jitted_decode = eqx.filter_jit(decoder.decode_block, donate="all")
        jitted_fill_lines = eqx.filter_jit(decoder.fill_lines, donate="all")

        while True:
            # Loop prefills in small chunks to make sure that the state decode gets is complete
            prefills, state = prefills.fill(decoder, state, jitted_fill_lines)

            state, completed_mask = jitted_decode(state)

            yield from state.extract_completed_sequences(completed_mask)

            if state.is_empty() and prefills.exhausted:
                break

            if prefills.exhausted:
                state = state.attempt_halving(min_allowed_lines)

            if fast_peak_memory:
                break
