import functools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import vmap
from jax._src.stages import Compiled
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from lalamo.common import decrease_batchsize_on_oom
from lalamo.message_processor import AssistantMessage, Message, MessageProcessor
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    ForwardPassMode,
    LalamoModule,
    State,
)
from lalamo.sampling import SamplingPolicy, make_policy

from .common import TextModel, TextModelConfig

__all__ = [
    "ForwardPassConfig",
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


_PREFILL_CHUNK_LENGTH = 1024
_COMPILED_PROMPT_LENGTHS = [512 * 2**i for i in range(10)]


type ForwardPassConfig = DecoderForwardPassConfig


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    stop_flags: Bool[Array, " batch"]


class GenerationStepResults(NamedTuple):
    token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, " batch k"] | None
    top_k_token_logits: Float[Array, " batch k"] | None


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None


@dataclass(frozen=True)
class GenerationConfig:
    stop_token_ids: tuple[int, ...] = tuple()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None

    def default_policy(self) -> SamplingPolicy:
        return make_policy(self.temperature, self.top_k, self.top_p, self.min_p, self.banned_tokens)


@dataclass(frozen=True)
class LanguageModelConfig(TextModelConfig[DecoderConfig]):
    generation_config: GenerationConfig

    def init(
        self,
        model: LalamoModule,
        message_processor: MessageProcessor,
    ) -> "LanguageModel":
        assert isinstance(model, Decoder)
        return LanguageModel(self, model, message_processor)

    @classmethod
    def load_model(cls, path: Path | str) -> "LanguageModel":
        result = super().load_model(path)
        assert isinstance(result, LanguageModel)
        return result


class LanguageModel(TextModel[LanguageModelConfig, Decoder]):
    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.config.generation_config.stop_token_ids

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    @eqx.filter_jit
    def _prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        state_capacity: int | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape

        if state_capacity is not None and sequence_length > _PREFILL_CHUNK_LENGTH:
            return self._chunked_prefill(
                token_ids,
                lengths_without_padding,
                state_capacity,
                forward_pass_config,
            )

        token_positions = jnp.repeat(jnp.arange(sequence_length, dtype=jnp.int32)[None, ...], batch_size, axis=0)
        if state_capacity is not None:
            state = self.model.init_static_state(batch_size, state_capacity)
        else:
            state = None

        decoder_outputs = self.model(
            token_ids,
            token_positions,
            state,
            return_updated_state=True,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=forward_pass_config,
        )

        if lengths_without_padding is not None:
            last_logits_indices = lengths_without_padding - 1
        else:
            last_logits_indices = jnp.array([sequence_length - 1] * batch_size, dtype=jnp.int32)

        last_token_logits = vmap(lambda logits, index: logits[index])(decoder_outputs.logits, last_logits_indices)

        assert decoder_outputs.updated_state is not None
        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=last_logits_indices,
            state=decoder_outputs.updated_state,
        )

    def _chunked_prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None,
        state_capacity: int,
        forward_pass_config: ForwardPassConfig | None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        num_chunks = (sequence_length + _PREFILL_CHUNK_LENGTH - 1) // _PREFILL_CHUNK_LENGTH
        chunked_length = num_chunks * _PREFILL_CHUNK_LENGTH
        padding_length = chunked_length - sequence_length

        token_ids = jnp.pad(token_ids, [(0, 0), (0, padding_length)])
        token_positions = jnp.repeat(
            jnp.arange(chunked_length, dtype=jnp.int32)[None, :],
            batch_size,
            axis=0,
        )

        chunked_token_ids = rearrange(
            token_ids,
            "batch (c cl) -> c batch cl",
            cl=_PREFILL_CHUNK_LENGTH,
        )
        chunked_positions = rearrange(
            token_positions,
            "batch (c cl) -> c batch cl",
            cl=_PREFILL_CHUNK_LENGTH,
        )

        if lengths_without_padding is not None:
            effective_lengths = lengths_without_padding
        else:
            effective_lengths = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)
        chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32)[:, None] * _PREFILL_CHUNK_LENGTH
        per_chunk_lengths = jnp.clip(effective_lengths[None, :] - chunk_starts, 0, _PREFILL_CHUNK_LENGTH)

        state = self.model.init_static_state(batch_size, state_capacity + padding_length)

        def apply_chunk(state: State, chunk_data: tuple) -> tuple[State, Float[Array, "batch cl vocab"]]:
            chunk_tokens, chunk_positions, chunk_lengths = chunk_data
            decoder_outputs = self.model(
                chunk_tokens,
                chunk_positions,
                state,
                return_updated_state=True,
                lengths_without_padding=chunk_lengths,
                forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_state is not None
            return decoder_outputs.updated_state, decoder_outputs.logits

        final_state, all_logits = jax.lax.scan(
            apply_chunk,
            state,
            (chunked_token_ids, chunked_positions, per_chunk_lengths),
        )

        all_logits = rearrange(all_logits, "c batch cl vocab -> batch (c cl) vocab")

        if lengths_without_padding is not None:
            last_logits_indices = lengths_without_padding - 1
        else:
            last_logits_indices = jnp.array([sequence_length - 1] * batch_size, dtype=jnp.int32)

        last_token_logits = vmap(lambda logits, index: logits[index])(all_logits, last_logits_indices)

        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=last_logits_indices,
            state=final_state,
        )

    @eqx.filter_jit
    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        num_top_logits_to_return: int | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> GenerationResults:
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)

        batch_size, sequence_length = prompt_token_ids.shape
        prefill_results = self._prefill(
            prompt_token_ids,
            prompt_lengths_without_padding,
            sequence_length + max_output_length,
            forward_pass_config=forward_pass_config,
        )

        initial_state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_indices,
            prefill_results.state,
            jnp.zeros(batch_size, dtype=jnp.bool),
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=max_output_length)

        def loop_iteration(
            state: DecodingState,
            key: PRNGKeyArray,
        ) -> tuple[DecodingState, GenerationStepResults]:
            def sample_and_update() -> tuple[DecodingState, GenerationStepResults]:
                upcasted_logits = state.last_token_logits.astype(jnp.float32)
                processed_logits = vmap(sampling_policy.process_logits)(upcasted_logits)
                next_token_ids = jax.random.categorical(key, processed_logits)
                next_token_ids = jnp.where(state.stop_flags, jnp.zeros(batch_size, dtype=jnp.int32), next_token_ids)
                if num_top_logits_to_return is not None:
                    next_top_k_token_logits, next_top_k_token_ids = jax.lax.top_k(
                        processed_logits,
                        num_top_logits_to_return,
                    )
                else:
                    next_top_k_token_ids = None
                    next_top_k_token_logits = None
                next_token_indices = state.last_token_indices + 1

                stop_flags = state.stop_flags | jnp.any(next_token_ids[:, None] == eos_token_ids[None, :], axis=-1)

                if batch_size == 1:
                    forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
                else:
                    forward_pass_mode = ForwardPassMode.MULTI_TOKEN

                decoder_outputs = self.model(
                    next_token_ids[:, None],
                    next_token_indices[:, None],
                    state.state,
                    return_updated_state=True,
                    forward_pass_mode=forward_pass_mode,
                    forward_pass_config=forward_pass_config,
                )
                assert decoder_outputs.updated_state is not None, "updated_state should not be None"
                new_state = DecodingState(
                    decoder_outputs.logits.squeeze(1),
                    next_token_indices,
                    decoder_outputs.updated_state,
                    stop_flags,
                )
                return new_state, GenerationStepResults(next_token_ids, next_top_k_token_ids, next_top_k_token_logits)

            def pad_and_repeat_state() -> tuple[DecodingState, GenerationStepResults]:
                (batch_size,) = state.stop_flags.shape
                pad_token = jnp.zeros(batch_size, dtype=jnp.int32)
                if num_top_logits_to_return is not None:
                    top_k_token_ids = jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.int32)
                    top_k_token_logits = jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.float32)
                else:
                    top_k_token_ids = None
                    top_k_token_logits = None
                return state, GenerationStepResults(pad_token, top_k_token_ids, top_k_token_logits)

            return jax.lax.cond(jnp.all(state.stop_flags), pad_and_repeat_state, sample_and_update)

        _, generated = jax.lax.scan(loop_iteration, initial_state, keys)

        token_ids = rearrange(generated.token_ids, "iteration batch -> batch iteration")

        if num_top_logits_to_return is not None:
            top_k_token_ids = rearrange(generated.top_k_token_ids, "iteration batch k -> batch iteration k")
            top_k_token_logits = rearrange(generated.top_k_token_logits, "iteration batch k -> batch iteration k")
        else:
            top_k_token_ids = None
            top_k_token_logits = None

        return GenerationResults(token_ids, top_k_token_ids, top_k_token_logits)

    def reply(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> AssistantMessage:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)[None, :]
        response_ids = self.generate_tokens(
            token_ids,
            sampling_policy,
            forward_pass_config=forward_pass_config,
            key=key,
        ).token_ids.squeeze(0)
        response_text = self.message_processor.detokenize(response_ids.tolist())
        return self.message_processor.parse_response(response_text)

    def reply_many(
        self,
        messages: Iterable[list[Message]],
        sampling_policy: SamplingPolicy | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        max_vram: int | None = None,
        max_output_length: int = 8192,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        # Automatically batches the list of message sequences and vmap's over batches,
        # while making sure that if OOM happens the batch size is decreased.
        # Sequences are grouped by padded input length and processed longest-first,
        # so output order may differ from input order. Each result includes its original index.

        # sadly we have to instantiate the dataset eagerly here - since to keep adequate batch sizes
        # but at least we only instantiate the tokens, which makes memory consumption a bit smaller.
        # We use numpy arrays to keep them in RAM instead of GPU memory.
        tokenized = [
            np.array(
                self.message_processor.tokenize_text(self.message_processor.render_request(msg)),
                dtype=np.int32,
            )
            for msg in messages
        ]
        length_buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for idx, tokens in enumerate(tokenized):
            padded_len = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= len(tokens))
            length_buckets.setdefault(padded_len, []).append((idx, tokens))

        if not length_buckets:
            return

        # Estimate batch sizes by interpolating between smallest and largest prompt lengths
        from lalamo.speculator.estimator import estimate_batchsize_from_memory

        sorted_lengths = sorted(length_buckets.keys())
        min_len, max_len = sorted_lengths[0], sorted_lengths[-1]

        if max_vram is None:
            # Fallback: batch_size=1 for all buckets
            batch_size_for_length = dict.fromkeys(sorted_lengths, 1)
        elif min_len == max_len:
            # Only one bucket size
            bs = estimate_batchsize_from_memory(self, min_len, max_output_length, None, max_vram)
            batch_size_for_length = {min_len: max(1, bs)}
        else:
            # Estimate for smallest (highest batch size) and largest (lowest batch size) buckets
            bs_at_min = estimate_batchsize_from_memory(self, min_len, max_output_length, None, max_vram)
            bs_at_max = estimate_batchsize_from_memory(self, max_len, max_output_length, None, max_vram)
            bs_at_min = max(1, bs_at_min)
            bs_at_max = max(1, bs_at_max)

            # Linear interpolation for intermediate lengths
            def interpolate_batch_size(length: int) -> int:
                t = (length - min_len) / (max_len - min_len)
                return max(1, int(bs_at_min + t * (bs_at_max - bs_at_min)))

            batch_size_for_length = {length: interpolate_batch_size(length) for length in sorted_lengths}

        # Merge small buckets and round to batch_size multiples.
        # Process small-to-large: keep multiples of batch_size, push remainder to next bucket.
        # Buckets with too few elements are merged entirely into the next bucket.
        min_batches = 10
        merged_buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        overflow: list[tuple[int, np.ndarray]] = []
        for i, padded_len in enumerate(sorted_lengths):
            batch_size = batch_size_for_length[padded_len]
            items = overflow + length_buckets[padded_len]
            is_last = i == len(sorted_lengths) - 1
            if is_last:
                # Last bucket takes everything
                if items:
                    merged_buckets[padded_len] = items
            elif len(items) < min_batches * batch_size:
                # Too small, push everything to next bucket
                overflow = items
            else:
                # Keep a multiple of batch_size, push remainder up
                keep_count = (len(items) // batch_size) * batch_size
                merged_buckets[padded_len] = items[:keep_count]
                overflow = items[keep_count:]
        length_buckets = merged_buckets

        @functools.cache
        def make_generate_tokens_compiled(batch_size: int, padded_input_length: int) -> Compiled:
            print(batch_size, padded_input_length)
            return (
                jax.jit(
                    functools.partial(
                        LanguageModel.generate_tokens,
                        sampling_policy=sampling_policy,
                        max_output_length=max_output_length,
                        forward_pass_config=forward_pass_config,
                        key=key,
                    ),
                )
                .lower(
                    self,
                    prompt_token_ids=jax.ShapeDtypeStruct((batch_size, padded_input_length), jnp.int32),
                    prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
                )
                .compile(compiler_options={"xla_gpu_autotune_level": "0"})
            )

        # Process longest sequences first so batchsize=1 OOM happens early before wasting work
        for padded_input_length in sorted(length_buckets.keys(), reverse=True):
            bucket = length_buckets[padded_input_length]

            def process_bucket(
                batch_size: int,
                bucket: list[tuple[int, np.ndarray]],
                padded_input_length: int,
            ) -> Iterator[tuple[int, AssistantMessage]]:
                for real_batch in batched(bucket, batch_size):
                    indices = [idx for idx, _ in real_batch]
                    tokens_list = [tokens for _, tokens in real_batch]

                    batch_padding = batch_size - len(tokens_list)
                    batch = (*tokens_list, *(np.array([0], dtype=np.int32),) * batch_padding)

                    # Pad in numpy (RAM), then convert to jnp (GPU) only at the end
                    padded = jnp.array(
                        np.array(
                            [
                                np.pad(tokens, (0, padded_input_length - len(tokens)), constant_values=0)
                                for tokens in batch
                            ],
                        ),
                    )
                    lengths = jnp.array([len(tokens) for tokens in batch], dtype=jnp.int32)

                    generate_compiled = make_generate_tokens_compiled(batch_size, padded_input_length)
                    results = generate_compiled(
                        self,
                        prompt_token_ids=padded,
                        prompt_lengths_without_padding=lengths,
                    )

                    for i, idx in enumerate(indices):
                        response_text = self.message_processor.detokenize(results.token_ids[i].tolist())
                        yield idx, self.message_processor.parse_response(response_text)

            yield from decrease_batchsize_on_oom(
                functools.partial(process_bucket, bucket=bucket, padded_input_length=padded_input_length),
                batch_size_for_length[padded_input_length],
            )

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        max_output_length: int = 8192,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[str]:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)
        for token_id in self.stream_tokens(
            token_ids,
            sampling_policy,
            max_output_length,
            forward_pass_config=forward_pass_config,
            key=key,
        ):
            yield self.message_processor.detokenize([token_id.item()])

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[Int[Array, ""]]:
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape

        padded_input_length = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length)
        padded_token_ids = jnp.zeros((padded_input_length,), dtype=jnp.int32)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)

        prefill_results = self._prefill(
            padded_token_ids[None, :],
            jnp.array([input_length], dtype=jnp.int32),
            padded_input_length + max_output_length,
            forward_pass_config=forward_pass_config,
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=max_output_length)

        state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_indices,
            prefill_results.state,
            jnp.array([0], dtype=jnp.bool),
        )

        for iter_key in keys:
            upcasted_logits = state.last_token_logits.astype(jnp.float32)
            processed_logits = sampling_policy.process_logits(upcasted_logits.squeeze(0))
            next_token_id = jax.random.categorical(iter_key, processed_logits)

            yield next_token_id

            if jnp.any(next_token_id == eos_token_ids):
                return

            next_token_indices = state.last_token_indices + 1
            decoder_outputs = self.model(
                next_token_id.reshape(1, 1),
                next_token_indices.reshape(1, 1),
                state.state,
                return_updated_state=True,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_state is not None, "updated_state should not be None"
            state = DecodingState(
                decoder_outputs.logits.squeeze(1),
                next_token_indices,
                decoder_outputs.updated_state,
                state.stop_flags,
            )
