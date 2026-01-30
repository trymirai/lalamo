import functools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
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
from jaxtyping import Array, Bool, Float, Int, Key, PRNGKeyArray

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

from .common import InferenceConfig, TextModel, TextModelConfig
from .lm_helpers import decrease_batchsize_on_oom, estimate_batchsizes_from_vram, merge_small_buckets

__all__ = [
    "ForwardPassConfig",
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = [256 * 2**i for i in range(12)]


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


_compile_cache = {}


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
        chunked_prefill_cutoff: int = 2048,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape

        if sequence_length > chunked_prefill_cutoff:
            assert state_capacity is not None, "state_capacity must be provided for chunked prefill"
            return self._chunked_prefill(
                token_ids,
                lengths_without_padding,
                state_capacity,
                forward_pass_config=forward_pass_config,
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
        chunk_size: int = 1024,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        chunked_length = num_chunks * chunk_size
        padding_length = chunked_length - sequence_length

        token_ids = jnp.pad(token_ids, [(0, 0), (0, padding_length)])
        token_positions = jnp.repeat(
            jnp.arange(chunked_length, dtype=jnp.int32)[None, :],
            batch_size,
            axis=0,
        )

        chunked_token_ids = rearrange(
            token_ids,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        chunked_positions = rearrange(
            token_positions,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )

        if lengths_without_padding is not None:
            effective_lengths = lengths_without_padding
        else:
            effective_lengths = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)
        chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32)[:, None] * chunk_size
        per_chunk_lengths = jnp.clip(effective_lengths[None, :] - chunk_starts, 0, chunk_size)

        state = self.model.init_static_state(batch_size, state_capacity + padding_length)

        def apply_chunk(state: State, chunk_data: tuple) -> tuple[State, Float[Array, "batch chunk_size vocab"]]:
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

        all_logits = rearrange(all_logits, "num_chunks batch chunk_size vocab -> batch (num_chunks chunk_size) vocab")

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
        keys: Key[Array, " batch"] | None = None,
    ) -> GenerationResults:
        batch_size, sequence_length = prompt_token_ids.shape

        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)
        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=batch_size)

        if len(keys) != batch_size:
            raise ValueError(
                f"Length of 'keys' should be equal to the batch size, or keys should be None; got {len(keys)}",
            )

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
            key=key[None, :],
        ).token_ids.squeeze(0)
        response_text = self.message_processor.detokenize(response_ids.tolist())
        return self.message_processor.parse_response(response_text)

    def compile_generate_tokens(
        self,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> Compiled:
        key = (inference_config, forward_pass_config)
        if key not in _compile_cache:
            _compile_cache[key] = (
                jax.jit(
                    functools.partial(
                        LanguageModel.generate_tokens,
                        sampling_policy=inference_config.sampling_policy,
                        max_output_length=inference_config.max_output_length,
                        num_top_logits_to_return=inference_config.num_top_logits_to_return,
                        forward_pass_config=forward_pass_config,
                    ),
                )
                .lower(
                    self,
                    prompt_token_ids=jax.ShapeDtypeStruct(
                        (inference_config.batch_size, inference_config.padded_length),
                        jnp.int32,
                    ),
                    prompt_lengths_without_padding=jax.ShapeDtypeStruct((inference_config.batch_size,), jnp.int32),
                    keys=jax.ShapeDtypeStruct((inference_config.batch_size,), jax.random.key(0).dtype),
                )
                .compile(compiler_options={"xla_gpu_autotune_level": "0"})
            )
        return _compile_cache[key]

    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        keys: Key[Array, " num_sequences"] | None = None,
    ) -> Iterator[GenerationResults]:
        tokenized = list(tokenized)  # load eagerly to RAM

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(tokenized))

        if len(keys) != len(tokenized):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        def process_batches(batch_size: int) -> Iterator[tuple[int, GenerationResults]]:
            new_inference_config = replace(inference_config, batch_size=batch_size)
            for batch_items in batched(zip(tokenized, keys, strict=True), batch_size):
                real_batch, batch_keys = zip(*batch_items, strict=True)
                batch_padding = batch_size - len(real_batch)
                batch = (*real_batch, *(np.array([0], dtype=np.int32),) * batch_padding)

                # Pad keys if necessary
                padded_keys = jnp.array(batch_keys)
                if batch_padding > 0:
                    dummy_keys = jax.random.split(jax.random.key(0), batch_padding)
                    padded_keys = jnp.concatenate([jnp.array(batch_keys), dummy_keys])

                padded = jnp.array(
                    np.array(
                        [
                            np.pad(tokens, (0, inference_config.padded_length - len(tokens)), constant_values=0)
                            for tokens in batch
                        ],
                    ),
                )
                lengths = jnp.array([len(tokens) for tokens in batch], dtype=jnp.int32)

                generate_tokens = self.compile_generate_tokens(new_inference_config, forward_pass_config)
                results = generate_tokens(
                    self,
                    prompt_token_ids=padded,
                    prompt_lengths_without_padding=lengths,
                    keys=padded_keys,
                )

                for i in range(len(real_batch)):
                    yield GenerationResults(
                        token_ids=results.token_ids[i],
                        top_k_token_ids=results.top_k_token_ids[i] if results.top_k_token_ids is not None else None,
                        top_k_token_logits=results.top_k_token_logits[i]
                        if results.top_k_token_logits is not None
                        else None,
                    )

        yield from decrease_batchsize_on_oom(
            process_batches,
            starting_batch_size=inference_config.batch_size,
        )

    def memory_consumption(
        self,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> int:
        memory_analysis = self.compile_generate_tokens(inference_config, forward_pass_config).memory_analysis()

        assert hasattr(memory_analysis, "argument_size_in_bytes")
        assert hasattr(memory_analysis, "output_size_in_bytes")
        assert hasattr(memory_analysis, "temp_size_in_bytes")

        return (
            memory_analysis.argument_size_in_bytes
            + memory_analysis.output_size_in_bytes
            + memory_analysis.temp_size_in_bytes
        )

    def reply_many(
        self,
        dataset: Iterable[Iterable[Message]],
        inference_config: InferenceConfig,
        vram_bytes: int | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        dataset = list(dataset)  # eagerly load the dataset into RAM

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(dataset))

        if len(keys) != len(dataset):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        if vram_bytes is not None and inference_config.batch_size is not None:
            raise ValueError("You have to specify only one of batch_size and vram_gb, not both.")

        if vram_bytes is None and inference_config.batch_size is None:
            raise ValueError("You have to specify either batch_size or vram_gb, but you provided neither.")

        tokenized: list[list[int]] = self.message_processor.tokenize_requests(dataset)

        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for idx, sequence in enumerate(tokenized):
            # we choose the smallest size from precomputed ones that is longer or equal to the current sequence
            padded_len = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= len(sequence))
            buckets.setdefault(padded_len, []).append((idx, sequence))
        sorted_lengths = sorted(buckets.keys())

        if inference_config.batch_size is not None:
            batch_size_per_bucket = dict.fromkeys(sorted_lengths, inference_config.batch_size)
        else:
            batch_size_per_bucket = estimate_batchsizes_from_vram(
                self.memory_consumption,
                sorted_lengths,
                vram_bytes,
                inference_config,
            )
        buckets = merge_small_buckets(buckets, batch_size_per_bucket, min_batches=4)

        # Process longest sequences first so batchsize=1 OOM happens as early as possible, if it does happen
        for padded_length in sorted(buckets.keys(), reverse=True):
            sequence_ids, sequence_tokenized = zip(*buckets[padded_length])
            sequence_ids = list(sequence_ids)
            batch_size = batch_size_per_bucket[padded_length]

            bucket_inference_config = replace(inference_config, batch_size=batch_size, padded_length=padded_length)

            all_results = self.generate_tokens_many(
                sequence_tokenized,
                bucket_inference_config,
                forward_pass_config,
                keys=keys[np.array(sequence_ids)],
            )

            for idx, result in zip(sequence_ids, all_results, strict=True):
                response = self.message_processor.parse_tokenized_response(result.token_ids.tolist())
                yield (idx, response)

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
