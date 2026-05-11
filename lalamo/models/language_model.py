from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, Key, PRNGKeyArray

from lalamo.message_processor import AssistantMessage, Message, MessageProcessor
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    ForwardPassMode,
    LalamoModule,
    ShardingConfig,
    State,
    get_current_sharding_config,
    pad_and_apply_data_sharding,
)
from lalamo.sampling import SamplingPolicy, make_policy
from lalamo.speculator.common import NoSpeculator, Speculator
from lalamo.speculator.proposal import AcceptedProposal
from lalamo.speculator.state import LMState, MemoryBuffers, StateRequest

from .common import (
    BatchSizeInfo,
    BatchSizesComputedEvent,
    InferenceConfig,
    TextModel,
    TextModelConfig,
    split_into_rolling_keys,
)
from .compile_helpers import compile_generate_tokens
from .lm_helpers import (
    decrease_batchsize_on_oom,
    estimate_batchsizes_from_vram,
    merge_small_buckets,
    pad_keys_to_size,
    pad_sequences,
)

__all__ = [
    "ForwardPassConfig",
    "GenerationConfig",
    "GenerationTraceConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = [256 * 2**i for i in range(12)]


type ForwardPassConfig = DecoderForwardPassConfig


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    memory: MemoryBuffers


class DecodingState(NamedTuple):
    sampling_policy: SamplingPolicy
    lm_state: LMState


class DecodingSetup(NamedTuple):
    initial_state: DecodingState
    state_request: StateRequest
    traced_layers: tuple[int, ...]
    traced_layers_array: Int[Array, " traced_layers"]
    step: Callable[[DecodingState], tuple[DecodingState, AcceptedProposal]]
    is_done: Callable[[DecodingState], Bool[Array, " batch"]]


class StepTrace(NamedTuple):
    top_k_ids: Int[Array, "batch k"]
    top_k_logits: Float[Array, "batch k"]
    logsumexp: Float[Array, " batch"]
    activation_output: Float[Array, "batch hidden"]
    layer_output: Float[Array, "batch n_layers hidden"]
    layer_indices: Int[Array, " n_layers"]

    def _collapse_scan_layer_indices(self) -> Int[Array, " n_layers"]:
        first_step_layer_indices = self.layer_indices[0]
        return eqx.error_if(
            first_step_layer_indices,
            jnp.logical_not(jnp.all(self.layer_indices == first_step_layer_indices[None, :])),
            "layer_indices must stay constant across generation steps.",
        )

    def rearrange_from_scan(self) -> "StepTrace":
        return StepTrace(
            top_k_ids=rearrange(self.top_k_ids, "generation_step batch top_k -> batch generation_step top_k"),
            top_k_logits=rearrange(self.top_k_logits, "generation_step batch top_k -> batch generation_step top_k"),
            logsumexp=rearrange(self.logsumexp, "generation_step batch -> batch generation_step"),
            activation_output=rearrange(
                self.activation_output,
                "generation_step batch hidden -> batch generation_step hidden",
            ),
            layer_output=rearrange(
                self.layer_output,
                "generation_step batch traced_layer hidden -> batch traced_layer generation_step hidden",
            ),
            layer_indices=self._collapse_scan_layer_indices(),
        )


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None
    trace: StepTrace | None = None


@dataclass(frozen=True)
class GenerationTraceConfig:
    num_logits_per_token: int = 1024
    trace_layers: tuple[int, ...] = tuple()

    def resolve_layers(self, num_layers: int) -> tuple[int, ...]:
        resolved = tuple(i if i >= 0 else num_layers + i for i in self.trace_layers)
        if any(i < 0 or i >= num_layers for i in resolved):
            raise ValueError("trace_layers index out of range.")
        return tuple(sorted(set(resolved)))


@dataclass(frozen=True)
class GenerationConfig:
    stop_token_ids: tuple[int, ...] = tuple()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def default_policy(self, vocab_size: int) -> SamplingPolicy:
        return make_policy(
            vocab_size=vocab_size,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            banned_tokens=self.banned_tokens,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )


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


class Chunk(eqx.Module):
    tokens: Int[Array, "num_chunks batch chunk_size"]
    indices: Int[Array, "num_chunks batch chunk_size"]
    sequence_ends: Int[Array, "num_chunks batch"]
    is_last_token_inside: Bool[Array, "num_chunks batch"]


class LanguageModel(TextModel[LanguageModelConfig, Decoder]):
    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.config.generation_config.stop_token_ids

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy(self.model.vocab_size)

    def resolve_eos_token_ids(
        self,
        generation_config: GenerationConfig | None,
        eos_token_ids: Int[Array, " eos_tokens"] | None,
    ) -> Int[Array, " eos_tokens"]:
        if eos_token_ids is not None:
            return eos_token_ids
        if generation_config is not None and generation_config.stop_token_ids:
            return jnp.array(generation_config.stop_token_ids, dtype=jnp.int32)
        return jnp.array(self.stop_token_ids, dtype=jnp.int32)

    def state_request_for_generation(
        self,
        speculator: Speculator,
        generation_trace_config: GenerationTraceConfig | None,
        num_top_logits_to_return: int | None,
        max_output_length: int,
    ) -> tuple[StateRequest, tuple[int, ...], Int[Array, " traced_layers"], int]:
        traced_layers: tuple[int, ...] = ()
        traced_layers_array = jnp.zeros(0, dtype=jnp.int32)
        trace_top_k = 0
        if generation_trace_config is not None:
            traced_layers = generation_trace_config.resolve_layers(len(self.model.transformer.layers))
            traced_layers_array = jnp.asarray(traced_layers, dtype=jnp.int32)
            trace_top_k = min(generation_trace_config.num_logits_per_token, self.model.vocab_size)

        speculator_request = speculator.state_request
        layer_indices = speculator_request.layer_indices + tuple(
            layer_index for layer_index in traced_layers if layer_index not in speculator_request.layer_indices
        )
        layer_capacity = speculator_request.layer_capacity
        if traced_layers:
            layer_capacity = max(layer_capacity, max_output_length)

        return (
            StateRequest(
                token_id_capacity=max(speculator_request.token_id_capacity, max_output_length),
                sampling_top_k_capacity=max_output_length if num_top_logits_to_return is not None else 0,
                sampling_top_k_count=num_top_logits_to_return or 0,
                trace_top_k_capacity=max_output_length if generation_trace_config is not None else 0,
                trace_top_k_count=trace_top_k if generation_trace_config is not None else 0,
                logsumexp_capacity=max_output_length if generation_trace_config is not None else 0,
                output_norm_capacity=max(
                    speculator_request.output_norm_capacity,
                    max_output_length if generation_trace_config is not None else 0,
                ),
                layer_capacity=layer_capacity,
                layer_indices=layer_indices if layer_capacity > 0 else (),
            ),
            traced_layers,
            traced_layers_array,
            trace_top_k,
        )

    @eqx.filter_jit
    def _make_chunks(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None,
        chunk_size: int,
    ) -> Chunk:
        batch_size, sequence_length = token_ids.shape
        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        # If all sequences fit in a single chunk, use sequence_length as the chunk size
        chunk_size = min(chunk_size, sequence_length)

        n_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = n_chunks * chunk_size

        token_ids = jnp.pad(token_ids, [(0, 0), (0, padded_length - sequence_length)])

        # Reshape tokens to (num_chunks, batch, chunk_size)
        tokens = rearrange(
            token_ids,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )

        # Create position indices (num_chunks, batch, chunk_size)
        indices = jnp.arange(padded_length, dtype=jnp.int32)
        indices = repeat(indices, "token_idx -> batch token_idx", batch=batch_size)
        indices = rearrange(
            indices,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )

        # sequence_ends: for each chunk, how many valid tokens per batch item
        chunk_starts = jnp.arange(n_chunks, dtype=jnp.int32) * chunk_size
        sequence_ends = jnp.clip(
            lengths_without_padding[None, :] - chunk_starts[:, None],
            0,
            chunk_size,
        )

        # last_token_inside: whether the last valid token (at index length-1) is in this chunk
        last_token_idx = lengths_without_padding - 1
        chunk_ends = chunk_starts + chunk_size
        is_last_token_inside = (last_token_idx[None, :] >= chunk_starts[:, None]) & (
            last_token_idx[None, :] < chunk_ends[:, None]
        )

        return Chunk(
            tokens=tokens,
            indices=indices,
            sequence_ends=sequence_ends,
            is_last_token_inside=is_last_token_inside,
        )

    @eqx.filter_jit
    def _prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        chunk_size: int = 512,  # vllm default
        state_request: StateRequest | None = None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        if state_request is None:
            state_request = StateRequest()

        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        chunks = self._make_chunks(token_ids, lengths_without_padding, chunk_size)

        num_chunks, _, chunk_size = chunks.tokens.shape
        state_capacity = max(state_capacity, num_chunks * chunk_size)

        state = self.model.init_static_state(batch_size, state_capacity)
        logits_like = jnp.zeros((batch_size, self.model.vocab_size), dtype=jnp.float32)
        hidden_dim = self.model.transformer.config.model_dim
        memory = MemoryBuffers.empty(state_request, batch_size, hidden_dim)

        def apply_chunk(state_logits_and_memory: tuple, chunk: Chunk) -> tuple:
            state, prev_logits, memory = state_logits_and_memory
            decoder_outputs = self.model(
                chunk.tokens,
                chunk.indices,
                state,
                return_updated_state=True,
                return_activation_trace=state_request.requires_activation_trace,
                lengths_without_padding=chunk.sequence_ends,
                forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_state is not None

            chunk_logits = decoder_outputs.logits[jnp.arange(batch_size), chunk.sequence_ends - 1, :]
            new_logits = jnp.where(chunk.is_last_token_inside[:, None], chunk_logits, prev_logits)
            memory = memory.append_prefill_chunk(
                state_request,
                chunk.tokens,
                decoder_outputs.activation_trace,
                chunk.sequence_ends,
            )

            return (decoder_outputs.updated_state, new_logits, memory), None

        (final_state, final_logits, final_memory), _ = jax.lax.scan(apply_chunk, (state, logits_like, memory), chunks)

        return PrefillResults(
            last_token_logits=final_logits,
            last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
            state=final_state,
            memory=final_memory,
        )

    def _prepare_decoding(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None,
        max_output_length: int,
        eos_token_ids: Int[Array, " eos_tokens"] | None,
        forward_pass_config: ForwardPassConfig | None,
        num_top_logits_to_return: int | None,
        generation_trace_config: GenerationTraceConfig | None,
        speculator: Speculator | None,
        keys: Key[Array, " batch"] | None,
    ) -> DecodingSetup:
        batch_size, sequence_length = prompt_token_ids.shape

        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy(self.model.vocab_size)

        eos_token_ids = self.resolve_eos_token_ids(generation_config, eos_token_ids)
        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=batch_size)
        if len(keys) != batch_size:
            raise ValueError(
                f"Length of 'keys' should be equal to the batch size, or keys should be None; got {len(keys)}",
            )
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")

        speculator = speculator if speculator is not None else NoSpeculator()
        state_request, traced_layers, traced_layers_array, trace_top_k = self.state_request_for_generation(
            speculator,
            generation_trace_config,
            num_top_logits_to_return,
            max_output_length,
        )

        prefill_results = self._prefill(
            prompt_token_ids,
            sequence_length + max_output_length,
            prompt_lengths_without_padding,
            forward_pass_config=forward_pass_config,
            state_request=state_request,
        )

        initial_sampling_policy = vmap(sampling_policy.init)(prompt_token_ids, prompt_lengths_without_padding)
        per_position_keys = jax.vmap(lambda key: split_into_rolling_keys(key, max_output_length + 1))(keys)
        initial_lm_state = LMState.from_prefill(
            prefill_results,
            prompt_lengths_without_padding,
            initial_sampling_policy,
            per_position_keys[:, 0],
        )
        initial_state = DecodingState(
            initial_sampling_policy,
            initial_lm_state,
        )

        def output_lengths(lm_state: LMState) -> Int[Array, " batch"]:
            assert lm_state.memory.token_ids is not None
            return lm_state.memory.token_ids.length - prompt_lengths_without_padding

        def stop_flags(lm_state: LMState) -> Bool[Array, " batch"]:
            if eos_token_ids.shape[0] == 0:
                return jnp.zeros((batch_size,), dtype=jnp.bool)
            assert lm_state.memory.token_ids is not None
            token_ids, token_mask = lm_state.memory.token_ids.window(prompt_lengths_without_padding, max_output_length)
            eos_hits = jnp.any(token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
            return jnp.any(jnp.logical_and(token_mask, eos_hits), axis=1)

        def is_done(state: DecodingState) -> Bool[Array, " batch"]:
            lm_state = state.lm_state
            current_output_lengths = output_lengths(lm_state)
            return jnp.logical_or(stop_flags(lm_state), current_output_lengths >= max_output_length)

        def loop_iteration(state: DecodingState) -> tuple[DecodingState, AcceptedProposal]:
            lm_state = state.lm_state
            current_output_lengths = output_lengths(lm_state)
            done = is_done(state)
            proposal = speculator.draft(lm_state)

            proposal_inputs = proposal.forward_inputs(lm_state.next_token_position)
            decoder_outputs = self.model(
                proposal_inputs.token_ids,
                proposal_inputs.token_positions,
                lm_state.kv_cache,
                return_updated_state=True,
                return_activation_trace=state_request.requires_activation_trace,
                lengths_without_padding=proposal_inputs.lengths_without_padding,
                forward_pass_mode=proposal_inputs.forward_pass_mode,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=proposal_inputs.attention_parent_indices,
            )
            processed_tree_logits, sampled_token_ids = proposal.sample(
                decoder_outputs.logits,
                state.sampling_policy,
                current_output_lengths,
                per_position_keys,
            )
            accepted = proposal.verify(sampled_token_ids)
            emitted_token_ids = accepted.accepted_token_ids
            accepted, write_mask = accepted.truncate(
                current_output_lengths,
                max_output_length,
                done,
                eos_token_ids,
            )

            accepted_token_logits = accepted.accepted_token_logits(processed_tree_logits, lm_state.root_sample_logits)

            sampling_top_k_ids = None
            sampling_top_k_logits = None
            if num_top_logits_to_return is not None:
                sampling_top_k_logits, sampling_top_k_ids = jax.lax.top_k(
                    accepted_token_logits,
                    num_top_logits_to_return,
                )

            trace_top_k_ids = None
            trace_top_k_logits = None
            trace_logsumexp = None
            if generation_trace_config is not None:
                trace_top_k_logits, trace_top_k_ids = jax.lax.top_k(accepted_token_logits, trace_top_k)
                trace_logsumexp = jax.nn.logsumexp(accepted_token_logits, axis=-1)

            next_sampling_policy = vmap(
                lambda policy, row_token_ids, row_mask: policy.update_many(row_token_ids, row_mask),
            )(state.sampling_policy, emitted_token_ids, write_mask)
            next_lm_state = lm_state.commit(
                state_request,
                decoder_outputs,
                processed_tree_logits,
                accepted,
                emitted_token_ids,
                sampling_top_k_ids=sampling_top_k_ids,
                sampling_top_k_logits=sampling_top_k_logits,
                trace_top_k_ids=trace_top_k_ids,
                trace_top_k_logits=trace_top_k_logits,
                logsumexp=trace_logsumexp,
            )
            next_state = DecodingState(
                next_sampling_policy,
                next_lm_state,
            )
            return next_state, accepted

        return DecodingSetup(
            initial_state=initial_state,
            state_request=state_request,
            traced_layers=traced_layers,
            traced_layers_array=traced_layers_array,
            step=loop_iteration,
            is_done=is_done,
        )

    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        num_top_logits_to_return: int | None = None,
        generation_trace_config: GenerationTraceConfig | None = None,
        speculator: Speculator | None = None,
        *,
        keys: Key[Array, " batch"] | None = None,
    ) -> GenerationResults:
        setup = self._prepare_decoding(
            prompt_token_ids,
            generation_config,
            prompt_lengths_without_padding,
            max_output_length,
            eos_token_ids,
            forward_pass_config,
            num_top_logits_to_return,
            generation_trace_config,
            speculator,
            keys,
        )
        batch_size, _ = prompt_token_ids.shape
        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jnp.full((batch_size,), prompt_token_ids.shape[1], dtype=jnp.int32)

        def scan_step(state: DecodingState, _: None) -> tuple[DecodingState, None]:
            next_state = jax.lax.cond(
                jnp.all(setup.is_done(state)),
                lambda state: state,
                lambda state: setup.step(state)[0],
                state,
            )
            return next_state, None

        final_state, _ = jax.lax.scan(scan_step, setup.initial_state, xs=None, length=max_output_length)
        final_lm_state = final_state.lm_state
        memory = final_lm_state.memory
        assert memory.token_ids is not None
        token_ids, token_mask = memory.token_ids.window(prompt_lengths_without_padding, max_output_length)
        token_ids = jnp.where(token_mask, token_ids, jnp.zeros_like(token_ids))

        if num_top_logits_to_return is not None:
            assert memory.sampling_top_k_ids is not None
            assert memory.sampling_top_k_logits is not None
            top_k_token_ids, top_k_mask = memory.sampling_top_k_ids.window(
                jnp.zeros((batch_size,), dtype=jnp.int32),
                max_output_length,
            )
            top_k_token_logits, _ = memory.sampling_top_k_logits.window(
                jnp.zeros((batch_size,), dtype=jnp.int32),
                max_output_length,
            )
            top_k_mask = top_k_mask[:, :, None]
            top_k_token_ids = jnp.where(top_k_mask, top_k_token_ids, jnp.zeros_like(top_k_token_ids))
            top_k_token_logits = jnp.where(top_k_mask, top_k_token_logits, jnp.zeros_like(top_k_token_logits))
        else:
            top_k_token_ids = None
            top_k_token_logits = None

        trace = None
        if generation_trace_config is not None:
            assert memory.trace_top_k_ids is not None
            assert memory.trace_top_k_logits is not None
            assert memory.logsumexp is not None
            assert memory.output_norm is not None
            zero_start = jnp.zeros((batch_size,), dtype=jnp.int32)
            trace_top_k_ids, trace_mask = memory.trace_top_k_ids.window(zero_start, max_output_length)
            trace_top_k_logits, _ = memory.trace_top_k_logits.window(zero_start, max_output_length)
            trace_logsumexp, _ = memory.logsumexp.window(zero_start, max_output_length)
            activation_output, activation_mask = memory.output_norm.window(
                prompt_lengths_without_padding,
                max_output_length,
            )
            trace_mask_3d = trace_mask[:, :, None]
            activation_mask_3d = activation_mask[:, :, None]
            layer_output = jnp.zeros(
                (batch_size, 0, max_output_length, self.model.transformer.config.model_dim),
                dtype=jnp.bfloat16,
            )
            if setup.traced_layers:
                layer_rows = []
                for layer_index in setup.traced_layers:
                    memory_index = setup.state_request.layer_indices.index(layer_index)
                    layer_values, layer_mask = memory.layer_outputs[memory_index].window(
                        prompt_lengths_without_padding,
                        max_output_length,
                    )
                    layer_rows.append(jnp.where(layer_mask[:, :, None], layer_values, jnp.zeros_like(layer_values)))
                layer_output = jnp.stack(layer_rows, axis=1)
            trace = StepTrace(
                top_k_ids=jnp.where(trace_mask_3d, trace_top_k_ids, jnp.zeros_like(trace_top_k_ids)),
                top_k_logits=jnp.where(trace_mask_3d, trace_top_k_logits, jnp.zeros_like(trace_top_k_logits)),
                logsumexp=jnp.where(trace_mask, trace_logsumexp, jnp.zeros_like(trace_logsumexp)),
                activation_output=jnp.where(
                    activation_mask_3d,
                    activation_output,
                    jnp.zeros_like(activation_output),
                ),
                layer_output=layer_output,
                layer_indices=setup.traced_layers_array,
            )

        return GenerationResults(token_ids, top_k_token_ids, top_k_token_logits, trace)

    def _generate_tokens_batch(
        self,
        batch: tuple[list[int], ...],
        batch_keys: tuple[Key[Array, ""], ...],
        *,
        generation_config: GenerationConfig | None,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig | None,
        generation_trace_config: GenerationTraceConfig | None,
        speculator: Speculator | None,
        sharding_config: ShardingConfig | None,
    ) -> Iterator[GenerationResults]:
        if sharding_config is None:
            sharding_config = get_current_sharding_config()

        assert inference_config.batch_size is not None
        batch_size = inference_config.batch_size

        padded_token_ids = pad_sequences(batch, (batch_size, inference_config.padded_length), dtype=jnp.int32)

        padded_lengths = jnp.array([len(tokens) for tokens in batch], dtype=jnp.int32)
        padded_lengths = jnp.pad(padded_lengths, (0, batch_size - len(batch)))

        padded_keys = pad_keys_to_size(batch_keys, batch_size)

        padded_token_ids, padded_lengths, padded_keys = pad_and_apply_data_sharding(
            (padded_token_ids, padded_lengths, padded_keys),
            sharding_config=sharding_config,
            batch_axis=0,
        )

        sharded_inference_config = replace(inference_config, batch_size=padded_token_ids.shape[0])
        generate_tokens_fn = compile_generate_tokens(
            self,
            generation_config,
            sharded_inference_config,
            forward_pass_config=forward_pass_config,
            generation_trace_config=generation_trace_config,
            speculator=speculator,
            prompt_token_ids=padded_token_ids,
            prompt_lengths_without_padding=padded_lengths,
            keys=padded_keys,
        )
        results = generate_tokens_fn(
            self,
            prompt_token_ids=padded_token_ids,
            prompt_lengths_without_padding=padded_lengths,
            keys=padded_keys,
        )
        for i in range(len(batch)):
            trace_i = None
            if results.trace is not None:
                trace_i = StepTrace(
                    top_k_ids=results.trace.top_k_ids[i],
                    top_k_logits=results.trace.top_k_logits[i],
                    logsumexp=results.trace.logsumexp[i],
                    activation_output=results.trace.activation_output[i],
                    layer_output=results.trace.layer_output[i],
                    layer_indices=results.trace.layer_indices,
                )
            yield GenerationResults(
                token_ids=results.token_ids[i],
                top_k_token_ids=results.top_k_token_ids[i] if results.top_k_token_ids is not None else None,
                top_k_token_logits=results.top_k_token_logits[i] if results.top_k_token_logits is not None else None,
                trace=trace_i,
            )

    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        generation_trace_config: GenerationTraceConfig | None = None,
        speculator: Speculator | None = None,
        sharding_config: ShardingConfig | None = None,
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
                yield from self._generate_tokens_batch(
                    real_batch,
                    batch_keys,
                    generation_config=generation_config,
                    inference_config=new_inference_config,
                    forward_pass_config=forward_pass_config,
                    generation_trace_config=generation_trace_config,
                    speculator=speculator,
                    sharding_config=sharding_config,
                )

        assert inference_config.batch_size is not None
        yield from decrease_batchsize_on_oom(
            process_batches,
            starting_batch_size=inference_config.batch_size,
        )

    def estimate_memory_consumption(
        self,
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        generation_trace_config: GenerationTraceConfig | None = None,
        speculator: Speculator | None = None,
        sharding_config: ShardingConfig | None = None,
    ) -> int:
        if sharding_config is None:
            sharding_config = get_current_sharding_config()

        assert inference_config.batch_size is not None
        batch_size = inference_config.batch_size

        dummy_token_ids = jnp.zeros((batch_size, inference_config.padded_length), dtype=jnp.int32)
        dummy_lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
        dummy_keys = jax.random.split(jax.random.key(0), num=batch_size)

        dummy_token_ids, dummy_lengths, dummy_keys = pad_and_apply_data_sharding(
            (dummy_token_ids, dummy_lengths, dummy_keys),
            sharding_config=sharding_config,
            batch_axis=0,
        )

        memory_analysis = compile_generate_tokens(
            self,
            generation_config=generation_config,
            inference_config=replace(inference_config, batch_size=dummy_token_ids.shape[0]),
            forward_pass_config=forward_pass_config,
            generation_trace_config=generation_trace_config,
            speculator=speculator,
            prompt_token_ids=dummy_token_ids,
            prompt_lengths_without_padding=dummy_lengths,
            keys=dummy_keys,
        ).memory_analysis()

        assert hasattr(memory_analysis, "argument_size_in_bytes")
        assert hasattr(memory_analysis, "output_size_in_bytes")
        assert hasattr(memory_analysis, "temp_size_in_bytes")

        return (
            memory_analysis.argument_size_in_bytes
            + memory_analysis.output_size_in_bytes
            + memory_analysis.temp_size_in_bytes
        )

    def _trim_at_eos(self, token_ids: list[int]) -> list[int]:
        if not self.stop_token_ids:
            return token_ids
        stop_set = set(self.stop_token_ids)
        end = next((i for i, token_id in enumerate(token_ids) if token_id in stop_set), len(token_ids))
        return token_ids[: end + 1]

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        speculator: Speculator | None = None,
        sharding_config: ShardingConfig | None = None,
        key: PRNGKeyArray | None = None,
    ) -> AssistantMessage:
        if sharding_config is None:
            sharding_config = get_current_sharding_config()

        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)[None, :]
        keys = key[None, ...] if key is not None else None
        token_ids, keys = pad_and_apply_data_sharding(
            (token_ids, keys),
            sharding_config=sharding_config,
            batch_axis=0,
        )
        response_ids = self.generate_tokens(
            token_ids,
            generation_config,
            forward_pass_config=forward_pass_config,
            speculator=speculator,
            keys=keys,
        ).token_ids[0]
        trimmed_ids = self._trim_at_eos(response_ids.tolist())
        response_text = self.message_processor.detokenize(trimmed_ids)
        return self.message_processor.parse_response(response_text)

    def reply_many(
        self,
        messages: Iterable[Iterable[Message]],
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        speculator: Speculator | None = None,
        sharding_config: ShardingConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
        vram_bytes: int | None = None,
        batch_sizes_callback: Callable[[BatchSizesComputedEvent], None] | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        messages = list(messages)  # eagerly load the dataset into RAM

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

        tokenized: list[list[int]] = self.message_processor.tokenize_requests(messages)

        buckets: dict[int, list[tuple[int, list[int]]]] = {}
        max_prompt_length = max(_COMPILED_PROMPT_LENGTHS)
        for idx, sequence in enumerate(tokenized):
            assert len(sequence) <= max_prompt_length, (
                f"Sequence length {len(sequence)} exceeds largest bucket {max_prompt_length}"
            )
            # we choose the smallest size from precomputed ones that is longer or equal to the current sequence
            padded_len = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= len(sequence))
            buckets.setdefault(padded_len, []).append((idx, sequence))
        sorted_lengths = sorted(buckets.keys())

        if inference_config.batch_size is not None:
            batch_size_per_bucket = dict.fromkeys(sorted_lengths, inference_config.batch_size)
        else:
            batch_size_per_bucket = estimate_batchsizes_from_vram(
                lambda config: self.estimate_memory_consumption(
                    inference_config=config,
                    speculator=speculator,
                    sharding_config=sharding_config,
                ),
                sorted_lengths,
                vram_bytes,  # type: ignore
                inference_config,
            )

        buckets = merge_small_buckets(buckets, batch_size_per_bucket, min_batches=2)
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

        # Process longest sequences first so batchsize=1 OOM happens as early as possible, if it does happen
        for padded_length in sorted(buckets.keys(), reverse=True):
            sequence_ids, sequence_tokenized = zip(*buckets[padded_length], strict=True)
            sequence_ids = list(sequence_ids)
            batch_size = batch_size_per_bucket[padded_length]

            bucket_inference_config = replace(inference_config, batch_size=batch_size, padded_length=padded_length)

            all_results = self.generate_tokens_many(
                sequence_tokenized,
                generation_config=generation_config,
                inference_config=bucket_inference_config,
                forward_pass_config=forward_pass_config,
                speculator=speculator,
                keys=keys[np.array(sequence_ids)],
                sharding_config=sharding_config,
            )

            for idx, result in zip(sequence_ids, all_results, strict=True):
                trimmed_ids = self._trim_at_eos(result.token_ids.tolist())
                response = self.message_processor.parse_tokenized_response(trimmed_ids)
                yield (idx, response)

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        speculator: Speculator | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[str]:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)
        all_token_ids: list[int] = []
        previous_text = ""
        for token_id in self.stream_tokens(
            token_ids,
            generation_config,
            max_output_length,
            forward_pass_config=forward_pass_config,
            speculator=speculator,
            key=key,
        ):
            all_token_ids.append(token_id.item())
            current_text = self.message_processor.detokenize(all_token_ids, hide_invalid_utf_chars=True)
            yield current_text[len(previous_text) :]
            previous_text = current_text

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        speculator: Speculator | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[Int[Array, ""]]:
        eos_token_ids = self.resolve_eos_token_ids(generation_config, eos_token_ids)

        (input_length,) = prompt_token_ids.shape

        padded_input_length = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length)
        padded_token_ids = jnp.zeros((padded_input_length,), dtype=jnp.int32)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)

        setup = self._prepare_decoding(
            padded_token_ids[None, :],
            generation_config,
            jnp.array([input_length], dtype=jnp.int32),
            max_output_length,
            eos_token_ids,
            forward_pass_config,
            None,
            None,
            speculator,
            key[None, ...] if key is not None else None,
        )

        state = setup.initial_state
        emitted_count = 0
        eos_token_set = {int(token_id) for token_id in np.asarray(eos_token_ids)}
        for _ in range(max_output_length):
            if bool(np.all(np.asarray(jax.device_get(setup.is_done(state))))):
                return
            state, accepted = setup.step(state)
            num_tokens = int(jax.device_get(accepted.num_compact_indices[0]))
            if num_tokens == 0:
                return
            accepted_token_ids = np.asarray(jax.device_get(accepted.accepted_token_ids[0, :num_tokens]))

            for token_id in accepted_token_ids.tolist():
                emitted_count += 1
                yield jnp.asarray(token_id, dtype=jnp.int32)
                if token_id in eos_token_set or emitted_count >= max_output_length:
                    return
