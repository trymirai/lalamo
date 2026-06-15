from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from jaxtyping import Bool, DTypeLike, Float, Int, Key, Shaped
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, Message
from lalamo.module import LogicalAxis, SpeculatorState
from lalamo.modules import (
    Decoder,
    DecoderActivationTrace,
    DecoderConfig,
    DecoderForwardPassConfig,
    DecoderResult,
    ForwardPassMode,
    Keychain,
    KeychainBroadcastMode,
)
from lalamo.modules.token_mixer import State
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.speculator import ChainProposal, NoSpeculator, Proposal, Speculator

__all__ = [
    "GenerationConfig",
    "GenerationResults",
    "LanguageModel",
    "LanguageModelConfig",
    "StreamedReply",
]


_COMPILED_PROMPT_LENGTHS = tuple(256 * 2**i for i in range(12))


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    speculator_state: SpeculatorState
    pending_activation_trace: DecoderActivationTrace | None


class Chunk(eqx.Module):
    tokens: Int[Array, "num_chunks batch chunk_size"]
    indices: Int[Array, "num_chunks batch chunk_size"]
    sequence_ends: Int[Array, "num_chunks batch"]
    is_last_token_inside: Bool[Array, "num_chunks batch"]


class DecodingState(NamedTuple):
    pending_decoder_result: DecoderResult
    pending_committed_length: Int[Array, " batch"]
    pending_proposal: Proposal
    stop_flags: Bool[Array, " batch"]
    sampling_policy: SamplingPolicy
    speculator_state: SpeculatorState
    num_generated_tokens: Int[Array, " batch"]

    @classmethod
    def from_prefill(
        cls,
        prefill_results: PrefillResults,
        sampling_policy: SamplingPolicy,
        speculator: Speculator,
        token_dtype: DTypeLike,
    ) -> "DecodingState":
        batch_size, vocab_size = prefill_results.last_token_logits.shape
        max_proposal_tokens = speculator.max_proposal_tokens
        initial_logits = jnp.zeros(
            (batch_size, max_proposal_tokens, vocab_size),
            dtype=prefill_results.last_token_logits.dtype,
        )
        initial_positions = (
            prefill_results.last_token_indices[:, None]
            + jnp.arange(
                max_proposal_tokens,
                dtype=prefill_results.last_token_indices.dtype,
            )[None, :]
        )
        return cls(
            pending_decoder_result=DecoderResult(
                logits=initial_logits.at[:, 0, :].set(prefill_results.last_token_logits),
                updated_state=prefill_results.state,
                activation_trace=prefill_results.pending_activation_trace,
            ),
            pending_committed_length=prefill_results.state.committed_length(),
            pending_proposal=ChainProposal(
                token_ids=jnp.full((batch_size, max_proposal_tokens), -1, dtype=token_dtype),
                token_positions=initial_positions,
                lengths=jnp.zeros((batch_size,), dtype=prefill_results.last_token_indices.dtype),
            ),
            stop_flags=jnp.zeros((batch_size,), dtype=bool),
            sampling_policy=sampling_policy,
            speculator_state=prefill_results.speculator_state,
            num_generated_tokens=jnp.zeros((batch_size,), dtype=jnp.int32),
        )


class GenerationStepResults(NamedTuple):
    token_ids: Int[Array, "batch accepted_tokens"]
    lengths: Int[Array, " batch"]
    top_k_token_ids: Int[Array, "batch accepted_tokens k"] | None
    top_k_token_logits: Float[Array, "batch accepted_tokens k"] | None


type GenerationLoopCarry = tuple[
    DecodingState,
    Int[Array, ""],
    Int[Array, "batch max_output"],
    Int[Array, "batch max_output k"] | None,
    Float[Array, "batch max_output k"] | None,
]


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None


class StreamedReply(NamedTuple):
    text: str
    num_tokens: int
    num_steps: int


@dataclass(frozen=True)
class GenerationConfig:
    stop_token_ids: tuple[int, ...] = ()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    suffix_repetition_length: int | None = None

    def override_with(self, other: Self) -> Self:
        default_config = GenerationConfig()
        merged_fields = {"stop_token_ids", "banned_tokens"}
        overrides = {}
        for config_field in fields(GenerationConfig):
            field_name = config_field.name
            ours = getattr(self, field_name)
            theirs = getattr(other, field_name)
            default = getattr(default_config, field_name)

            if theirs == default:
                continue
            if field_name in merged_fields and ours != default:
                overrides[field_name] = tuple(sorted({*ours, *theirs}))
            else:
                overrides[field_name] = theirs

        return replace(self, **overrides)

    def default_policy(self) -> SamplingPolicy:
        return SamplingPolicy.init(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            banned_tokens=self.banned_tokens,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            suffix_repetition_length=self.suffix_repetition_length,
        )


@dataclass(frozen=True)
class LanguageModelConfig(ModelConfig[ChatCodecConfig]):
    decoder_config: DecoderConfig
    generation_config: GenerationConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "LanguageModel":
        decoder = self.decoder_config.init(initializer)
        token_codec = self.token_codec_config.init(tokenizer)
        return LanguageModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=token_codec,
            decoder=decoder,
        )


class LanguageModel(Model[ChatCodecConfig, LanguageModelConfig, ChatCodec]):
    token_codec: ChatCodec
    decoder: Decoder

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    def trim_at_eos(self, token_ids: list[int]) -> list[int]:
        if not self.config.generation_config.stop_token_ids:
            return token_ids
        stop_token_ids = set(self.config.generation_config.stop_token_ids)
        response_length = next(
            (idx + 1 for idx, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )
        return token_ids[:response_length]

    @eqx.filter_jit
    def _make_attention_chunks(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None,
        chunk_size: int,
    ) -> Chunk:
        batch_size, sequence_length = token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        if lengths_without_padding is None:
            lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), sequence_length, dtype=jnp.int32),
                self.sharding_config.make_sharding((batch_axis,)),
            )

        chunk_size = min(chunk_size, sequence_length)
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size
        chunk_token_sharding = self.sharding_config.make_sharding((None, batch_axis, None))
        chunk_vector_sharding = self.sharding_config.make_sharding((None, batch_axis))

        tokens = rearrange(
            jnp.pad(token_ids, ((0, 0), (0, padded_length - sequence_length))),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        tokens = jax.device_put(tokens, chunk_token_sharding)
        indices = rearrange(
            jnp.broadcast_to(jnp.arange(padded_length, dtype=jnp.int32), (batch_size, padded_length)),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        indices = jax.device_put(indices, chunk_token_sharding)
        chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32) * chunk_size
        sequence_ends = jnp.clip(lengths_without_padding[None, :] - chunk_starts[:, None], 0, chunk_size)
        sequence_ends = jax.device_put(sequence_ends, chunk_vector_sharding)
        last_token_idx = lengths_without_padding - 1
        is_last_token_inside = (last_token_idx[None, :] >= chunk_starts[:, None]) & (
            last_token_idx[None, :] < chunk_starts[:, None] + chunk_size
        )
        is_last_token_inside = jax.device_put(is_last_token_inside, chunk_vector_sharding)

        return Chunk(
            tokens=tokens,
            indices=indices,
            sequence_ends=sequence_ends,
            is_last_token_inside=is_last_token_inside,
        )

    @eqx.filter_jit
    def prefill_tokens(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: DecoderForwardPassConfig | None = None,
        chunk_size: int = 512,
        *,
        speculator: Speculator,
        keychain: Keychain,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batch_vector_sharding = self.sharding_config.make_sharding((batch_axis,))
        if lengths_without_padding is None:
            lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), sequence_length, dtype=jnp.int32),
                batch_vector_sharding,
            )
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig.for_inference()

        chunks = self._make_attention_chunks(token_ids, lengths_without_padding, chunk_size)
        num_chunks, _, chunk_size = chunks.tokens.shape
        state_dtype = forward_pass_config.embedding_forward_pass_config.activation_dtype
        context_capacity = max(state_capacity, num_chunks * chunk_size)
        state = self.decoder.init_static_state(
            batch_size,
            context_capacity,
            state_dtype,
        )
        logits_like = jax.device_put(
            jnp.zeros((batch_size, self.decoder.vocab_size), dtype=jnp.float32),
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        logits_sharding = self.sharding_config.make_sharding((batch_axis, None))
        batch_indices = jax.device_put(jnp.arange(batch_size), batch_vector_sharding)
        chunk_keychains = jax.tree.map(lambda *nodes: jnp.stack(nodes), *keychain.split(num_chunks))

        def take_last_tokens(array: Shaped[Array, "batch tokens *channels"]) -> Shaped[Array, "batch out *channels"]:
            _, num_tokens, *channel_dims = array.shape
            if num_tokens >= speculator.max_proposal_tokens:
                return array[:, num_tokens - speculator.max_proposal_tokens :]
            pad_widths = ((0, 0), (speculator.max_proposal_tokens - num_tokens, 0), *((0, 0),) * len(channel_dims))
            return jnp.pad(array, pad_widths)

        def apply_chunk(
            state_speculator_state_and_logits: tuple[State, SpeculatorState, Float[Array, "batch vocabulary"]],
            chunk_inputs: tuple[Chunk, Keychain],
        ) -> tuple[
            tuple[State, SpeculatorState, Float[Array, "batch vocabulary"]],
            DecoderActivationTrace | None,
        ]:
            current_state, current_speculator_state, previous_logits = state_speculator_state_and_logits
            chunk, current_chunk_keychain = chunk_inputs
            decoder_result = self.decoder(
                token_ids=chunk.tokens,
                token_positions=chunk.indices,
                state=current_state,
                return_updated_state=True,
                return_activation_trace=speculator.requires_activation_trace,
                lengths_without_padding=chunk.sequence_ends,
                forward_pass_config=forward_pass_config,
                keychain=current_chunk_keychain,
            )
            assert decoder_result.updated_state is not None

            chunk_logits = decoder_result.logits.at[batch_indices, chunk.sequence_ends - 1, :].get(
                out_sharding=logits_sharding,
            )
            if decoder_result.activation_trace is None:
                pending_trace = None
            else:
                pending_trace = jax.tree.map(
                    take_last_tokens,
                    decoder_result.activation_trace.without_states(),
                )
            return (
                decoder_result.updated_state,
                speculator.prefill_chunk(
                    current_speculator_state,
                    decoder_result,
                    chunk.sequence_ends,
                    keychain=current_chunk_keychain,
                ),
                jnp.where(chunk.is_last_token_inside[:, None], chunk_logits, previous_logits),
            ), pending_trace

        initial_speculator_state = speculator.init_state(batch_size, context_capacity, state_dtype)
        (state, speculator_state, last_token_logits), pending_traces = jax.lax.scan(
            apply_chunk,
            (state, initial_speculator_state, logits_like),
            (chunks, chunk_keychains),
        )
        if pending_traces is None:
            pending_activation_trace = None
        else:
            pending_activation_trace = jax.tree.map(lambda array: array[-1], pending_traces)

        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
            state=state,
            speculator_state=speculator_state,
            pending_activation_trace=pending_activation_trace,
        )

    @eqx.filter_jit
    def decode_generation_step(
        self,
        state: DecodingState,
        step_keys: tuple[Key[Array, "batch nodes"], Key[Array, "..."]],
        max_output_length: int,
        eos_token_ids: Int[Array, " eos_tokens"],
        use_count_penalties: bool,
        num_top_logits_to_return: int | None,
        decode_forward_pass_config: DecoderForwardPassConfig,
        decoding_keychain: Keychain,
        speculator: Speculator,
    ) -> tuple[DecodingState, GenerationStepResults]:
        sampling_keys, decoding_key = step_keys
        batch_size, _ = state.pending_decoder_result.logits.shape[:2]
        stopped_token_ids = jnp.zeros(batch_size, dtype=jnp.int32)
        processed_logits = call_vmapped(
            lambda policy, logits: jax.vmap(
                lambda node_logits: policy.process_logits(node_logits.astype(jnp.float32)),
            )(logits),
            state.sampling_policy,
            state.pending_decoder_result.logits,
        )
        sampled_token_ids = jax.vmap(jax.vmap(jax.random.categorical))(sampling_keys, processed_logits)
        active_mask = jnp.logical_not(state.stop_flags) & (state.num_generated_tokens < max_output_length)
        sampled_token_ids = jnp.where(active_mask[:, None], sampled_token_ids, stopped_token_ids[:, None])
        remaining_budget = max_output_length - state.num_generated_tokens
        accepted = (
            state.pending_proposal.accept(sampled_token_ids).trim_at_eos(eos_token_ids).where_active(active_mask)
        )
        accepted = accepted.with_lengths(jnp.minimum(accepted.lengths, remaining_budget))

        pending_state = state.pending_decoder_result.updated_state
        assert pending_state is not None
        target_state = state.pending_proposal.rollback_state(
            pending_state,
            state.pending_committed_length,
            accepted,
        )
        current_keychain = Keychain(
            vmapped_keys=decoding_key,
            batch_key=decoding_keychain.batch_key,
            sharding_config=decoding_keychain.sharding_config,
        )
        next_speculator_state = speculator.update_state(
            state.speculator_state,
            state.pending_decoder_result,
            accepted,
            keychain=current_keychain,
        )
        next_num_generated_tokens = state.num_generated_tokens + accepted.lengths
        next_stop_flags = (
            state.stop_flags | accepted.has_eos(eos_token_ids) | (next_num_generated_tokens >= max_output_length)
        )

        last_token_ids = accepted.last_token_ids(stopped_token_ids)
        last_token_indices = accepted.last_token_indices()

        if use_count_penalties:
            next_sampling_policy = call_vmapped(
                lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
                state.sampling_policy,
                last_token_ids,
                accepted.lengths > 0,
            )
        else:
            next_sampling_policy = state.sampling_policy

        proposal = speculator.draft(
            next_speculator_state,
            last_token_ids,
            last_token_indices,
            self.decoder.embedding,
            self.decoder.embedding,
            keychain=current_keychain,
        )
        proposal_inputs = proposal.forward_inputs()
        decoder_result = self.decoder(
            token_ids=proposal_inputs.token_ids,
            token_positions=proposal_inputs.token_positions,
            state=target_state,
            return_updated_state=True,
            return_activation_trace=speculator.requires_activation_trace,
            lengths_without_padding=proposal_inputs.lengths_without_padding,
            attention_parent_indices=proposal_inputs.attention_parent_indices,
            forward_pass_config=decode_forward_pass_config,
            keychain=current_keychain,
        )
        assert decoder_result.updated_state is not None
        if decoder_result.activation_trace is None:
            pending_activation_trace = None
        else:
            pending_activation_trace = decoder_result.activation_trace.without_states()

        if num_top_logits_to_return is None:
            top_k_token_ids = None
            top_k_token_logits = None
        else:
            node_top_k_logits, node_top_k_token_ids = jax.lax.top_k(processed_logits, num_top_logits_to_return)
            top_k_token_ids, top_k_token_logits = accepted.gather_top_k(
                node_top_k_token_ids,
                node_top_k_logits,
            )

        return (
            DecodingState(
                pending_decoder_result=DecoderResult(
                    logits=decoder_result.logits,
                    updated_state=decoder_result.updated_state,
                    activation_trace=pending_activation_trace,
                ),
                pending_committed_length=target_state.committed_length(),
                pending_proposal=proposal,
                stop_flags=next_stop_flags,
                sampling_policy=next_sampling_policy,
                speculator_state=next_speculator_state,
                num_generated_tokens=next_num_generated_tokens,
            ),
            GenerationStepResults(
                token_ids=accepted.token_ids,
                lengths=accepted.lengths,
                top_k_token_ids=top_k_token_ids,
                top_k_token_logits=top_k_token_logits,
            ),
        )

    @eqx.filter_jit
    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        num_top_logits_to_return: int | None = None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
        speculator: Speculator | None = None,
    ) -> GenerationResults:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        if speculator is None:
            speculator = NoSpeculator.build(self.sharding_config)
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()
        if decode_forward_pass_config is None:
            decode_forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)

        batch_size, prompt_length = prompt_token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batch_vector_sharding = self.sharding_config.make_sharding((batch_axis,))
        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        use_count_penalties = sampling_policy.has_count_penalties
        if not isinstance(speculator, NoSpeculator) and use_count_penalties:
            raise ValueError("Speculator decoding only supports stateless sampling policies.")
        sampling_policy = sampling_policy.broadcast(batch_size)

        def shard_sampling_leaf(leaf: object) -> object:
            if not isinstance(leaf, jax.Array):
                return leaf
            if leaf.ndim > 0 and leaf.shape[0] == batch_size:
                sharding_axes = (batch_axis, *((None,) * (leaf.ndim - 1)))
            else:
                sharding_axes = (None,) * leaf.ndim
            leaf_sharding = self.sharding_config.make_sharding(sharding_axes)
            return jax.device_put(leaf, leaf_sharding)

        sampling_policy = jax.tree.map(shard_sampling_leaf, sampling_policy)
        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), prompt_length, dtype=jnp.int32),
                batch_vector_sharding,
            )
        if use_count_penalties:
            sampling_policy = call_vmapped(
                lambda policy, prompt_token_ids, prompt_length: policy.with_prompt_token_counts(
                    prompt_token_ids,
                    prompt_length,
                    self.decoder.vocab_size,
                ),
                sampling_policy,
                prompt_token_ids,
                prompt_lengths_without_padding,
            )
        if eos_token_ids is None:
            eos_token_ids = jax.device_put(
                jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32),
                self.sharding_config.make_sharding((None,)),
            )

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            prompt_token_ids,
            prompt_length + max_output_length + speculator.max_proposal_tokens,
            prompt_lengths_without_padding,
            prefill_forward_pass_config,
            keychain=prefill_keychain,
            speculator=speculator,
        )

        max_proposal_tokens = speculator.max_proposal_tokens
        batch_indices = jax.device_put(jnp.arange(batch_size, dtype=jnp.int32), batch_vector_sharding)
        initial_state = DecodingState.from_prefill(
            prefill_results,
            sampling_policy,
            speculator,
            prompt_token_ids.dtype,
        )
        initial_state = initial_state._replace(
            stop_flags=jax.device_put(initial_state.stop_flags, batch_vector_sharding),
            num_generated_tokens=jax.device_put(initial_state.num_generated_tokens, batch_vector_sharding),
        )

        sampling_keys = rearrange(
            sampling_keychain.rolling_broadcast(
                (max_output_length, max_proposal_tokens, batch_size),
                mode=KeychainBroadcastMode.PREFIX,
            ).vmapped_keys,
            "step nodes batch -> step batch nodes",
        )
        sampling_keys = jax.device_put(
            sampling_keys,
            self.sharding_config.make_sharding((None, batch_axis, None)),
        )
        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, *decoding_keychain.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys

        proposal_slots = jnp.arange(max_proposal_tokens, dtype=jnp.int32)
        initial_token_ids = jnp.zeros((batch_size, max_output_length), dtype=prompt_token_ids.dtype)
        if num_top_logits_to_return is None:
            initial_top_k_token_ids = None
            initial_top_k_token_logits = None
        else:
            initial_top_k_token_ids = jnp.zeros(
                (batch_size, max_output_length, num_top_logits_to_return),
                dtype=jnp.int32,
            )
            initial_top_k_token_logits = jnp.zeros(
                (batch_size, max_output_length, num_top_logits_to_return),
                dtype=jnp.float32,
            )

        def loop_condition(carry: GenerationLoopCarry) -> Bool[Array, ""]:
            state, step_index, *_ = carry
            return (step_index < max_output_length) & jnp.logical_not(jnp.all(state.stop_flags))

        def loop_body(carry: GenerationLoopCarry) -> GenerationLoopCarry:
            state, step_index, token_ids, top_k_token_ids, top_k_token_logits = carry
            step_keys = (sampling_keys[step_index], decoding_keys[step_index])
            output_offsets = state.num_generated_tokens
            state, step_results = self.decode_generation_step(
                state,
                step_keys,
                max_output_length,
                eos_token_ids,
                use_count_penalties,
                num_top_logits_to_return,
                decode_forward_pass_config,
                decoding_keychain,
                speculator,
            )
            output_positions = jnp.where(
                proposal_slots[None, :] < step_results.lengths[:, None],
                output_offsets[:, None] + proposal_slots[None, :],
                max_output_length,
            )
            token_ids = token_ids.at[batch_indices[:, None], output_positions].set(
                step_results.token_ids,
                mode="drop",
            )
            if num_top_logits_to_return is not None:
                assert step_results.top_k_token_ids is not None
                assert step_results.top_k_token_logits is not None
                assert top_k_token_ids is not None
                assert top_k_token_logits is not None
                top_k_token_ids = top_k_token_ids.at[batch_indices[:, None], output_positions, :].set(
                    step_results.top_k_token_ids,
                    mode="drop",
                )
                top_k_token_logits = top_k_token_logits.at[batch_indices[:, None], output_positions, :].set(
                    step_results.top_k_token_logits,
                    mode="drop",
                )
            return (state, step_index + 1, token_ids, top_k_token_ids, top_k_token_logits)

        (_, _, token_ids, top_k_token_ids, top_k_token_logits) = jax.lax.while_loop(
            loop_condition,
            loop_body,
            (
                initial_state,
                jnp.zeros((), dtype=jnp.int32),
                initial_token_ids,
                initial_top_k_token_ids,
                initial_top_k_token_logits,
            ),
        )

        return GenerationResults(
            token_ids=token_ids,
            top_k_token_ids=top_k_token_ids,
            top_k_token_logits=top_k_token_logits,
        )

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
    ) -> AssistantMessage:
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        token_ids = jax.device_put(
            jnp.asarray(self.token_codec.encode_request(messages), dtype=jnp.int32)[None, :],
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        response_ids = self.generate_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            keychain=keychain,
        ).token_ids[0]
        return self.token_codec.decode_response(self.trim_at_eos(response_ids.tolist()))

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
        speculator: Speculator | None = None,
    ) -> Iterable[Int[np.ndarray, ""]]:
        for block_token_ids in self.stream_token_blocks(
            prompt_token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            eos_token_ids=eos_token_ids,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            keychain=keychain,
            speculator=speculator,
        ):
            yield from block_token_ids

    def stream_token_blocks(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
        speculator: Speculator | None = None,
    ) -> Iterable[Int[np.ndarray, " block_tokens"]]:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        if speculator is None:
            speculator = NoSpeculator.build(self.sharding_config)
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()
        if decode_forward_pass_config is None:
            decode_forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)

        if eos_token_ids is not None:
            stop_token_ids = eos_token_ids
        else:
            stop_token_ids = jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape
        padded_input_length = next(
            (length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length),
            None,
        )
        if padded_input_length is None:
            raise ValueError(f"Input sequence length {input_length} exceeds largest compiled prompt bucket.")

        padded_token_ids = jnp.zeros((padded_input_length,), dtype=prompt_token_ids.dtype)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batched_token_ids = jax.device_put(
            padded_token_ids[None, :],
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        length_without_padding = jax.device_put(
            jnp.asarray([input_length], dtype=jnp.int32),
            self.sharding_config.make_sharding((batch_axis,)),
        )

        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        use_count_penalties = sampling_policy.has_count_penalties
        if not isinstance(speculator, NoSpeculator) and use_count_penalties:
            raise ValueError("Speculator decoding only supports stateless sampling policies.")
        if use_count_penalties:
            sampling_policy = sampling_policy.with_prompt_token_counts(
                padded_token_ids,
                jnp.asarray(input_length, dtype=jnp.int32),
                self.decoder.vocab_size,
            )
        sampling_policy = sampling_policy.broadcast(1)

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            batched_token_ids,
            padded_input_length + max_output_length + speculator.max_proposal_tokens,
            length_without_padding,
            prefill_forward_pass_config,
            keychain=prefill_keychain,
            speculator=speculator,
        )

        max_proposal_tokens = speculator.max_proposal_tokens
        state = DecodingState.from_prefill(prefill_results, sampling_policy, speculator, prompt_token_ids.dtype)
        sampling_keys = rearrange(
            sampling_keychain.rolling_broadcast(
                (max_output_length, max_proposal_tokens, 1),
                mode=KeychainBroadcastMode.PREFIX,
            ).vmapped_keys,
            "step nodes batch -> step batch nodes",
        )
        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, *decoding_keychain.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        host_stop_token_ids = np.asarray(stop_token_ids)

        for sampling_key, decoding_key in zip(sampling_keys, decoding_keys, strict=True):
            state, generated = self.decode_generation_step(
                state,
                (sampling_key, decoding_key),
                max_output_length,
                stop_token_ids,
                use_count_penalties,
                None,
                decode_forward_pass_config,
                decoding_keychain,
                speculator,
            )
            num_tokens, step_token_ids, stopped = jax.device_get(
                (generated.lengths[0], generated.token_ids[0], state.stop_flags[0]),
            )
            block_token_ids = step_token_ids[:num_tokens]
            eos_hits = np.isin(block_token_ids, host_stop_token_ids)
            if eos_hits.any():
                yield block_token_ids[: int(np.argmax(eos_hits)) + 1]
                return
            yield block_token_ids
            if stopped:
                return

    def stream_reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
        speculator: Speculator | None = None,
        enable_thinking: bool = True,
    ) -> Iterable[StreamedReply]:
        token_ids = jnp.asarray(
            self.token_codec.encode_request(messages, enable_thinking=enable_thinking),
            dtype=jnp.int32,
        )
        response_token_ids: list[int] = []
        for num_steps, block_token_ids in enumerate(
            self.stream_token_blocks(
                token_ids,
                generation_config=generation_config,
                max_output_length=max_output_length,
                prefill_forward_pass_config=prefill_forward_pass_config,
                decode_forward_pass_config=decode_forward_pass_config,
                keychain=keychain,
                speculator=speculator,
            ),
            start=1,
        ):
            response_token_ids.extend(int(token_id) for token_id in block_token_ids)
            text = self.token_codec.decode_tokens(response_token_ids, hide_invalid_utf_chars=True)
            yield StreamedReply(text=text, num_tokens=len(response_token_ids), num_steps=num_steps)
