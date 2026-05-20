from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import Array
from jaxtyping import Bool, Float, Int, Key
from tokenizers import Tokenizer

from lalamo.data.completion_features import FeatureRequest
from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, Message
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    Keychain,
    KeychainBroadcastMode,
)
from lalamo.modules.token_mixer import State
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.common import NoSpeculator, Speculator
from lalamo.speculator.state import LMState, PrefillResults

__all__ = [
    "GenerationConfig",
    "GenerationResults",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = tuple(256 * 2**i for i in range(12))


class Chunk(eqx.Module):
    tokens: Int[Array, "num_chunks batch chunk_size"]
    indices: Int[Array, "num_chunks batch chunk_size"]
    sequence_ends: Int[Array, "num_chunks batch"]
    is_last_token_inside: Bool[Array, "num_chunks batch"]


class GenerationStepResults(NamedTuple):
    token_ids: Int[Array, "batch step_tokens"]
    num_token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, "batch step_tokens k"] | None
    top_k_token_logits: Float[Array, "batch step_tokens k"] | None


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None
    num_tokens_per_step: Int[Array, "steps batch"]

    @staticmethod
    def flatten_token_ids(
        token_ids: Int[Array, "steps batch step_tokens"],
        num_token_ids: Int[Array, "steps batch"],
        max_output_length: int,
    ) -> Int[Array, "batch response_tokens"]:
        _, batch_size, step_tokens = token_ids.shape
        offsets = jnp.cumsum(num_token_ids, axis=0) - num_token_ids
        slots = jnp.arange(step_tokens, dtype=jnp.int32)[None, None, :]
        positions = offsets[:, :, None] + slots
        valid = (slots < num_token_ids[:, :, None]) & (positions < max_output_length)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[None, :, None]
        return jnp.zeros((batch_size, max_output_length), dtype=token_ids.dtype).at[batch_indices, positions].add(
            jnp.where(valid, token_ids, 0),
            mode="drop",
        )

    @staticmethod
    def flatten_top_k(
        values: Int[Array, "steps batch step_tokens k"] | Float[Array, "steps batch step_tokens k"],
        num_token_ids: Int[Array, "steps batch"],
        max_output_length: int,
    ) -> Int[Array, "batch response_tokens k"] | Float[Array, "batch response_tokens k"]:
        _, batch_size, step_tokens, top_k = values.shape
        offsets = jnp.cumsum(num_token_ids, axis=0) - num_token_ids
        slots = jnp.arange(step_tokens, dtype=jnp.int32)[None, None, :]
        positions = offsets[:, :, None] + slots
        valid = (slots < num_token_ids[:, :, None]) & (positions < max_output_length)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[None, :, None, None]
        top_k_indices = jnp.arange(top_k, dtype=jnp.int32)[None, None, None, :]
        return jnp.zeros((batch_size, max_output_length, top_k), dtype=values.dtype).at[
            batch_indices,
            positions[:, :, :, None],
            top_k_indices,
        ].add(
            jnp.where(valid[:, :, :, None], values, 0),
            mode="drop",
        )

    @classmethod
    def from_steps(
        cls,
        generated: GenerationStepResults,
        max_output_length: int,
        num_top_logits_to_return: int | None,
    ) -> "GenerationResults":
        token_ids = cls.flatten_token_ids(generated.token_ids, generated.num_token_ids, max_output_length)
        if num_top_logits_to_return is None:
            top_k_token_ids = None
            top_k_token_logits = None
        else:
            assert generated.top_k_token_ids is not None
            assert generated.top_k_token_logits is not None
            top_k_token_ids = cls.flatten_top_k(
                generated.top_k_token_ids,
                generated.num_token_ids,
                max_output_length,
            )
            top_k_token_logits = cls.flatten_top_k(
                generated.top_k_token_logits,
                generated.num_token_ids,
                max_output_length,
            )
        return cls(
            token_ids=token_ids,
            top_k_token_ids=top_k_token_ids,
            top_k_token_logits=top_k_token_logits,
            num_tokens_per_step=generated.num_token_ids,
        )


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
        return LanguageModel(self, token_codec, decoder)


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
        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        chunk_size = min(chunk_size, sequence_length)
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size

        tokens = rearrange(
            jnp.pad(token_ids, ((0, 0), (0, padded_length - sequence_length))),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        indices = rearrange(
            jnp.broadcast_to(jnp.arange(padded_length, dtype=jnp.int32), (batch_size, padded_length)),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32) * chunk_size
        sequence_ends = jnp.clip(lengths_without_padding[None, :] - chunk_starts[:, None], 0, chunk_size)
        last_token_idx = lengths_without_padding - 1

        return Chunk(
            tokens=tokens,
            indices=indices,
            sequence_ends=sequence_ends,
            is_last_token_inside=(last_token_idx[None, :] >= chunk_starts[:, None])
            & (last_token_idx[None, :] < chunk_starts[:, None] + chunk_size),
        )

    @eqx.filter_jit
    def prefill_tokens(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: DecoderForwardPassConfig | None = None,
        chunk_size: int = 512,
        feature_request: FeatureRequest | None = None,
        *,
        keychain: Keychain,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig.for_inference()
        return_activation_trace = (
            feature_request is not None and (feature_request.output_features or bool(feature_request.layer_indices))
        )
        input_token_ids = jnp.pad(token_ids, ((0, 0), (0, max(state_capacity - sequence_length, 0))))

        if return_activation_trace:
            state_dtype = forward_pass_config.embedding_forward_pass_config.activation_dtype
            state = self.decoder.init_static_state(
                batch_size,
                max(state_capacity, sequence_length),
                state_dtype,
            )
            token_positions = jnp.broadcast_to(
                jnp.arange(sequence_length, dtype=jnp.int32),
                (batch_size, sequence_length),
            )
            decoder_result = self.decoder(
                token_ids=token_ids,
                token_positions=token_positions,
                state=state,
                return_updated_state=True,
                return_activation_trace=True,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=forward_pass_config,
                keychain=keychain,
            )
            assert decoder_result.updated_state is not None
            return PrefillResults(
                last_token_logits=decoder_result.logits[jnp.arange(batch_size), lengths_without_padding - 1, :],
                last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
                state=decoder_result.updated_state,
                input_token_ids=input_token_ids,
                input_lengths=lengths_without_padding,
                activation_trace=decoder_result.activation_trace,
            )

        chunks = self._make_attention_chunks(token_ids, lengths_without_padding, chunk_size)
        num_chunks, _, chunk_size = chunks.tokens.shape
        state_dtype = forward_pass_config.embedding_forward_pass_config.activation_dtype
        state = self.decoder.init_static_state(
            batch_size,
            max(state_capacity, num_chunks * chunk_size),
            state_dtype,
        )
        logits_like = jnp.zeros((batch_size, self.decoder.vocab_size), dtype=jnp.float32)
        chunk_keychains = jax.tree.map(lambda *nodes: jnp.stack(nodes), *keychain.split(num_chunks))

        def apply_chunk(
            state_and_logits: tuple[State, Float[Array, "batch vocabulary"]],
            chunk_inputs: tuple[Chunk, Keychain],
        ) -> tuple[tuple[State, Float[Array, "batch vocabulary"]], None]:
            current_state, previous_logits = state_and_logits
            chunk, current_chunk_keychain = chunk_inputs
            decoder_result = self.decoder(
                token_ids=chunk.tokens,
                token_positions=chunk.indices,
                state=current_state,
                return_updated_state=True,
                lengths_without_padding=chunk.sequence_ends,
                forward_pass_config=forward_pass_config,
                keychain=current_chunk_keychain,
            )
            assert decoder_result.updated_state is not None

            chunk_logits = decoder_result.logits[jnp.arange(batch_size), chunk.sequence_ends - 1, :]
            return (
                decoder_result.updated_state,
                jnp.where(chunk.is_last_token_inside[:, None], chunk_logits, previous_logits),
            ), None

        (state, last_token_logits), _ = jax.lax.scan(
            apply_chunk,
            (state, logits_like),
            (chunks, chunk_keychains),
        )

        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
            state=state,
            input_token_ids=input_token_ids,
            input_lengths=lengths_without_padding,
        )

    def resolve_eos_token_ids(
        self,
        generation_config: GenerationConfig | None,
        eos_token_ids: Int[Array, " eos_tokens"] | None,
    ) -> Int[Array, " eos_tokens"]:
        if eos_token_ids is not None:
            return eos_token_ids
        if generation_config is not None and generation_config.stop_token_ids:
            return jnp.asarray(generation_config.stop_token_ids, dtype=jnp.int32)
        return jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32)

    def _init_sampling_policy(
        self,
        generation_config: GenerationConfig | None,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        prompt_lengths_without_padding: Int[Array, " batch"],
    ) -> SamplingPolicy:
        batch_size, _ = prompt_token_ids.shape
        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        use_count_penalties = sampling_policy.has_count_penalties
        sampling_policy = sampling_policy.broadcast(batch_size)
        if not use_count_penalties:
            return sampling_policy
        return call_vmapped(
            lambda policy, token_ids, prompt_length: policy.with_prompt_token_counts(
                token_ids,
                prompt_length,
                self.decoder.vocab_size,
            ),
            sampling_policy,
            prompt_token_ids,
            prompt_lengths_without_padding,
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
        speculator: Speculator | None = None,
        *,
        keychain: Keychain,
    ) -> GenerationResults:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()

        active_speculator: Speculator = speculator if speculator is not None else NoSpeculator()
        batch_size, prompt_length = prompt_token_ids.shape
        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jnp.full((batch_size,), prompt_length, dtype=jnp.int32)

        eos_token_ids = self.resolve_eos_token_ids(generation_config, eos_token_ids)
        sampling_policy_template = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy_template = generation_config.default_policy()
        if (
            not isinstance(active_speculator, NoSpeculator)
            and sampling_policy_template.has_unsupported_tree_verification_processors
        ):
            raise ValueError(
                "Speculative decoding only supports temperature, top_k, top_p, and min_p sampling processors.",
            )
        sampling_policy = self._init_sampling_policy(
            generation_config,
            prompt_token_ids,
            prompt_lengths_without_padding,
        )
        feature_request = active_speculator.feature_request

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            prompt_token_ids,
            prompt_length + max_output_length + 1,
            prompt_lengths_without_padding,
            prefill_forward_pass_config,
            feature_request=feature_request,
            keychain=prefill_keychain,
        )
        sampling_keys = sampling_keychain.rolling_broadcast(
            (batch_size,),
            mode=KeychainBroadcastMode.SUFFIX,
        ).vmapped_keys
        initial_lm_state = active_speculator.init_state(
            prefill_results,
            prompt_lengths_without_padding,
            sampling_policy,
            sampling_keys,
        )
        max_step_tokens = active_speculator.max_step_tokens

        def is_done(lm_state: LMState) -> Bool[Array, " batch"]:
            return lm_state.stop_flags | (lm_state.output_lengths >= max_output_length)

        def stop_flags_after_step(
            lm_state: LMState,
            token_ids: Int[Array, "batch step_tokens"],
            write_mask: Bool[Array, "batch step_tokens"],
        ) -> Bool[Array, " batch"]:
            if eos_token_ids.shape[0] == 0:
                return lm_state.stop_flags
            eos_hits = jnp.any(token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
            return lm_state.stop_flags | jnp.any(write_mask & eos_hits, axis=1)

        def decode_step(
            lm_state: LMState,
            decoding_key: Key[Array, "..."],
        ) -> tuple[LMState, GenerationStepResults]:
            proposal = active_speculator.draft(lm_state)
            proposal_inputs = proposal.forward_inputs(lm_state.next_token_position)
            forward_pass_config = decode_forward_pass_config
            if forward_pass_config is None:
                forward_pass_config = DecoderForwardPassConfig.for_inference(proposal_inputs.forward_pass_mode)
            decoder_result = self.decoder(
                token_ids=proposal_inputs.token_ids,
                token_positions=proposal_inputs.token_positions,
                state=lm_state.kv_cache,
                return_updated_state=True,
                return_activation_trace=feature_request.output_features or bool(feature_request.layer_indices),
                lengths_without_padding=proposal_inputs.lengths_without_padding,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=proposal_inputs.attention_parent_indices,
                keychain=Keychain(vmapped_keys=decoding_key, batch_key=decoding_keychain.batch_key),
            )
            processed_tree_logits, sampled_token_ids, next_sampling_policies = proposal.sample(decoder_result.logits)
            accepted = proposal.verify(sampled_token_ids, next_sampling_policies)
            accepted, write_mask = accepted.truncate(
                lm_state.output_lengths,
                max_output_length,
                is_done(lm_state),
                eos_token_ids,
            )
            next_stop_flags = stop_flags_after_step(lm_state, accepted.accepted_token_ids, write_mask)

            if num_top_logits_to_return is None:
                top_k_token_ids = None
                top_k_token_logits = None
            else:
                accepted_token_logits = accepted.accepted_token_logits(
                    processed_tree_logits,
                    lm_state.root_sample_logits,
                )
                top_k_token_logits, top_k_token_ids = jax.lax.top_k(
                    accepted_token_logits,
                    num_top_logits_to_return,
                )

            next_lm_state = lm_state.commit(
                decoder_result,
                processed_tree_logits,
                accepted,
                next_stop_flags,
                feature_request,
            )
            return next_lm_state, GenerationStepResults(
                token_ids=jnp.where(write_mask, accepted.accepted_token_ids, 0),
                num_token_ids=accepted.num_compact_indices,
                top_k_token_ids=top_k_token_ids,
                top_k_token_logits=top_k_token_logits,
            )

        def no_op_step(
            lm_state: LMState,
            decoding_key: Key[Array, "..."],
        ) -> tuple[LMState, GenerationStepResults]:
            del decoding_key
            token_ids = jnp.zeros((batch_size, max_step_tokens), dtype=jnp.int32)
            if num_top_logits_to_return is None:
                top_k_token_ids = None
                top_k_token_logits = None
            else:
                top_k_token_ids = jnp.zeros((batch_size, max_step_tokens, num_top_logits_to_return), dtype=jnp.int32)
                top_k_token_logits = jnp.zeros(
                    (batch_size, max_step_tokens, num_top_logits_to_return),
                    dtype=jnp.float32,
                )
            return lm_state, GenerationStepResults(
                token_ids=token_ids,
                num_token_ids=jnp.zeros((batch_size,), dtype=jnp.int32),
                top_k_token_ids=top_k_token_ids,
                top_k_token_logits=top_k_token_logits,
            )

        def scan_step(
            lm_state: LMState,
            decoding_key: Key[Array, "..."],
        ) -> tuple[LMState, GenerationStepResults]:
            return jax.lax.cond(
                jnp.all(is_done(lm_state)),
                no_op_step,
                decode_step,
                lm_state,
                decoding_key,
            )

        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, *decoding_keychain.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        _, generated = jax.lax.scan(
            scan_step,
            initial_lm_state,
            decoding_keys,
        )

        return GenerationResults.from_steps(generated, max_output_length, num_top_logits_to_return)

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        speculator: Speculator | None = None,
        *,
        keychain: Keychain,
    ) -> AssistantMessage:
        token_ids = jnp.asarray(self.token_codec.encode_request(messages), dtype=jnp.int32)[None, :]
        response_ids = self.generate_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            speculator=speculator,
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
        speculator: Speculator | None = None,
        step_callback: Callable[[int], None] | None = None,
        *,
        keychain: Keychain,
    ) -> Iterable[Int[Array, ""]]:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()

        (input_length,) = prompt_token_ids.shape
        padded_input_length = next(
            (length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length),
            None,
        )
        if padded_input_length is None:
            raise ValueError(f"Input sequence length {input_length} exceeds largest compiled prompt bucket.")

        padded_token_ids = jnp.zeros((padded_input_length,), dtype=prompt_token_ids.dtype)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)
        length_without_padding = jnp.asarray([input_length], dtype=jnp.int32)
        active_speculator: Speculator = speculator if speculator is not None else NoSpeculator()
        sampling_policy_template = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy_template = generation_config.default_policy()
        if (
            not isinstance(active_speculator, NoSpeculator)
            and sampling_policy_template.has_unsupported_tree_verification_processors
        ):
            raise ValueError(
                "Speculative decoding only supports temperature, top_k, top_p, and min_p sampling processors.",
            )
        sampling_policy = self._init_sampling_policy(
            generation_config,
            padded_token_ids[None, :],
            length_without_padding,
        )
        feature_request = active_speculator.feature_request

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            padded_token_ids[None, :],
            padded_input_length + max_output_length + 1,
            length_without_padding,
            prefill_forward_pass_config,
            feature_request=feature_request,
            keychain=prefill_keychain,
        )
        sampling_keys = sampling_keychain.rolling_broadcast(
            (1,),
            mode=KeychainBroadcastMode.SUFFIX,
        ).vmapped_keys
        state = active_speculator.init_state(
            prefill_results,
            length_without_padding,
            sampling_policy,
            sampling_keys,
        )
        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, *decoding_keychain.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        stop_token_ids = self.resolve_eos_token_ids(generation_config, eos_token_ids)
        stop_token_id_set = set(jax.device_get(stop_token_ids).tolist())

        def is_done(lm_state: LMState) -> Bool[Array, " batch"]:
            return lm_state.stop_flags | (lm_state.output_lengths >= max_output_length)

        def stop_flags_after_step(
            lm_state: LMState,
            token_ids: Int[Array, "batch step_tokens"],
            write_mask: Bool[Array, "batch step_tokens"],
        ) -> Bool[Array, " batch"]:
            if stop_token_ids.shape[0] == 0:
                return lm_state.stop_flags
            eos_hits = jnp.any(token_ids[:, :, None] == stop_token_ids[None, None, :], axis=-1)
            return lm_state.stop_flags | jnp.any(write_mask & eos_hits, axis=1)

        emitted_count = 0
        for decoding_key in decoding_keys:
            if bool(jax.device_get(jnp.all(is_done(state)))):
                return

            proposal = active_speculator.draft(state)
            proposal_inputs = proposal.forward_inputs(state.next_token_position)
            forward_pass_config = decode_forward_pass_config
            if forward_pass_config is None:
                forward_pass_config = DecoderForwardPassConfig.for_inference(proposal_inputs.forward_pass_mode)
            decoder_result = self.decoder(
                token_ids=proposal_inputs.token_ids,
                token_positions=proposal_inputs.token_positions,
                state=state.kv_cache,
                return_updated_state=True,
                return_activation_trace=feature_request.output_features or bool(feature_request.layer_indices),
                lengths_without_padding=proposal_inputs.lengths_without_padding,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=proposal_inputs.attention_parent_indices,
                keychain=Keychain(vmapped_keys=decoding_key, batch_key=decoding_keychain.batch_key),
            )
            processed_tree_logits, sampled_token_ids, next_sampling_policies = proposal.sample(decoder_result.logits)
            accepted = proposal.verify(sampled_token_ids, next_sampling_policies)
            accepted, write_mask = accepted.truncate(
                state.output_lengths,
                max_output_length,
                is_done(state),
                stop_token_ids,
            )
            state = state.commit(
                decoder_result,
                processed_tree_logits,
                accepted,
                stop_flags_after_step(state, accepted.accepted_token_ids, write_mask),
                feature_request,
            )
            num_tokens = int(jax.device_get(accepted.num_compact_indices[0]))
            if num_tokens == 0:
                return
            if step_callback is not None:
                step_callback(num_tokens)

            token_ids = jax.device_get(
                jnp.where(write_mask, accepted.accepted_token_ids, 0)[0, :num_tokens],
            ).tolist()
            for token_id in token_ids:
                emitted_count += 1
                yield jnp.asarray(token_id, dtype=jnp.int32)
                if token_id in stop_token_id_set or emitted_count >= max_output_length:
                    return

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        speculator: Speculator | None = None,
        step_callback: Callable[[int], None] | None = None,
        *,
        keychain: Keychain,
    ) -> Iterable[str]:
        token_ids = jnp.asarray(self.token_codec.encode_request(messages), dtype=jnp.int32)
        response_token_ids: list[int] = []
        previous_text = ""
        for token_id in self.stream_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            speculator=speculator,
            step_callback=step_callback,
            keychain=keychain,
        ):
            response_token_ids.append(int(token_id.item()))
            current_text = self.token_codec.decode_tokens(response_token_ids, hide_invalid_utf_chars=True)
            yield current_text[len(previous_text) :]
            previous_text = current_text
