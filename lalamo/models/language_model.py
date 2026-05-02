from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import Array
from jaxtyping import Bool, Float, Int, Key
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, Message
from lalamo.modules import Decoder, DecoderConfig, DecoderForwardPassConfig, ForwardPassMode, Keychain
from lalamo.modules.token_mixer import State
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy

__all__ = [
    "GenerationConfig",
    "GenerationResults",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = tuple(256 * 2**i for i in range(12))


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    stop_flags: Bool[Array, " batch"]
    sampling_policy: SamplingPolicy


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
    stop_token_ids: tuple[int, ...] = ()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def default_policy(self) -> SamplingPolicy:
        return SamplingPolicy.init(
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.banned_tokens,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
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

    def _trim_at_eos(self, token_ids: list[int]) -> list[int]:
        if not self.config.generation_config.stop_token_ids:
            return token_ids
        stop_token_ids = set(self.config.generation_config.stop_token_ids)
        response_length = next(
            (idx + 1 for idx, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )
        return token_ids[:response_length]

    def _prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        *,
        keychain: Keychain,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        token_positions = jnp.broadcast_to(
            jnp.arange(sequence_length, dtype=jnp.int32),
            (batch_size, sequence_length),
        )
        state = self.decoder.init_static_state(
            batch_size,
            state_capacity,
            self.decoder.embedding.embedding_matrix.dtype,
        )
        decoder_result = self.decoder(
            token_ids=token_ids,
            token_positions=token_positions,
            state=state,
            return_updated_state=True,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=DecoderForwardPassConfig.for_inference(),
            keychain=keychain,
        )
        assert decoder_result.updated_state is not None

        last_token_indices = jnp.maximum(lengths_without_padding - 1, 0)
        last_token_logits = decoder_result.logits[jnp.arange(batch_size), last_token_indices, :]
        return PrefillResults(
            last_token_logits=last_token_logits.astype(jnp.float32),
            last_token_indices=last_token_indices,
            state=decoder_result.updated_state,
        )

    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        num_top_logits_to_return: int | None = None,
        *,
        keychain: Keychain,
    ) -> GenerationResults:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")

        batch_size, prompt_length = prompt_token_ids.shape
        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        use_count_penalties = sampling_policy.has_count_penalties()
        sampling_policy = sampling_policy.broadcast(batch_size)
        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jnp.full((batch_size,), prompt_length, dtype=jnp.int32)
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
            eos_token_ids = jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32)

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self._prefill(
            prompt_token_ids,
            prompt_length + max_output_length + 1,
            prompt_lengths_without_padding,
            keychain=prefill_keychain,
        )
        initial_state = DecodingState(
            last_token_logits=prefill_results.last_token_logits,
            last_token_indices=prefill_results.last_token_indices,
            state=prefill_results.state,
            stop_flags=jnp.zeros(batch_size, dtype=jnp.bool_),
            sampling_policy=sampling_policy,
        )

        def sample_token(
            logits: Float[Array, " vocabulary"],
            sample_key: Key[Array, ""],
        ) -> Int[Array, ""]:
            return jax.random.categorical(sample_key, logits)

        def loop_iteration(
            state: DecodingState,
            step_keys: tuple[Key[Array, " batch"], Key[Array, ""]],
        ) -> tuple[DecodingState, GenerationStepResults]:
            sampling_keys, decoding_key = step_keys
            processed_logits = call_vmapped(
                lambda policy, logits: policy.process_logits(logits),
                state.sampling_policy,
                state.last_token_logits.astype(jnp.float32),
            )
            next_token_ids = call_vmapped(sample_token, processed_logits, sampling_keys)
            next_token_ids = jnp.where(state.stop_flags, jnp.zeros(batch_size, dtype=jnp.int32), next_token_ids)
            next_sampling_policy = call_vmapped(
                lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
                state.sampling_policy,
                next_token_ids,
                jnp.logical_not(state.stop_flags),
            )

            if num_top_logits_to_return is None:
                top_k_token_ids = None
                top_k_token_logits = None
            else:
                top_k_token_logits, top_k_token_ids = jax.lax.top_k(processed_logits, num_top_logits_to_return)

            next_token_indices = state.last_token_indices + 1
            next_stop_flags = state.stop_flags | jnp.any(next_token_ids[:, None] == eos_token_ids[None, :], axis=-1)
            forward_pass_mode = ForwardPassMode.MULTI_TOKEN
            if batch_size == 1:
                forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
            decoder_result = self.decoder(
                token_ids=next_token_ids[:, None],
                token_positions=next_token_indices[:, None],
                state=state.state,
                return_updated_state=True,
                forward_pass_config=DecoderForwardPassConfig.for_inference(forward_pass_mode),
                keychain=Keychain(vmapped_keys=decoding_key, batch_key=decoding_keychain.batch_key),
            )
            assert decoder_result.updated_state is not None

            return (
                DecodingState(
                    last_token_logits=decoder_result.logits[:, 0, :].astype(jnp.float32),
                    last_token_indices=next_token_indices,
                    state=decoder_result.updated_state,
                    stop_flags=next_stop_flags,
                    sampling_policy=next_sampling_policy,
                ),
                GenerationStepResults(
                    token_ids=next_token_ids,
                    top_k_token_ids=top_k_token_ids,
                    top_k_token_logits=top_k_token_logits,
                ),
            )

        sampling_keys = sampling_keychain.rolling_broadcast((max_output_length, batch_size)).vmapped_keys
        decoding_keys = decoding_keychain.rolling_broadcast((max_output_length,)).vmapped_keys
        _, generated = jax.lax.scan(loop_iteration, initial_state, (sampling_keys, decoding_keys))

        token_ids = rearrange(generated.token_ids, "step batch -> batch step")
        if num_top_logits_to_return is None:
            top_k_token_ids = None
            top_k_token_logits = None
        else:
            top_k_token_ids = rearrange(generated.top_k_token_ids, "step batch k -> batch step k")
            top_k_token_logits = rearrange(generated.top_k_token_logits, "step batch k -> batch step k")

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
        *,
        keychain: Keychain,
    ) -> AssistantMessage:
        token_ids = jnp.asarray(self.token_codec.encode_request(messages), dtype=jnp.int32)[None, :]
        response_ids = self.generate_tokens(
            token_ids,
            generation_config,
            max_output_length=max_output_length,
            keychain=keychain,
        ).token_ids[0]
        return self.token_codec.decode_response(self._trim_at_eos(response_ids.tolist()))

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> Iterable[Int[Array, ""]]:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")

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
        length_without_padding = jnp.asarray([input_length], dtype=jnp.int32)

        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        if sampling_policy.has_count_penalties():
            sampling_policy = sampling_policy.with_prompt_token_counts(
                padded_token_ids,
                jnp.asarray(input_length, dtype=jnp.int32),
                self.decoder.vocab_size,
            )

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self._prefill(
            padded_token_ids[None, :],
            padded_input_length + max_output_length + 1,
            length_without_padding,
            keychain=prefill_keychain,
        )

        last_token_logits = prefill_results.last_token_logits[0]
        last_token_index = prefill_results.last_token_indices[0]
        state = prefill_results.state
        sampling_keys = sampling_keychain.rolling_broadcast((max_output_length,)).vmapped_keys
        decoding_keys = decoding_keychain.rolling_broadcast((max_output_length,)).vmapped_keys
        for sampling_key, decoding_key in zip(sampling_keys, decoding_keys, strict=True):
            processed_logits = sampling_policy.process_logits(last_token_logits.astype(jnp.float32))
            next_token_id = jax.random.categorical(sampling_key, processed_logits)
            yield next_token_id

            if bool(jnp.any(next_token_id == stop_token_ids).item()):
                return

            sampling_policy = sampling_policy.with_next_token_count(next_token_id)
            next_token_index = last_token_index + 1
            decoder_result = self.decoder(
                token_ids=next_token_id.reshape(1, 1),
                token_positions=next_token_index.reshape(1, 1),
                state=state,
                return_updated_state=True,
                forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
                keychain=Keychain(vmapped_keys=decoding_key, batch_key=decoding_keychain.batch_key),
            )
            assert decoder_result.updated_state is not None

            last_token_logits = decoder_result.logits[0, 0, :].astype(jnp.float32)
            last_token_index = next_token_index
            state = decoder_result.updated_state

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        *,
        keychain: Keychain,
    ) -> Iterable[str]:
        token_ids = jnp.asarray(self.token_codec.encode_request(messages), dtype=jnp.int32)
        response_token_ids: list[int] = []
        previous_text = ""
        for token_id in self.stream_tokens(
            token_ids,
            generation_config,
            max_output_length,
            keychain=keychain,
        ):
            response_token_ids.append(int(token_id.item()))
            current_text = self.token_codec.decode_tokens(response_token_ids, hide_invalid_utf_chars=True)
            yield current_text[len(previous_text) :]
            previous_text = current_text
