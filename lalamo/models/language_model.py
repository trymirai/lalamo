from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

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
