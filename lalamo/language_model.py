import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tokenizers import Tokenizer

from lalamo.common import DTypeLike, ParameterTree, unflatten_parameters
from lalamo.message_processor import AssistantMessage, Message, MessageProcessor, MessageProcessorConfig
from lalamo.modules import Decoder, DecoderConfig, KVCache, LalamoModule, config_converter
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.sampling import SamplingPolicy, make_policy
from lalamo.utils import open_safetensors

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
    kv_cache: KVCache


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    kv_cache: KVCache
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
    stop_token_ids: tuple[int, ...]
    temperature: float | None
    top_k: int | None
    top_p: float | None
    banned_tokens: tuple[int, ...] | None

    def default_policy(self) -> SamplingPolicy:
        return make_policy(self.temperature, self.top_k, self.top_p, self.banned_tokens)


@dataclass(frozen=True)
class LanguageModelConfig:
    decoder_config: DecoderConfig
    message_processor_config: MessageProcessorConfig
    generation_config: GenerationConfig


class LanguageModel(LalamoModule[LanguageModelConfig]):
    decoder: Decoder
    message_processor: MessageProcessor = eqx.field(static=True)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], LanguageModelConfig)
        with open_safetensors(path / "model.safetensors") as weights_dict:
            weights = unflatten_parameters(weights_dict)
            decoder = config.decoder_config.empty().import_weights(weights)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = MessageProcessor(config.message_processor_config, tokenizer)
        return cls(config, decoder, message_processor)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.decoder.activation_precision

    def export_weights(self) -> ParameterTree:
        return self.decoder.export_weights()

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return replace(
            self,
            decoder=self.decoder.import_weights(weights),
        )

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
        kv_cache_capacity: int | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        token_positions = jnp.repeat(jnp.arange(sequence_length, dtype=jnp.int32)[None, ...], batch_size, axis=0)
        if kv_cache_capacity is not None:
            kv_cache = self.decoder.init_static_kv_cache(batch_size, kv_cache_capacity)
        else:
            kv_cache = None

        decoder_outputs = self.decoder(
            token_ids,
            token_positions,
            kv_cache,
            return_updated_kv_cache=True,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=forward_pass_config,
        )

        if lengths_without_padding is not None:
            last_logits_indices = lengths_without_padding - 1
        else:
            last_logits_indices = jnp.array([sequence_length - 1] * batch_size, dtype=jnp.int32)

        last_token_logits = vmap(lambda logits, index: logits[index])(decoder_outputs.logits, last_logits_indices)

        assert decoder_outputs.updated_kv_cache is not None
        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=last_logits_indices,
            kv_cache=decoder_outputs.updated_kv_cache,
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
            prefill_results.kv_cache,
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

                decoder_outputs = self.decoder(
                    next_token_ids[:, None],
                    next_token_indices[:, None],
                    state.kv_cache,
                    return_updated_kv_cache=True,
                    forward_pass_mode=forward_pass_mode,
                    forward_pass_config=forward_pass_config,
                )
                assert decoder_outputs.updated_kv_cache is not None, "updated_kv_cache should not be None"
                new_state = DecodingState(
                    decoder_outputs.logits.squeeze(1),
                    next_token_indices,
                    decoder_outputs.updated_kv_cache,
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
        token_ids = jnp.array(self.message_processor.tokenize(formatted_messages), dtype=jnp.int32)[None, :]
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
        token_ids = jnp.array(self.message_processor.tokenize(formatted_messages), dtype=jnp.int32)
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
            prefill_results.kv_cache,
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
            decoder_outputs = self.decoder(
                next_token_id.reshape(1, 1),
                next_token_indices.reshape(1, 1),
                state.kv_cache,
                return_updated_kv_cache=True,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_kv_cache is not None, "updated_kv_cache should not be None"
            state = DecodingState(
                decoder_outputs.logits.squeeze(1),
                next_token_indices,
                decoder_outputs.updated_kv_cache,
                state.stop_flags,
            )
