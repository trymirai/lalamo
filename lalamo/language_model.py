import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from safetensors.flax import load_file
from tokenizers import Tokenizer

from lalamo.common import DTypeLike, ParameterTree, unflatten_parameters
from lalamo.message_processor import AssistantMessage, Message, MessageProcessor, MessageProcessorConfig
from lalamo.modules import Decoder, DecoderConfig, KVCache, LalamoModule, WeightLayout, config_converter
from lalamo.sampling import SamplingPolicy, make_policy

__all__ = [
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, " vocabulary"]
    last_token_position: Int[Array, ""]
    kv_cache: KVCache


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, " vocabulary"]
    last_token_position: Int[Array, ""]
    kv_cache: KVCache
    stop_flag: Bool[Array, ""]


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
    def load(cls, path: Path | str, weight_layout: WeightLayout = WeightLayout.AUTO) -> Self:
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json") as config_file:
            config_json = json.load(config_file)
        config = config_converter.structure(config_json["model_config"], LanguageModelConfig)
        weights = unflatten_parameters(load_file(path / "model.safetensors"))
        decoder = config.decoder_config.empty().import_weights(weights, weight_layout)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        message_processor = MessageProcessor(config.message_processor_config, tokenizer)
        return cls(config, decoder, message_processor)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.decoder.activation_precision

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterTree:
        return self.decoder.export_weights(weight_layout)

    def import_weights(
        self,
        weights: ParameterTree[Array],
        weight_layout: WeightLayout = WeightLayout.AUTO,
    ) -> Self:
        return replace(
            self,
            decoder=self.decoder.import_weights(weights, weight_layout),
        )

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.config.generation_config.stop_token_ids

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    @eqx.filter_jit
    def _prefill(
        self,
        token_ids: Int[Array, " tokens"],
        length_without_padding: Int[Array, ""] | int | None = None,
        kv_cache_capacity: int | None = None,
    ) -> PrefillResults:
        (num_tokens,) = token_ids.shape
        token_positions = jnp.arange(num_tokens, dtype=jnp.int32)
        if kv_cache_capacity is not None:
            kv_cache = self.decoder.init_static_kv_cache(kv_cache_capacity)
        else:
            kv_cache = None

        decoder_outputs = self.decoder(
            token_ids,
            token_positions,
            kv_cache,
            return_updated_kv_cache=True,
            length_without_padding=length_without_padding,
        )

        if length_without_padding is not None:
            last_logits_index = length_without_padding - 1
        else:
            last_logits_index = num_tokens - 1

        last_token_logits = decoder_outputs.logits[last_logits_index, :]
        last_token_position = jnp.array(last_logits_index, dtype=jnp.int32)

        assert decoder_outputs.updated_kv_cache is not None
        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_position=last_token_position,
            kv_cache=decoder_outputs.updated_kv_cache,
        )

    @eqx.filter_jit
    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        prompt_length_without_padding: Int[Array, ""] | int | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, " response_tokens"]:
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape
        prefill_results = self._prefill(
            prompt_token_ids,
            prompt_length_without_padding,
            input_length + max_output_length,
        )

        initial_state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_position,
            prefill_results.kv_cache,
            jnp.array(0, dtype=jnp.bool),
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=max_output_length)

        def loop_iteration(
            state: DecodingState,
            key: PRNGKeyArray,
        ) -> tuple[DecodingState, Int[Array, ""]]:
            def sample_and_update() -> tuple[DecodingState, Int[Array, ""]]:
                processed_logits = sampling_policy.process_logits(state.last_token_logits)
                next_token_id = jax.random.categorical(key, processed_logits)
                next_token_position = state.last_token_position + 1

                stop_flag = state.stop_flag | jnp.any(next_token_id == eos_token_ids)

                decoder_outputs = self.decoder(
                    next_token_id.reshape(1),
                    next_token_position.reshape(1),
                    state.kv_cache,
                    return_updated_kv_cache=True,
                )
                assert decoder_outputs.updated_kv_cache is not None, "updated_kv_cache should not be None"
                new_state = DecodingState(
                    decoder_outputs.logits.squeeze(),
                    next_token_position,
                    decoder_outputs.updated_kv_cache,
                    stop_flag,
                )
                return new_state, next_token_id

            def pad_and_repeat_state() -> tuple[DecodingState, Int[Array, ""]]:
                pad_token = jnp.array(0, dtype=jnp.int32)
                return state, pad_token

            return jax.lax.cond(state.stop_flag, pad_and_repeat_state, sample_and_update)

        _, tokens = jax.lax.scan(loop_iteration, initial_state, keys)

        return tokens

    def reply(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> AssistantMessage:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize(formatted_messages), dtype=jnp.int32)
        response_ids = self.generate_tokens(token_ids, sampling_policy, key=key)
        response_text = self.message_processor.detokenize(response_ids.tolist())
        return self.message_processor.parse_response(response_text)

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[str]:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize(formatted_messages), dtype=jnp.int32)
        for token_id in self.stream_tokens(token_ids, sampling_policy, key=key):
            yield self.message_processor.detokenize([token_id.item()])

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        prompt_length_without_padding: Int[Array, ""] | int | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[Int[Array, ""]]:
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape
        prefill_results = self._prefill(
            prompt_token_ids,
            prompt_length_without_padding,
            input_length + max_output_length,
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=max_output_length)

        state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_position,
            prefill_results.kv_cache,
            jnp.array(0, dtype=jnp.bool),
        )

        for iter_key in keys:
            processed_logits = sampling_policy.process_logits(state.last_token_logits)
            next_token_id = jax.random.categorical(iter_key, processed_logits)

            yield next_token_id

            if jnp.any(next_token_id == eos_token_ids):
                return

            next_token_position = state.last_token_position + 1
            decoder_outputs = self.decoder(
                next_token_id.reshape(1),
                next_token_position.reshape(1),
                state.kv_cache,
                return_updated_kv_cache=True,
            )
            assert decoder_outputs.updated_kv_cache is not None, "updated_kv_cache should not be None"
            state = DecodingState(
                decoder_outputs.logits.squeeze(),
                next_token_position,
                decoder_outputs.updated_kv_cache,
                state.stop_flag,
            )
