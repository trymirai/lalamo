from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from lalamo.modules import Decoder, KVCache

__all__ = [
    "BanTokensPolicy",
    "CompositePolicy",
    "GreedyPolicy",
    "LanguageModel",
    "SamplingPolicy",
    "TemperaturePolicy",
    "TopKPolicy",
    "TopPPolicy",
]


class SamplingPolicy(eqx.Module):
    @abstractmethod
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process_logits(logits))


class GreedyPolicy(SamplingPolicy):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        max_logit_value = jnp.max(logits)
        return jnp.where(logits == max_logit_value, 1.0, -jnp.inf)


class TemperaturePolicy(SamplingPolicy):
    temperature: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits / self.temperature


class TopKPolicy(SamplingPolicy):
    k: int = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        top_k_logits, _ = jax.lax.top_k(logits, self.k)
        min_logit_val = jnp.min(top_k_logits)
        return jnp.where(logits >= min_logit_val, logits, -jnp.inf)


class TopPPolicy(SamplingPolicy):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        to_remove = cumulative_probs > self.p
        to_remove = jnp.roll(to_remove, 1)
        to_remove = to_remove.at[0].set(False)

        return jnp.where(to_remove, -jnp.inf, logits)


class BanTokensPolicy(SamplingPolicy):
    banned_tokens: list[int] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        banned_tokens_indices = jnp.asarray(self.banned_tokens, dtype=jnp.int32)
        return logits.at[banned_tokens_indices].set(-jnp.inf)


class CompositePolicy(SamplingPolicy):
    policies: list[SamplingPolicy] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process_logits(logits)
        return logits


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
class LanguageModel:
    decoder: Decoder

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

    def generate(
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
            sampling_policy = TemperaturePolicy(temperature=1.0)

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

                if eos_token_ids is not None:
                    stop_flag = state.stop_flag | jnp.any(next_token_id == eos_token_ids)
                else:
                    stop_flag = state.stop_flag

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

    def stream(
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
            sampling_policy = TemperaturePolicy(temperature=1.0)

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

            if eos_token_ids is not None and jnp.any(next_token_id == eos_token_ids):
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
