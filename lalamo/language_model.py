from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from lalamo.modules import Decoder, KVCacheLayer


class PrefillResults(NamedTuple):
    logits: Float[Array, " vocabulary"]
    kv_cache: tuple[KVCacheLayer, ...]


class SamplingPolicy(eqx.Module):
    @abstractmethod
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process_logits(logits))


class GreedyPolicy(SamplingPolicy):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        max_logit_value = jnp.max(logits)
        return jnp.where(logits == max_logit_value, 1.0, -jnp.inf)


class Temperature(SamplingPolicy):
    temperature: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits / self.temperature


class TopK(SamplingPolicy):
    k: int = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        top_k_logits, _ = jax.lax.top_k(logits, self.k)
        min_logit_val = jnp.min(top_k_logits)
        return jnp.where(logits >= min_logit_val, logits, -jnp.inf)


class TopP(SamplingPolicy):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        to_remove = cumulative_probs > self.p
        to_remove = jnp.roll(to_remove, 1)
        to_remove = to_remove.at[0].set(False)

        return jnp.where(to_remove, -jnp.inf, logits)


class BanTokens(SamplingPolicy):
    banned_tokens: list[int] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        banned_tokens_indices = jnp.asarray(self.banned_tokens, dtype=jnp.int32)
        return logits.at[banned_tokens_indices].set(-jnp.inf)


class Compose(SamplingPolicy):
    policies: list[SamplingPolicy] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process_logits(logits)
        return logits


@dataclass
class LanguageModel:
    decoder: Decoder

    def prefill(self, token_ids: Int[Array, " tokens"], kv_cache_capacity: int | None = None) -> PrefillResults:
        (num_tokens,) = token_ids.shape
        token_positions = jnp.arange(num_tokens, dtype=jnp.int32)
        if kv_cache_capacity is not None:
            kv_cache = self.decoder.init_static_kv_cache(kv_cache_capacity)
        else:
            kv_cache = None
        decoder_outputs = self.decoder(token_ids, token_positions, kv_cache, return_updated_kv_cache=True)
        last_token_logits = decoder_outputs.logits[-1, :]
        assert decoder_outputs.updated_kv_cache is not None
        return PrefillResults(logits=last_token_logits, kv_cache=decoder_outputs.updated_kv_cache)

    def decode(
        self,
        prompt_tokens: Int[Array, " prompt_tokens"],
        max_sequence_length: int | None = None,
    ) -> Int[Array, " response_tokens"]:
        jax.lax.scan()
