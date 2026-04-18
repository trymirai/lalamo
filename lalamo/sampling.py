from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import replace
from math import log
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

__all__ = [
    "BanTokensPolicy",
    "CompositePolicy",
    "CountingPenalty",
    "FrequencyPenalty",
    "GreedyPolicy",
    "MinPPolicy",
    "PresencePenalty",
    "RepetitionPenalty",
    "SamplingPolicy",
    "TemperaturePolicy",
    "TopKPolicy",
    "TopPPolicy",
    "make_policy",
]


class SamplingPolicy(eqx.Module):
    @abstractmethod
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    def init(self, prompt_token_ids: Int[Array, " tokens"], prompt_length: Int[Array, ""]) -> Self:  # noqa: ARG002
        return self

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:  # noqa: ARG002
        return self

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process_logits(logits))


class GreedyPolicy(SamplingPolicy):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        best = jnp.argmax(logits)
        return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)


class TemperaturePolicy(SamplingPolicy):
    temperature: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature == 0.0:
            best = jnp.argmax(logits)
            return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)
        return logits / self.temperature


class TopKPolicy(SamplingPolicy):
    k: int = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        # jax.lax.top_k triggers an XLA topk-decomposer bug under SPMD: the decomposed sort
        # comparator gets 2 params instead of the 4 required for multi-device sort.
        k = min(self.k, logits.shape[0])
        min_logit_val = jnp.sort(logits, descending=True)[k - 1]
        return jnp.where(logits >= min_logit_val, logits, -jnp.inf)


class TopPPolicy(SamplingPolicy):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        to_remove_sorted = cumulative_probs > self.p
        to_remove_sorted = jnp.roll(to_remove_sorted, 1)
        to_remove_sorted = to_remove_sorted.at[0].set(False)

        to_remove_unsorted = jnp.empty_like(to_remove_sorted).at[sorted_indices].set(to_remove_sorted)

        return jnp.where(to_remove_unsorted, -jnp.inf, logits)


class MinPPolicy(SamplingPolicy):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.p == 0.0:
            return logits
        max_logit = jnp.max(logits)
        logit_cutoff = max_logit + log(self.p)
        return jnp.where(logits >= logit_cutoff, logits, -jnp.inf)


class BanTokensPolicy(SamplingPolicy):
    banned_tokens: tuple[int, ...] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        banned_tokens_indices = jnp.asarray(self.banned_tokens, dtype=jnp.int32)
        return logits.at[banned_tokens_indices].set(-jnp.inf)


class CountingPenalty(SamplingPolicy):
    """Base for penalties that track per-vocab token counts seen in the prompt and generated so far."""

    penalty: float = eqx.field(static=True)
    token_counts: Int[Array, " vocabulary"]

    @classmethod
    def zero(cls, penalty: float, vocab_size: int) -> Self:
        return cls(penalty=penalty, token_counts=jnp.zeros(vocab_size, dtype=jnp.int32))

    def init(self, prompt_token_ids: Int[Array, " tokens"], prompt_length: Int[Array, ""]) -> Self:
        mask = (jnp.arange(prompt_token_ids.shape[0]) < prompt_length) & (
            prompt_token_ids < self.token_counts.shape[0]
        )
        return replace(self, token_counts=self.token_counts.at[prompt_token_ids].add(mask.astype(jnp.int32)))

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:
        return replace(self, token_counts=self.token_counts.at[next_token].add(active.astype(jnp.int32)))


class RepetitionPenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        seen = self.token_counts > 0
        return jnp.where(
            seen & (logits > 0),
            logits / self.penalty,
            jnp.where(seen, logits * self.penalty, logits),
        )


class PresencePenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return jnp.where(self.token_counts > 0, logits - self.penalty, logits)


class FrequencyPenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits - self.penalty * self.token_counts.astype(logits.dtype)


class CompositePolicy(SamplingPolicy):
    policies: tuple[SamplingPolicy, ...]

    def init(self, prompt_token_ids: Int[Array, " tokens"], prompt_length: Int[Array, ""]) -> Self:
        return replace(self, policies=tuple(p.init(prompt_token_ids, prompt_length) for p in self.policies))

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:
        return replace(self, policies=tuple(p.update(next_token, active) for p in self.policies))

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process_logits(logits)
        return logits


def make_policy(
    vocab_size: int,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    banned_tokens: Iterable[int] | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> SamplingPolicy:
    policies: list[SamplingPolicy] = []
    if banned_tokens:
        policies.append(BanTokensPolicy(tuple(banned_tokens)))
    if repetition_penalty is not None:
        policies.append(RepetitionPenalty.zero(repetition_penalty, vocab_size))
    if presence_penalty is not None:
        policies.append(PresencePenalty.zero(presence_penalty, vocab_size))
    if frequency_penalty is not None:
        policies.append(FrequencyPenalty.zero(frequency_penalty, vocab_size))
    if temperature is not None:
        policies.append(TemperaturePolicy(temperature))
    if top_k is not None:
        policies.append(TopKPolicy(top_k))
    if top_p is not None:
        policies.append(TopPPolicy(top_p))
    if min_p is not None:
        policies.append(MinPPolicy(min_p))
    return CompositePolicy(tuple(policies))
