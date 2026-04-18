from abc import abstractmethod
from collections.abc import Iterable
from math import log
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

__all__ = [
    "BanTokensPolicy",
    "CountingPenalty",
    "FrequencyPenalty",
    "GreedyPolicy",
    "LogitTransform",
    "MinPPolicy",
    "PresencePenalty",
    "RepetitionPenalty",
    "SamplingPolicy",
    "TemperaturePolicy",
    "TopKPolicy",
    "TopPPolicy",
    "make_policy",
]


class LogitTransform(eqx.Module):
    @abstractmethod
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],  # noqa: ARG002
        prompt_length: Int[Array, ""],  # noqa: ARG002
        vocab_size: int,  # noqa: ARG002
    ) -> Self:
        return self

    def update(self, next_token: Int[Array, ""]) -> Self:  # noqa: ARG002
        return self

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process_logits(logits))


class GreedyPolicy(LogitTransform):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        best = jnp.argmax(logits)
        return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)


class TemperaturePolicy(LogitTransform):
    temperature: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature == 0.0:
            best = jnp.argmax(logits)
            return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)
        return logits / self.temperature


class TopKPolicy(LogitTransform):
    k: int = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        # jax.lax.top_k triggers an XLA topk-decomposer bug under SPMD: the decomposed sort
        # comparator gets 2 params instead of the 4 required for multi-device sort.
        k = min(self.k, logits.shape[0])
        kth_logit = jnp.sort(logits, descending=True)[k - 1]
        return jnp.where(logits >= kth_logit, logits, -jnp.inf)


class TopPPolicy(LogitTransform):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        drop_sorted = jnp.roll(cumulative_probs > self.p, 1).at[0].set(False)
        drop = jnp.zeros_like(drop_sorted).at[sorted_indices].set(drop_sorted)
        return jnp.where(drop, -jnp.inf, logits)


class MinPPolicy(LogitTransform):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.p == 0.0:
            return logits
        cutoff = jnp.max(logits) + log(self.p)
        return jnp.where(logits >= cutoff, logits, -jnp.inf)


class BanTokensPolicy(LogitTransform):
    banned_tokens: tuple[int, ...] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits.at[jnp.asarray(self.banned_tokens, dtype=jnp.int32)].set(-jnp.inf)


class CountingPenalty(LogitTransform):
    """Base for penalties that track how often each token has appeared."""

    penalty: float = eqx.field(static=True)
    token_counts: Int[Array, " vocabulary"] | None = None

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> Self:
        mask = (jnp.arange(prompt_token_ids.shape[0]) < prompt_length).astype(jnp.int32)
        counts = jnp.zeros(vocab_size, dtype=jnp.int32).at[prompt_token_ids].add(mask)
        return eqx.tree_at(lambda s: s.token_counts, self, counts)

    def update(self, next_token: Int[Array, ""]) -> Self:
        assert self.token_counts is not None
        return eqx.tree_at(lambda s: s.token_counts, self, self.token_counts.at[next_token].add(1))


class RepetitionPenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        assert self.token_counts is not None
        seen = self.token_counts > 0
        return jnp.where(
            seen & (logits > 0),
            logits / self.penalty,
            jnp.where(seen, logits * self.penalty, logits),
        )


class PresencePenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        assert self.token_counts is not None
        return jnp.where(self.token_counts > 0, logits - self.penalty, logits)


class FrequencyPenalty(CountingPenalty):
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        assert self.token_counts is not None
        return logits - self.penalty * self.token_counts.astype(logits.dtype)


class SamplingPolicy(LogitTransform):
    policies: tuple[LogitTransform, ...]

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process_logits(logits)
        return logits

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> Self:
        return eqx.tree_at(
            lambda s: s.policies,
            self,
            tuple(p.init(prompt_token_ids, prompt_length, vocab_size) for p in self.policies),
        )

    def update(self, next_token: Int[Array, ""]) -> Self:
        return eqx.tree_at(
            lambda s: s.policies,
            self,
            tuple(p.update(next_token) for p in self.policies),
        )


def make_policy(
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    banned_tokens: Iterable[int] | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> SamplingPolicy:
    policies: list[LogitTransform] = []
    if banned_tokens:
        policies.append(BanTokensPolicy(tuple(banned_tokens)))
    if repetition_penalty is not None:
        policies.append(RepetitionPenalty(repetition_penalty))
    if presence_penalty is not None:
        policies.append(PresencePenalty(presence_penalty))
    if frequency_penalty is not None:
        policies.append(FrequencyPenalty(frequency_penalty))
    if temperature is not None:
        policies.append(TemperaturePolicy(temperature))
    if top_k is not None:
        policies.append(TopKPolicy(top_k))
    if top_p is not None:
        policies.append(TopPPolicy(top_p))
    if min_p is not None:
        policies.append(MinPPolicy(min_p))
    return SamplingPolicy(tuple(policies))
