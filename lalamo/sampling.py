from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import replace
from math import log
from typing import ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

__all__ = [
    "BanTokensPolicy",
    "CountingPenalty",
    "CountingPenaltyConfig",
    "FrequencyPenalty",
    "FrequencyPenaltyConfig",
    "GreedyPolicy",
    "LogitTransform",
    "LogitTransformConfig",
    "MinPPolicy",
    "PresencePenalty",
    "PresencePenaltyConfig",
    "RepetitionPenalty",
    "RepetitionPenaltyConfig",
    "SamplingPolicy",
    "SamplingPolicyConfig",
    "StatelessLogitTransform",
    "TemperaturePolicy",
    "TopKPolicy",
    "TopPPolicy",
    "make_policy",
]


class LogitTransformConfig(eqx.Module):
    @abstractmethod
    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> "LogitTransform": ...


class LogitTransform(eqx.Module):
    @abstractmethod
    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    @abstractmethod
    def update(self, next_token: Int[Array, ""]) -> "LogitTransform": ...


class StatelessLogitTransform(LogitTransformConfig, LogitTransform):
    """A transform whose config and runtime are the same object."""

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],  # noqa: ARG002
        prompt_length: Int[Array, ""],  # noqa: ARG002
        vocab_size: int,  # noqa: ARG002
    ) -> Self:
        return self

    def update(self, next_token: Int[Array, ""]) -> Self:  # noqa: ARG002
        return self


class GreedyPolicy(StatelessLogitTransform):
    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        best = jnp.argmax(logits)
        return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)


class TemperaturePolicy(StatelessLogitTransform):
    temperature: float = eqx.field(static=True)

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature == 0.0:
            best = jnp.argmax(logits)
            return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)
        return logits / self.temperature


class TopKPolicy(StatelessLogitTransform):
    k: int = eqx.field(static=True)

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        # jax.lax.top_k triggers an XLA topk-decomposer bug under SPMD: the decomposed sort
        # comparator gets 2 params instead of the 4 required for multi-device sort.
        kth_logit = jnp.sort(logits, descending=True)[self.k - 1]
        return jnp.where(logits >= kth_logit, logits, -jnp.inf)


class TopPPolicy(StatelessLogitTransform):
    p: float = eqx.field(static=True)

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        drop_sorted = jnp.roll(cumulative_probs > self.p, 1).at[0].set(False)
        drop = jnp.empty_like(drop_sorted).at[sorted_indices].set(drop_sorted)
        return jnp.where(drop, -jnp.inf, logits)


class MinPPolicy(StatelessLogitTransform):
    p: float = eqx.field(static=True)

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.p == 0.0:
            return logits
        cutoff = jnp.max(logits) + log(self.p)
        return jnp.where(logits >= cutoff, logits, -jnp.inf)


class BanTokensPolicy(StatelessLogitTransform):
    banned_tokens: tuple[int, ...] = eqx.field(static=True)

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits.at[jnp.asarray(self.banned_tokens, dtype=jnp.int32)].set(-jnp.inf)


def _prompt_token_counts(
    prompt_token_ids: Int[Array, " tokens"],
    prompt_length: Int[Array, ""],
    vocab_size: int,
) -> Int[Array, " vocabulary"]:
    mask = (jnp.arange(prompt_token_ids.shape[0]) < prompt_length).astype(jnp.int32)
    return jnp.zeros(vocab_size, dtype=jnp.int32).at[prompt_token_ids].add(mask)


class CountingPenalty(LogitTransform):
    """Runtime base for penalties that track how often each token has appeared."""

    penalty: float = eqx.field(static=True)
    token_counts: Int[Array, " vocabulary"]

    def update(self, next_token: Int[Array, ""]) -> Self:
        return replace(self, token_counts=self.token_counts.at[next_token].add(1))


class CountingPenaltyConfig(LogitTransformConfig):
    """Config base for `CountingPenalty` descendants."""

    penalty: float = eqx.field(static=True)
    runtime_class: ClassVar[type[CountingPenalty]]

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> CountingPenalty:
        return self.runtime_class(self.penalty, _prompt_token_counts(prompt_token_ids, prompt_length, vocab_size))


class RepetitionPenalty(CountingPenalty):
    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        seen = self.token_counts > 0
        return jnp.where(
            seen & (logits > 0),
            logits / self.penalty,
            jnp.where(seen, logits * self.penalty, logits),
        )


class PresencePenalty(CountingPenalty):
    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return jnp.where(self.token_counts > 0, logits - self.penalty, logits)


class FrequencyPenalty(CountingPenalty):
    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits - self.penalty * self.token_counts.astype(logits.dtype)


class RepetitionPenaltyConfig(CountingPenaltyConfig):
    runtime_class: ClassVar[type[CountingPenalty]] = RepetitionPenalty


class PresencePenaltyConfig(CountingPenaltyConfig):
    runtime_class: ClassVar[type[CountingPenalty]] = PresencePenalty


class FrequencyPenaltyConfig(CountingPenaltyConfig):
    runtime_class: ClassVar[type[CountingPenalty]] = FrequencyPenalty


class SamplingPolicyConfig(eqx.Module):
    policies: tuple[LogitTransformConfig, ...] = eqx.field(static=True)

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> "SamplingPolicy":
        return SamplingPolicy(tuple(p.init(prompt_token_ids, prompt_length, vocab_size) for p in self.policies))


class SamplingPolicy(eqx.Module):
    policies: tuple[LogitTransform, ...]

    def process(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process(logits)
        return logits

    def update(self, next_token: Int[Array, ""]) -> "SamplingPolicy":
        return SamplingPolicy(tuple(p.update(next_token) for p in self.policies))

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process(logits))


def make_policy(
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    banned_tokens: Iterable[int] | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> SamplingPolicyConfig:
    policies: list[LogitTransformConfig] = []
    if banned_tokens:
        policies.append(BanTokensPolicy(tuple(banned_tokens)))
    if repetition_penalty is not None:
        policies.append(RepetitionPenaltyConfig(repetition_penalty))
    if presence_penalty is not None:
        policies.append(PresencePenaltyConfig(presence_penalty))
    if frequency_penalty is not None:
        policies.append(FrequencyPenaltyConfig(frequency_penalty))
    if temperature is not None:
        policies.append(TemperaturePolicy(temperature))
    if top_k is not None:
        policies.append(TopKPolicy(top_k))
    if top_p is not None:
        policies.append(TopPPolicy(top_p))
    if min_p is not None:
        policies.append(MinPPolicy(min_p))
    return SamplingPolicyConfig(tuple(policies))
