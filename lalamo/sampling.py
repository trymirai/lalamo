from abc import abstractmethod
from collections.abc import Iterable
from math import log
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

__all__ = [
    "BanTokensPolicy",
    "CountingPenalty",
    "CountingPenaltyConfig",
    "FrequencyPenalty",
    "FrequencyPenaltyConfig",
    "LogitTransform",
    "LogitTransformConfig",
    "MinPPolicy",
    "PresencePenalty",
    "PresencePenaltyConfig",
    "RepetitionPenalty",
    "RepetitionPenaltyConfig",
    "SamplingPipeline",
    "SamplingPipelineConfig",
    "TemperaturePolicy",
    "TopKPolicy",
    "TopPPolicy",
    "make_policy",
]


class LogitTransform(eqx.Module):
    """Runtime transform: may carry per-sequence state updated after each sampled token."""

    @abstractmethod
    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]: ...

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:  # noqa: ARG002
        return self

    def __call__(self, logits: Float[Array, " vocabulary"], *, key: PRNGKeyArray) -> Int[Array, ""]:
        return jax.random.categorical(key, self.process_logits(logits))


class LogitTransformConfig(eqx.Module):
    """Config that builds a LogitTransform from prompt context."""

    @abstractmethod
    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> LogitTransform: ...


class _StatelessTransform(LogitTransform, LogitTransformConfig):
    """A stateless transform is its own config (identity init, identity update)."""

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],  # noqa: ARG002
        prompt_length: Int[Array, ""],  # noqa: ARG002
        vocab_size: int,  # noqa: ARG002
    ) -> Self:
        return self


class TemperaturePolicy(_StatelessTransform):
    temperature: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature == 0.0:
            best = jnp.argmax(logits)
            return jnp.where(jnp.arange(logits.shape[0]) == best, 1.0, -jnp.inf)
        return logits / self.temperature


class TopKPolicy(_StatelessTransform):
    k: int = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        # jax.lax.top_k triggers an XLA topk-decomposer bug under SPMD: the decomposed sort
        # comparator gets 2 params instead of the 4 required for multi-device sort.
        k = min(self.k, logits.shape[0])
        kth_logit = jnp.sort(logits, descending=True)[k - 1]
        return jnp.where(logits >= kth_logit, logits, -jnp.inf)


class TopPPolicy(_StatelessTransform):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, descending=True)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

        # Shift the drop-mask by one so the first token above-p is kept; position 0 is always kept.
        drop_sorted = jnp.roll(cumulative_probs > self.p, 1).at[0].set(False)
        drop = jnp.zeros_like(drop_sorted).at[sorted_indices].set(drop_sorted)
        return jnp.where(drop, -jnp.inf, logits)


class MinPPolicy(_StatelessTransform):
    p: float = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.p == 0.0:
            return logits
        cutoff = jnp.max(logits) + log(self.p)
        return jnp.where(logits >= cutoff, logits, -jnp.inf)


class BanTokensPolicy(_StatelessTransform):
    banned_tokens: tuple[int, ...] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        return logits.at[jnp.asarray(self.banned_tokens, dtype=jnp.int32)].set(-jnp.inf)


class CountingPenalty(LogitTransform):
    """Base for penalties that track how often each token has appeared."""

    penalty: float = eqx.field(static=True)
    token_counts: Int[Array, " vocabulary"]

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:
        increment = active.astype(self.token_counts.dtype)
        return eqx.tree_at(lambda s: s.token_counts, self, self.token_counts.at[next_token].add(increment))


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


class CountingPenaltyConfig(LogitTransformConfig):
    penalty: float = eqx.field(static=True)

    @abstractmethod
    def _build(self, token_counts: Int[Array, " vocabulary"]) -> CountingPenalty: ...

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> CountingPenalty:
        in_range = (jnp.arange(prompt_token_ids.shape[0]) < prompt_length) & (prompt_token_ids < vocab_size)
        counts = jnp.zeros(vocab_size, dtype=jnp.int32).at[prompt_token_ids].add(in_range.astype(jnp.int32))
        return self._build(counts)


class RepetitionPenaltyConfig(CountingPenaltyConfig):
    def _build(self, token_counts: Int[Array, " vocabulary"]) -> RepetitionPenalty:
        return RepetitionPenalty(penalty=self.penalty, token_counts=token_counts)


class PresencePenaltyConfig(CountingPenaltyConfig):
    def _build(self, token_counts: Int[Array, " vocabulary"]) -> PresencePenalty:
        return PresencePenalty(penalty=self.penalty, token_counts=token_counts)


class FrequencyPenaltyConfig(CountingPenaltyConfig):
    def _build(self, token_counts: Int[Array, " vocabulary"]) -> FrequencyPenalty:
        return FrequencyPenalty(penalty=self.penalty, token_counts=token_counts)


class SamplingPipeline(LogitTransform):
    stages: tuple[LogitTransform, ...]

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for stage in self.stages:
            logits = stage.process_logits(logits)
        return logits

    def update(self, next_token: Int[Array, ""], active: Bool[Array, ""]) -> Self:
        return eqx.tree_at(
            lambda s: s.stages,
            self,
            tuple(stage.update(next_token, active) for stage in self.stages),
        )


class SamplingPipelineConfig(LogitTransformConfig):
    stages: tuple[LogitTransformConfig, ...]

    def init(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocab_size: int,
    ) -> SamplingPipeline:
        return SamplingPipeline(
            tuple(stage.init(prompt_token_ids, prompt_length, vocab_size) for stage in self.stages),
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
) -> SamplingPipelineConfig:
    stages: list[LogitTransformConfig] = []
    if banned_tokens:
        stages.append(BanTokensPolicy(tuple(banned_tokens)))
    if repetition_penalty is not None:
        stages.append(RepetitionPenaltyConfig(repetition_penalty))
    if presence_penalty is not None:
        stages.append(PresencePenaltyConfig(presence_penalty))
    if frequency_penalty is not None:
        stages.append(FrequencyPenaltyConfig(frequency_penalty))
    if temperature is not None:
        stages.append(TemperaturePolicy(temperature))
    if top_k is not None:
        stages.append(TopKPolicy(top_k))
    if top_p is not None:
        stages.append(TopPPolicy(top_p))
    if min_p is not None:
        stages.append(MinPPolicy(min_p))
    return SamplingPipelineConfig(tuple(stages))
