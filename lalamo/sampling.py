from abc import abstractmethod
from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

__all__ = [
    "BanTokensPolicy",
    "CompositePolicy",
    "GreedyPolicy",
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

        to_remove_sorted = cumulative_probs > self.p
        to_remove_sorted = jnp.roll(to_remove_sorted, 1)
        to_remove_sorted = to_remove_sorted.at[0].set(False)

        to_remove_unsorted = jnp.empty_like(to_remove_sorted).at[sorted_indices].set(to_remove_sorted)

        return jnp.where(to_remove_unsorted, -jnp.inf, logits)


class BanTokensPolicy(SamplingPolicy):
    banned_tokens: tuple[int, ...] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        banned_tokens_indices = jnp.asarray(self.banned_tokens, dtype=jnp.int32)
        return logits.at[banned_tokens_indices].set(-jnp.inf)


class CompositePolicy(SamplingPolicy):
    policies: tuple[SamplingPolicy, ...] = eqx.field(static=True)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        for policy in self.policies:
            logits = policy.process_logits(logits)
        return logits


def make_policy(
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    banned_tokens: Iterable[int] | None = None,
) -> SamplingPolicy:
    policies = []
    if banned_tokens:
        policies.append(BanTokensPolicy(tuple(banned_tokens)))
    if temperature is not None:
        policies.append(TemperaturePolicy(temperature))
    if top_k is not None:
        policies.append(TopKPolicy(top_k))
    if top_p is not None:
        policies.append(TopPPolicy(top_p))
    return CompositePolicy(tuple(policies))
