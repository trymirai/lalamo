from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.module import Keychain

__all__ = ["SamplingPolicy"]


_SENTINEL = -1
_MAX_BANNED_TOKENS = 16


class SamplingPolicy(eqx.Module):
    temperature: Float[Array, "*batch"]
    top_k: Int[Array, "*batch"]
    top_p: Float[Array, "*batch"]
    min_p: Float[Array, "*batch"]
    banned_tokens: Int[Array, "*batch max_banned_tokens"]

    @classmethod
    def init(
        cls,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        banned_tokens: Iterable[int] | None = None,
    ) -> "SamplingPolicy":
        return cls(
            temperature=jnp.asarray(1.0 if temperature is None else temperature, dtype=jnp.float32),
            top_k=jnp.asarray(0 if top_k is None else top_k, dtype=jnp.int32),
            top_p=jnp.asarray(1.0 if top_p is None else top_p, dtype=jnp.float32),
            min_p=jnp.asarray(0.0 if min_p is None else min_p, dtype=jnp.float32),
            banned_tokens=jnp.asarray(
                _pad_banned_tokens(()) if banned_tokens is None else _pad_banned_tokens(banned_tokens),
                dtype=jnp.int32,
            ),
        )

    @classmethod
    def init_batch(
        cls,
        temperature: Iterable[float],
        top_k: Iterable[int],
        top_p: Iterable[float],
        min_p: Iterable[float],
        banned_tokens: Iterable[Iterable[int]],
    ) -> "SamplingPolicy":
        temperatures = tuple(temperature)
        top_ks = tuple(top_k)
        top_ps = tuple(top_p)
        min_ps = tuple(min_p)
        banned_token_rows = tuple(tuple(row) for row in banned_tokens)
        _raise_if_different_lengths(temperatures, top_ks, top_ps, min_ps, banned_token_rows)

        return cls(
            temperature=jnp.asarray(temperatures, dtype=jnp.float32),
            top_k=jnp.asarray(top_ks, dtype=jnp.int32),
            top_p=jnp.asarray(top_ps, dtype=jnp.float32),
            min_p=jnp.asarray(min_ps, dtype=jnp.float32),
            banned_tokens=jnp.asarray(
                tuple(_pad_banned_tokens(tokens) for tokens in banned_token_rows),
                dtype=jnp.int32,
            ),
        )

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        self._raise_if_batched()
        logits = self._apply_banned_tokens(logits)
        logits = self._apply_temperature(logits)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        return self._apply_min_p(logits)

    def __call__(self, logits: Float[Array, " vocabulary"], *, keychain: Keychain) -> Int[Array, ""]:
        self._raise_if_batched()
        return jax.random.categorical(keychain.vmapped_keys, self.process_logits(logits))

    def _raise_if_batched(self) -> None:
        if self.temperature.ndim != 0:
            raise ValueError(
                "Attempted to call a method on a batched version of SamplingPolicy. Use vmap instead.",
            )

    def _apply_banned_tokens(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        (vocabulary_size,) = logits.shape
        vocabulary_indices = jnp.arange(vocabulary_size)
        banned_token_mask = jnp.any(
            self.banned_tokens[:, None] == vocabulary_indices,
            axis=0,
        )
        return jnp.where(banned_token_mask, -jnp.inf, logits)

    def _apply_temperature(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        (vocabulary_size,) = logits.shape
        best_token = jnp.argmax(logits, axis=-1)
        greedy_logits = jnp.where(jnp.arange(vocabulary_size) == best_token, 1.0, -jnp.inf)
        return jnp.where(
            self.temperature == 0.0,
            greedy_logits,
            logits / jnp.where(self.temperature == 0.0, 1.0, self.temperature),
        )

    def _apply_top_k(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        (vocabulary_size,) = logits.shape
        effective_top_k = jnp.clip(self.top_k, 1, vocabulary_size)
        sorted_logits = jnp.sort(logits, axis=-1, descending=True)
        min_logit = sorted_logits[effective_top_k - 1]
        filtered_logits = jnp.where(logits >= min_logit, logits, -jnp.inf)
        return jnp.where(self.top_k > 0, filtered_logits, logits)

    def _apply_top_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        sorted_indices = jnp.argsort(logits, axis=-1, descending=True)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        to_remove_sorted = cumulative_probs > self.top_p
        to_remove_sorted = jnp.roll(to_remove_sorted, shift=1, axis=-1)
        to_remove_sorted = to_remove_sorted.at[0].set(False)

        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        to_remove_unsorted = jnp.take_along_axis(to_remove_sorted, unsort_indices, axis=-1)

        return jnp.where(to_remove_unsorted, -jnp.inf, logits)

    def _apply_min_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        max_logit = jnp.max(logits)
        logit_cutoff = max_logit + jnp.log(self.min_p)
        filtered_logits = jnp.where(logits >= logit_cutoff, logits, -jnp.inf)
        return jnp.where(self.min_p == 0.0, logits, filtered_logits)


def _raise_if_different_lengths(*batches: tuple[object, ...]) -> None:
    lengths = tuple(len(batch) for batch in batches)
    first_length = lengths[0]
    if any(length != first_length for length in lengths):
        raise ValueError("init_batch iterable arguments must have the same length.")


def _pad_banned_tokens(banned_tokens: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(banned_tokens)
    if len(tokens) > _MAX_BANNED_TOKENS:
        raise ValueError(f"At most {_MAX_BANNED_TOKENS} banned tokens are supported.")
    if any(token < 0 for token in tokens):
        raise ValueError(f"Banned tokens must be non-negative token ids. {_SENTINEL} is reserved as a sentinel.")
    return tokens + (_SENTINEL,) * (_MAX_BANNED_TOKENS - len(tokens))
