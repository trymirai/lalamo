from collections.abc import Iterable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

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
    repetition_penalty: Float[Array, "*batch"]
    presence_penalty: Float[Array, "*batch"]
    frequency_penalty: Float[Array, "*batch"]
    token_counts: Int[Array, "*batch vocabulary"] | None

    @classmethod
    def init(
        cls,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        banned_tokens: Iterable[int] | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> "SamplingPolicy":
        repetition_penalty = _normalize_repetition_penalty(repetition_penalty)
        return cls(
            temperature=jnp.asarray(1.0 if temperature is None else temperature, dtype=jnp.float32),
            top_k=jnp.asarray(0 if top_k is None else top_k, dtype=jnp.int32),
            top_p=jnp.asarray(1.0 if top_p is None else top_p, dtype=jnp.float32),
            min_p=jnp.asarray(0.0 if min_p is None else min_p, dtype=jnp.float32),
            banned_tokens=jnp.asarray(
                _pad_banned_tokens(()) if banned_tokens is None else _pad_banned_tokens(banned_tokens),
                dtype=jnp.int32,
            ),
            repetition_penalty=jnp.asarray(repetition_penalty, dtype=jnp.float32),
            presence_penalty=jnp.asarray(0.0 if presence_penalty is None else presence_penalty, dtype=jnp.float32),
            frequency_penalty=jnp.asarray(0.0 if frequency_penalty is None else frequency_penalty, dtype=jnp.float32),
            token_counts=None,
        )

    @classmethod
    def init_batch(
        cls,
        temperature: Iterable[float],
        top_k: Iterable[int],
        top_p: Iterable[float],
        min_p: Iterable[float],
        banned_tokens: Iterable[Iterable[int]],
        repetition_penalty: Iterable[float | None] | None = None,
        presence_penalty: Iterable[float | None] | None = None,
        frequency_penalty: Iterable[float | None] | None = None,
    ) -> "SamplingPolicy":
        temperatures = tuple(temperature)
        top_ks = tuple(top_k)
        top_ps = tuple(top_p)
        min_ps = tuple(min_p)
        banned_token_rows = tuple(tuple(row) for row in banned_tokens)
        batch_size = _raise_if_different_lengths(temperatures, top_ks, top_ps, min_ps, banned_token_rows)
        repetition_penalties = tuple(
            _normalize_repetition_penalty(penalty)
            for penalty in _fill_optional_batch(repetition_penalty, default=1.0, length=batch_size)
        )
        presence_penalties = _fill_optional_batch(presence_penalty, default=0.0, length=batch_size)
        frequency_penalties = _fill_optional_batch(frequency_penalty, default=0.0, length=batch_size)

        return cls(
            temperature=jnp.asarray(temperatures, dtype=jnp.float32),
            top_k=jnp.asarray(top_ks, dtype=jnp.int32),
            top_p=jnp.asarray(top_ps, dtype=jnp.float32),
            min_p=jnp.asarray(min_ps, dtype=jnp.float32),
            banned_tokens=jnp.asarray(
                tuple(_pad_banned_tokens(tokens) for tokens in banned_token_rows),
                dtype=jnp.int32,
            ),
            repetition_penalty=jnp.asarray(repetition_penalties, dtype=jnp.float32),
            presence_penalty=jnp.asarray(presence_penalties, dtype=jnp.float32),
            frequency_penalty=jnp.asarray(frequency_penalties, dtype=jnp.float32),
            token_counts=None,
        )

    def with_prompt_token_counts(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocabulary_size: int,
    ) -> "SamplingPolicy":
        token_ids = jnp.clip(prompt_token_ids, 0, vocabulary_size - 1)
        token_mask = (jnp.arange(prompt_token_ids.shape[0], dtype=jnp.int32) < prompt_length) & (
            (prompt_token_ids >= 0) & (prompt_token_ids < vocabulary_size)
        )
        token_counts = jnp.zeros(vocabulary_size, dtype=jnp.int32).at[token_ids].add(token_mask.astype(jnp.int32))
        return replace(self, token_counts=token_counts)

    def with_next_token_count(
        self,
        token_id: Int[Array, ""],
        should_count: Bool[Array, ""] | bool = True,
    ) -> "SamplingPolicy":
        if self.token_counts is None:
            return self
        in_vocabulary = (token_id >= 0) & (token_id < self.token_counts.shape[0])
        token_id = jnp.clip(token_id, 0, self.token_counts.shape[0] - 1)
        count = (jnp.asarray(should_count) & in_vocabulary).astype(jnp.int32)
        return replace(self, token_counts=self.token_counts.at[token_id].add(count))

    def has_count_penalties(self) -> bool:
        self._raise_if_batched()
        return bool(
            (self.repetition_penalty != 1.0).item()
            or (self.presence_penalty != 0.0).item()
            or (self.frequency_penalty != 0.0).item(),
        )

    def broadcast(self, batch_size: int) -> "SamplingPolicy":
        def broadcast_leaf(leaf: object) -> object:
            if isinstance(leaf, jax.Array):
                return jnp.broadcast_to(leaf, (batch_size, *leaf.shape))
            return leaf

        return jax.tree.map(broadcast_leaf, self)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        self._raise_if_batched()
        logits = self._apply_banned_tokens(logits)
        logits = self._apply_repetition_penalty(logits)
        logits = self._apply_presence_penalty(logits)
        logits = self._apply_frequency_penalty(logits)
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

    def _token_counts_or_zeros(self, logits: Float[Array, " vocabulary"]) -> Int[Array, " vocabulary"]:
        if self.token_counts is None:
            return jnp.zeros(logits.shape, dtype=jnp.int32)
        return self.token_counts

    def _apply_repetition_penalty(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        token_counts = self._token_counts_or_zeros(logits)
        seen_token_mask = token_counts > 0
        penalized_logits = jnp.where(logits > 0, logits / self.repetition_penalty, logits * self.repetition_penalty)
        return jnp.where(seen_token_mask, penalized_logits, logits)

    def _apply_presence_penalty(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        token_counts = self._token_counts_or_zeros(logits)
        return jnp.where(token_counts > 0, logits - self.presence_penalty, logits)

    def _apply_frequency_penalty(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        token_counts = self._token_counts_or_zeros(logits).astype(logits.dtype)
        return logits - self.frequency_penalty * token_counts

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


def _raise_if_different_lengths(*batches: tuple[object, ...]) -> int:
    lengths = tuple(len(batch) for batch in batches)
    first_length = lengths[0]
    if any(length != first_length for length in lengths):
        raise ValueError("init_batch iterable arguments must have the same length.")
    return first_length


def _fill_optional_batch(
    values: Iterable[float | None] | None,
    *,
    default: float,
    length: int,
) -> tuple[float, ...]:
    if values is None:
        return (default,) * length
    filled = tuple(default if value is None else value for value in values)
    if len(filled) != length:
        raise ValueError("init_batch iterable arguments must have the same length.")
    return filled


def _normalize_repetition_penalty(repetition_penalty: float | None) -> float:
    if repetition_penalty is None:
        return 1.0
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive.")
    return repetition_penalty


def _pad_banned_tokens(banned_tokens: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(banned_tokens)
    if len(tokens) > _MAX_BANNED_TOKENS:
        raise ValueError(f"At most {_MAX_BANNED_TOKENS} banned tokens are supported.")
    if any(token < 0 for token in tokens):
        raise ValueError(f"Banned tokens must be non-negative token ids. {_SENTINEL} is reserved as a sentinel.")
    return tokens + (_SENTINEL,) * (_MAX_BANNED_TOKENS - len(tokens))
