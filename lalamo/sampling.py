from collections.abc import Iterable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int

from lalamo.module import Keychain

__all__ = ["SamplingPolicy"]


_SENTINEL = -1
_MAX_BANNED_TOKENS = 16
type SamplingLeaf = Float[Array, "..."] | Int[Array, "..."]


class SamplingPolicy(eqx.Module):
    temperature: Float[Array, "*batch"] | None = None
    top_k: Int[Array, "*batch"] | None = None
    top_p: Float[Array, "*batch"] | None = None
    min_p: Float[Array, "*batch"] | None = None
    banned_tokens: Int[Array, "*batch max_banned_tokens"] | None = None
    repetition_penalty: Float[Array, "*batch"] | None = None
    presence_penalty: Float[Array, "*batch"] | None = None
    frequency_penalty: Float[Array, "*batch"] | None = None
    token_counts: Int[Array, "*batch vocabulary"] | None = None
    token_history: Int[Array, "*batch suffix"] | None = None

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
        suffix_repetition_length: int | None = None,
    ) -> "SamplingPolicy":
        banned_tokens = () if banned_tokens is None else tuple(banned_tokens)
        repetition_penalty = 1.0 if repetition_penalty is None else repetition_penalty
        presence_penalty = 0.0 if presence_penalty is None else presence_penalty
        frequency_penalty = 0.0 if frequency_penalty is None else frequency_penalty
        if repetition_penalty is not None and repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive.")
        return cls(
            temperature=(
                None if temperature is None or temperature == 1.0 else jnp.asarray(temperature, dtype=jnp.float32)
            ),
            top_k=None if top_k is None or top_k <= 0 else jnp.asarray(top_k, dtype=jnp.int32),
            top_p=None if top_p is None or top_p >= 1.0 else jnp.asarray(top_p, dtype=jnp.float32),
            min_p=None if min_p is None or min_p <= 0.0 else jnp.asarray(min_p, dtype=jnp.float32),
            banned_tokens=(
                None if not banned_tokens else jnp.asarray(_pad_banned_tokens(banned_tokens), dtype=jnp.int32)
            ),
            repetition_penalty=(
                None if repetition_penalty == 1.0 else jnp.asarray(repetition_penalty, dtype=jnp.float32)
            ),
            presence_penalty=None if presence_penalty == 0.0 else jnp.asarray(presence_penalty, dtype=jnp.float32),
            frequency_penalty=None if frequency_penalty == 0.0 else jnp.asarray(frequency_penalty, dtype=jnp.float32),
            token_counts=None,
            token_history=(
                None
                if suffix_repetition_length is None or suffix_repetition_length <= 0
                else jnp.full(suffix_repetition_length, _SENTINEL, dtype=jnp.int32)
            ),
        )

    @classmethod
    def init_batch(
        cls,
        temperature: Iterable[float | None] | None = None,
        top_k: Iterable[int | None] | None = None,
        top_p: Iterable[float | None] | None = None,
        min_p: Iterable[float | None] | None = None,
        banned_tokens: Iterable[Iterable[int] | None] | None = None,
        repetition_penalty: Iterable[float | None] | None = None,
        presence_penalty: Iterable[float | None] | None = None,
        frequency_penalty: Iterable[float | None] | None = None,
    ) -> "SamplingPolicy":
        empty_banned_tokens = _pad_banned_tokens(())
        padded_banned_tokens = (
            None
            if banned_tokens is None
            else tuple(empty_banned_tokens if row is None else _pad_banned_tokens(row) for row in banned_tokens)
        )
        banned_tokens_array = (
            None
            if padded_banned_tokens is None or all(row == empty_banned_tokens for row in padded_banned_tokens)
            else jnp.asarray(padded_banned_tokens, dtype=jnp.int32)
        )
        arrays = {
            "temperature": _optional_array(temperature, default=1.0, dtype=jnp.float32),
            "top_k": _optional_array(top_k, default=0, dtype=jnp.int32),
            "top_p": _optional_array(top_p, default=1.0, dtype=jnp.float32),
            "min_p": _optional_array(min_p, default=0.0, dtype=jnp.float32),
            "banned_tokens": banned_tokens_array,
            "repetition_penalty": _optional_array(repetition_penalty, default=1.0, dtype=jnp.float32),
            "presence_penalty": _optional_array(presence_penalty, default=0.0, dtype=jnp.float32),
            "frequency_penalty": _optional_array(frequency_penalty, default=0.0, dtype=jnp.float32),
        }
        _raise_if_different_batch_sizes(*jax.tree.leaves(arrays))
        return cls(token_counts=None, token_history=None, **arrays)

    @property
    def has_count_penalties(self) -> bool:
        return (
            self.repetition_penalty is not None
            or self.presence_penalty is not None
            or self.frequency_penalty is not None
        )

    def with_prompt_token_counts(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocabulary_size: int,
    ) -> "SamplingPolicy":
        positions = jnp.arange(prompt_token_ids.shape[0], dtype=jnp.int32)
        token_ids = jnp.clip(prompt_token_ids, 0, vocabulary_size - 1)
        in_vocabulary = (prompt_token_ids >= 0) & (prompt_token_ids < vocabulary_size)
        token_mask = (positions < prompt_length) & in_vocabulary
        if self.token_history is None:
            return replace(self, token_counts=_count_tokens(token_ids, token_mask, vocabulary_size))

        window_size = self.token_history.shape[0]
        history_source = prompt_length - window_size + jnp.arange(window_size, dtype=jnp.int32)
        history = jnp.where(
            history_source >= 0,
            token_ids[jnp.clip(history_source, 0, prompt_token_ids.shape[0] - 1)],
            _SENTINEL,
        )
        suffix_mask = token_mask & (positions >= prompt_length - window_size)
        return replace(
            self,
            token_counts=_count_tokens(token_ids, suffix_mask, vocabulary_size),
            token_history=history,
        )

    def with_empty_token_counts(self, vocabulary_size: int) -> "SamplingPolicy":
        return replace(
            self,
            token_counts=jnp.zeros(vocabulary_size, dtype=jnp.int32),
            token_history=None if self.token_history is None else jnp.full_like(self.token_history, _SENTINEL),
        )

    def with_next_token_count(
        self,
        token_id: Int[Array, ""],
        should_count: Bool[Array, ""] | bool = True,
    ) -> "SamplingPolicy":
        if self.token_counts is None:
            return self
        vocabulary_size = self.token_counts.shape[0]
        in_vocabulary = (token_id >= 0) & (token_id < vocabulary_size)
        safe_token_id = jnp.clip(token_id, 0, vocabulary_size - 1)
        should_add = jnp.asarray(should_count) & in_vocabulary
        count = should_add.astype(jnp.int32)

        if self.token_history is None:
            return replace(self, token_counts=self.token_counts.at[safe_token_id].add(count))

        oldest_id = self.token_history[0]
        safe_oldest_id = jnp.clip(oldest_id, 0, vocabulary_size - 1)
        remove_count = (should_add & (oldest_id >= 0)).astype(jnp.int32)
        token_counts = self.token_counts.at[safe_token_id].add(count).at[safe_oldest_id].add(-remove_count)
        shifted_history = jnp.concatenate([self.token_history[1:], safe_token_id[None]])
        return replace(
            self,
            token_counts=token_counts,
            token_history=jnp.where(should_add, shifted_history, self.token_history),
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
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        logits = self._apply_min_p(logits)
        return self._apply_temperature(logits)

    def __call__(self, logits: Float[Array, " vocabulary"], *, keychain: Keychain) -> Int[Array, ""]:
        self._raise_if_batched()
        return jax.random.categorical(keychain.vmapped_keys, self.process_logits(logits))

    def _raise_if_batched(self) -> None:
        scalar_fields: tuple[SamplingLeaf | None, ...] = (
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        )
        vector_fields: tuple[SamplingLeaf | None, ...] = (self.banned_tokens, self.token_counts, self.token_history)
        if any(field is not None and field.ndim != 0 for field in scalar_fields) or any(
            field is not None and field.ndim != 1 for field in vector_fields
        ):
            raise ValueError(
                "Attempted to call a method on a batched version of SamplingPolicy. Use vmap instead.",
            )

    def _apply_banned_tokens(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.banned_tokens is None:
            return logits
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
        if self.repetition_penalty is None:
            return logits
        token_counts = self._token_counts_or_zeros(logits)
        seen_token_mask = token_counts > 0
        penalized_logits = jnp.where(logits > 0, logits / self.repetition_penalty, logits * self.repetition_penalty)
        return jnp.where(seen_token_mask, penalized_logits, logits)

    def _apply_presence_penalty(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.presence_penalty is None:
            return logits
        token_counts = self._token_counts_or_zeros(logits)
        return jnp.where(token_counts > 0, logits - self.presence_penalty, logits)

    def _apply_frequency_penalty(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.frequency_penalty is None:
            return logits
        token_counts = self._token_counts_or_zeros(logits).astype(logits.dtype)
        return logits - self.frequency_penalty * token_counts

    def _apply_temperature(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature is None:
            return logits
        (vocabulary_size,) = logits.shape
        best_token = jnp.argmax(logits, axis=-1)
        greedy_logits = jnp.where(jnp.arange(vocabulary_size) == best_token, 1.0, -jnp.inf)
        return jnp.where(
            self.temperature == 0.0,
            greedy_logits,
            logits / jnp.where(self.temperature == 0.0, 1.0, self.temperature),
        )

    def _apply_top_k(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.top_k is None:
            return logits
        (vocabulary_size,) = logits.shape
        effective_top_k = jnp.clip(self.top_k, 1, vocabulary_size)
        sorted_indices = jnp.argsort(logits, axis=-1, descending=True)
        ranks = jnp.empty_like(sorted_indices).at[sorted_indices].set(jnp.arange(vocabulary_size, dtype=jnp.int32))
        filtered_logits = jnp.where(ranks < effective_top_k, logits, -jnp.inf)
        return jnp.where(self.top_k > 0, filtered_logits, logits)

    def _apply_top_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.top_p is None:
            return logits
        sorted_indices = jnp.argsort(logits, axis=-1, descending=True)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        cumulative_probs_before_token = cumulative_probs - sorted_probs

        to_remove_sorted = cumulative_probs_before_token >= self.top_p

        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        to_remove_unsorted = jnp.take_along_axis(to_remove_sorted, unsort_indices, axis=-1)

        return jnp.where(to_remove_unsorted, -jnp.inf, logits)

    def _apply_min_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.min_p is None:
            return logits
        max_logit = jnp.max(logits)
        logit_cutoff = max_logit + jnp.log(self.min_p)
        filtered_logits = jnp.where(logits >= logit_cutoff, logits, -jnp.inf)
        return jnp.where(self.min_p == 0.0, logits, filtered_logits)


def _optional_array[T](values: Iterable[T | None] | None, *, default: T, dtype: DTypeLike) -> SamplingLeaf | None:
    if values is None:
        return None
    values = tuple(default if value is None else value for value in values)
    if all(value == default for value in values):
        return None
    return jnp.asarray(values, dtype=dtype)


def _raise_if_different_batch_sizes(*arrays: SamplingLeaf) -> None:
    if arrays and any(array.shape[0] != arrays[0].shape[0] for array in arrays):
        raise ValueError("init_batch iterable arguments must have the same length.")


def _pad_banned_tokens(banned_tokens: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(banned_tokens)
    if len(tokens) > _MAX_BANNED_TOKENS:
        raise ValueError(f"At most {_MAX_BANNED_TOKENS} banned tokens are supported.")
    if any(token < 0 for token in tokens):
        raise ValueError(f"Banned tokens must be non-negative token ids. {_SENTINEL} is reserved as a sentinel.")
    return tokens + (_SENTINEL,) * (_MAX_BANNED_TOKENS - len(tokens))


def _count_tokens(
    token_ids: Int[Array, " tokens"],
    token_mask: Bool[Array, " tokens"],
    vocabulary_size: int,
) -> Int[Array, " vocabulary"]:
    return jnp.zeros(vocabulary_size, dtype=jnp.int32).at[token_ids].add(token_mask.astype(jnp.int32))
