import math
from collections.abc import Iterable
from dataclasses import replace
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from lalamo.module import Keychain

__all__ = ["SamplingPolicy"]


_SENTINEL = -1
_MAX_BANNED_TOKENS = 16
type SamplingLeaf = Float[Array, "..."] | Int[Array, "..."]


class SamplingPolicy(eqx.Module):
    is_greedy: bool = eqx.field(static=True)
    temperature: Float[Array, "*batch"] | None = None
    top_k: Int[Array, "*batch"] | None = None
    top_p: Float[Array, "*batch"] | None = None
    min_p: Float[Array, "*batch"] | None = None
    banned_tokens: Int[Array, "*batch max_banned_tokens"] | None = None
    allowed_token_bitmask: Int[Array, "*batch vocabulary_words"] | None = None
    repetition_penalty: Float[Array, "*batch"] | None = None
    presence_penalty: Float[Array, "*batch"] | None = None
    frequency_penalty: Float[Array, "*batch"] | None = None
    logit_bias: Float[Array, "*batch vocabulary"] | None = None
    repetition_token_counts: Int[Array, "*batch vocabulary"] | None = None
    generated_token_counts: Int[Array, "*batch vocabulary"] | None = None
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
        logit_bias: Iterable[tuple[int, float]] | None = None,
        vocabulary_size: int | None = None,
    ) -> "SamplingPolicy":
        batch = cls.init_batch(
            temperature=(temperature,),
            top_k=(top_k,),
            top_p=(top_p,),
            min_p=(min_p,),
            banned_tokens=(banned_tokens,),
            repetition_penalty=(repetition_penalty,),
            presence_penalty=(presence_penalty,),
            frequency_penalty=(frequency_penalty,),
            logit_bias=(logit_bias,),
            vocabulary_size=vocabulary_size,
        )
        return replace(
            jax.tree.map(lambda leaf: leaf[0], batch),
            is_greedy=temperature == 0.0,
            token_history=(
                None
                if batch.repetition_penalty is None
                or suffix_repetition_length is None
                or suffix_repetition_length <= 0
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
        logit_bias: Iterable[Iterable[tuple[int, float]] | None] | None = None,
        vocabulary_size: int | None = None,
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
            "temperature": _optional_array(temperature, name="temperature", default=1.0),
            "top_k": _optional_array(top_k, name="top_k", default=0.0),
            "top_p": _optional_array(top_p, name="top_p", default=1.0),
            "min_p": _optional_array(min_p, name="min_p", default=0.0),
            "banned_tokens": banned_tokens_array,
            "repetition_penalty": _optional_array(repetition_penalty, name="repetition_penalty", default=1.0),
            "presence_penalty": _optional_array(presence_penalty, name="presence_penalty", default=0.0),
            "frequency_penalty": _optional_array(frequency_penalty, name="frequency_penalty", default=0.0),
            "logit_bias": _logit_bias_array(logit_bias, vocabulary_size),
        }
        _raise_if_different_batch_sizes(*jax.tree.leaves(arrays))
        return cls(is_greedy=False, **arrays)

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
        policy = self.with_empty_token_counts(vocabulary_size)
        if self.repetition_penalty is None:
            return policy
        positions = jnp.arange(prompt_token_ids.shape[0], dtype=jnp.int32)
        token_ids = jnp.clip(prompt_token_ids, 0, vocabulary_size - 1)
        in_vocabulary = (prompt_token_ids >= 0) & (prompt_token_ids < vocabulary_size)
        token_mask = (positions < prompt_length) & in_vocabulary
        if self.token_history is None:
            return replace(policy, repetition_token_counts=_count_tokens(token_ids, token_mask, vocabulary_size))

        window_size = self.token_history.shape[0]
        history_source = prompt_length - window_size + jnp.arange(window_size, dtype=jnp.int32)
        history = jnp.where(
            history_source >= 0,
            token_ids[jnp.clip(history_source, 0, prompt_token_ids.shape[0] - 1)],
            _SENTINEL,
        )
        suffix_mask = token_mask & (positions >= prompt_length - window_size)
        return replace(
            policy,
            repetition_token_counts=_count_tokens(token_ids, suffix_mask, vocabulary_size),
            token_history=history,
        )

    def with_empty_token_counts(self, vocabulary_size: int) -> "SamplingPolicy":
        return replace(
            self,
            repetition_token_counts=(
                None if self.repetition_penalty is None else jnp.zeros(vocabulary_size, dtype=jnp.int32)
            ),
            generated_token_counts=(
                None
                if self.presence_penalty is None and self.frequency_penalty is None
                else jnp.zeros(vocabulary_size, dtype=jnp.int32)
            ),
            token_history=None if self.token_history is None else jnp.full_like(self.token_history, _SENTINEL),
        )

    def with_next_token_count(
        self,
        token_id: Int[Array, ""],
        should_count: Bool[Array, ""] | bool = True,
    ) -> "SamplingPolicy":
        counts = self.repetition_token_counts
        if counts is None:
            counts = self.generated_token_counts
        if counts is None:
            return self
        vocabulary_size = counts.shape[0]
        in_vocabulary = (token_id >= 0) & (token_id < vocabulary_size)
        safe_token_id = jnp.clip(token_id, 0, vocabulary_size - 1)
        should_add = jnp.asarray(should_count) & in_vocabulary
        count = should_add.astype(jnp.int32)

        generated_counts = self.generated_token_counts
        if generated_counts is not None:
            generated_counts = generated_counts.at[safe_token_id].add(count)
        repetition_counts = self.repetition_token_counts
        history = self.token_history
        if repetition_counts is not None:
            repetition_counts = repetition_counts.at[safe_token_id].add(count)
            if history is not None:
                oldest_id = history[0]
                safe_oldest_id = jnp.clip(oldest_id, 0, vocabulary_size - 1)
                remove_count = (should_add & (oldest_id >= 0)).astype(jnp.int32)
                repetition_counts = repetition_counts.at[safe_oldest_id].add(-remove_count)
                shifted_history = jnp.concatenate([history[1:], safe_token_id[None]])
                history = jnp.where(should_add, shifted_history, history)
        return replace(
            self,
            repetition_token_counts=repetition_counts,
            generated_token_counts=generated_counts,
            token_history=history,
        )

    def broadcast(self, batch_size: int) -> "SamplingPolicy":
        def broadcast_leaf(leaf: object) -> object:
            if isinstance(leaf, jax.Array):
                return jnp.broadcast_to(leaf, (batch_size, *leaf.shape))
            return leaf

        return jax.tree.map(broadcast_leaf, self)

    def process_logits(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        self._raise_if_batched()
        logits = self._mask_tokens(logits)
        if (
            self.has_count_penalties
            or self.logit_bias is not None
            or (self.temperature is not None and not self.is_greedy)
        ):
            with jax.enable_x64(new_val=True):
                logits = self._apply_count_penalties(logits.astype(jnp.float64))
                logits = self._apply_logit_bias(logits)
                logits = _center_logits(self._apply_temperature(logits))
        else:
            logits = self._apply_temperature(logits)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        return self._apply_min_p(logits)

    def __call__(self, logits: Float[Array, " vocabulary"], *, keychain: Keychain) -> Int[Array, ""]:
        self._raise_if_batched()
        if self.is_greedy:
            logits = self._mask_tokens(logits)
            if self.has_count_penalties or self.logit_bias is not None:
                with jax.enable_x64(new_val=True):
                    logits = self._apply_count_penalties(logits.astype(jnp.float64))
                    logits = _center_logits(self._apply_logit_bias(logits))
            return jnp.argmax(logits).astype(jnp.int32)
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
        vector_fields: tuple[SamplingLeaf | None, ...] = (
            self.banned_tokens,
            self.allowed_token_bitmask,
            self.repetition_token_counts,
            self.generated_token_counts,
            self.token_history,
            self.logit_bias,
        )
        if any(field is not None and field.ndim != 0 for field in scalar_fields) or any(
            field is not None and field.ndim != 1 for field in vector_fields
        ):
            raise ValueError(
                "Attempted to call a method on a batched version of SamplingPolicy. Use vmap instead.",
            )

    def _mask_tokens(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        (vocabulary_size,) = logits.shape
        vocabulary_indices = jnp.arange(vocabulary_size, dtype=jnp.int32)
        if self.allowed_token_bitmask is not None:
            # XGrammar packs the vocabulary into int32 words, with the lowest token in the low bit.
            allowed = (self.allowed_token_bitmask[vocabulary_indices // 32] >> (vocabulary_indices % 32)) & 1
            logits = jnp.where(allowed != 0, logits, -jnp.inf)
        if self.banned_tokens is None:
            return logits
        banned_token_mask = jnp.any(
            self.banned_tokens[:, None] == vocabulary_indices,
            axis=0,
        )
        return jnp.where(banned_token_mask, -jnp.inf, logits)

    def _apply_count_penalties(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if not self.has_count_penalties:
            return logits
        if self.repetition_penalty is not None:
            seen_token_mask = (
                jnp.zeros(logits.shape, dtype=jnp.bool_)
                if self.repetition_token_counts is None
                else self.repetition_token_counts > 0
            )
            penalty = self.repetition_penalty.astype(logits.dtype)
            penalized_logits = jnp.where(logits > 0, logits / penalty, logits * penalty)
            logits = jnp.where(seen_token_mask, penalized_logits, logits)
        token_counts = self.generated_token_counts
        if token_counts is None:
            token_counts = jnp.zeros(logits.shape, dtype=jnp.int32)
        offsets = jnp.zeros_like(logits)
        if self.presence_penalty is not None:
            offsets = jnp.where(token_counts > 0, -self.presence_penalty.astype(logits.dtype), offsets)
        if self.frequency_penalty is not None:
            offsets -= self.frequency_penalty.astype(logits.dtype) * token_counts.astype(logits.dtype)
        # Remove common offsets before adding logits: even float64 loses 99 versus 100 beside 1e38.
        maximum = jnp.max(jnp.where(jnp.isneginf(logits), -jnp.inf, offsets))
        offsets -= jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        return logits + offsets

    def _apply_temperature(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature is None:
            return logits
        (vocabulary_size,) = logits.shape
        # Keep argmax operands and indices explicitly 32-bit across JAX's scoped x64 lowering.
        best_token = jax.lax.argmax(_center_logits(logits), axis=0, index_dtype=jnp.int32)
        greedy_logits = jnp.where(jnp.arange(vocabulary_size, dtype=jnp.int32) == best_token, 1.0, -jnp.inf)
        maximum = jnp.max(logits)
        logits -= jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        temperature = self.temperature.astype(logits.dtype)
        return jnp.where(
            temperature == 0.0,
            greedy_logits,
            logits / jnp.where(temperature == 0.0, 1.0, temperature),
        )

    def _apply_logit_bias(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.logit_bias is None:
            return logits
        maximum = jnp.max(logits)
        logits -= jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        return logits + self.logit_bias.astype(logits.dtype)

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

        to_remove_sorted = (cumulative_probs_before_token >= self.top_p) & (jnp.arange(logits.shape[0]) > 0)

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


def _center_logits(logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
    maximum = jnp.max(logits)
    centered = logits - jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    return jnp.where(jnp.isneginf(logits), -jnp.inf, jnp.maximum(centered, float(jnp.finfo(jnp.float32).min))).astype(
        jnp.float32
    )


def _optional_array(
    values: Iterable[int | float | None] | None,
    *,
    name: Literal[
        "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"
    ],
    default: float,
) -> SamplingLeaf | None:
    if values is None:
        return None
    resolved_values = tuple(default if value is None else value for value in values)
    if name == "top_k":
        resolved_values = tuple(min(max(value, 0), int(jnp.iinfo(jnp.int32).max)) for value in resolved_values)
        dtype = jnp.int32
    else:
        if any(not math.isfinite(value) for value in resolved_values):
            raise ValueError(f"{name} must be finite in float32.")
        if name == "top_p":
            resolved_values = tuple(min(value, 1.0) for value in resolved_values)
        elif name == "min_p":
            resolved_values = tuple(max(value, 0.0) for value in resolved_values)
        limits = jnp.finfo(jnp.float32)
        if any(abs(value) > float(limits.max) for value in resolved_values):
            raise ValueError(f"{name} must be finite in float32.")
        # JAX arithmetic can flush representable subnormal controls to zero.
        if name == "repetition_penalty" and any(value < float(limits.tiny) for value in resolved_values):
            raise ValueError(f"{name} must be positive and normal in float32.")
        if name in ("temperature", "top_p", "min_p") and any(
            value < 0.0 or 0.0 < value < float(limits.tiny) for value in resolved_values
        ):
            raise ValueError(f"{name} must be zero or positive and normal in float32.")
        dtype = jnp.float32
    if all(value == default for value in resolved_values):
        return None
    return jnp.asarray(resolved_values, dtype=dtype)


def _raise_if_different_batch_sizes(*arrays: SamplingLeaf) -> None:
    if arrays and any(array.shape[0] != arrays[0].shape[0] for array in arrays):
        raise ValueError("init_batch iterable arguments must have the same length.")


def _logit_bias_array(
    values: Iterable[Iterable[tuple[int, float]] | None] | None,
    vocabulary_size: int | None,
) -> Float[Array, "batch vocabulary"] | None:
    if values is None:
        return None
    rows = tuple(tuple(row or ()) for row in values)
    if not any(rows):
        return None
    if vocabulary_size is None or vocabulary_size < 1:
        raise ValueError("logit_bias requires a positive vocabulary_size.")
    for row in rows:
        if any(token_id < 0 or token_id >= vocabulary_size for token_id, _ in row):
            raise ValueError("logit_bias token ids must be within the vocabulary.")
        if len({token_id for token_id, _ in row}) != len(row):
            raise ValueError("logit_bias token ids must be distinct.")
        if any(not math.isfinite(bias) or not -100.0 <= bias <= 100.0 for _, bias in row):
            raise ValueError("logit_bias values must be finite and between -100 and 100.")
    if all(bias == 0.0 for row in rows for _, bias in row):
        return None
    return jnp.stack(
        [
            jnp.zeros(vocabulary_size, dtype=jnp.float32)
            .at[jnp.asarray([token_id for token_id, _ in row], dtype=jnp.int32)]
            .set(jnp.asarray([bias for _, bias in row], dtype=jnp.float32))
            for row in rows
        ]
    )


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
