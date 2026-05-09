from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = ["GumbelSampler", "GumbelSeed"]


@dataclass(frozen=True)
class GumbelSeed:
    value: int

    def for_position(self, position: int) -> "GumbelSeed":
        mixed = (self.value ^ (position * 0x9E3779B185EBCA87)) & 0xFFFFFFFFFFFFFFFF
        mixed = ((mixed >> 30) ^ mixed) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        mixed = ((mixed >> 27) ^ mixed) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return GumbelSeed(((mixed >> 31) ^ mixed) & 0xFFFFFFFFFFFFFFFF)

    def sample_logits(self, logits: Float[Array, " vocab"], temperature: float = 1.0) -> int:
        logits_array = jnp.asarray(logits)
        if temperature < 1e-5:
            return int(jnp.argmax(logits_array, axis=-1))

        key = jax.random.key(self.value & 0xFFFFFFFF)
        noise = jax.random.gumbel(key, logits_array.shape, dtype=jnp.float32)
        scores = logits_array.astype(jnp.float32) / temperature + noise
        return int(jnp.argmax(scores, axis=-1))


@dataclass(frozen=True)
class GumbelSampler:
    seed: GumbelSeed
    temperature: float = 1.0

    def seed_for_position(self, position: int) -> GumbelSeed:
        return self.seed.for_position(position)

    def sample_logits(self, logits: Float[Array, " vocab"], position: int) -> int:
        return self.seed_for_position(position).sample_logits(logits, self.temperature)

    def rank_logits(self, logits: Float[Array, " vocab"], position: int, k: int) -> tuple[int, ...]:
        logits_array = np.asarray(logits, dtype=np.float32)
        if self.temperature < 1e-5:
            scores = logits_array
        else:
            key = jax.random.key(self.seed_for_position(position).value & 0xFFFFFFFF)
            noise = np.asarray(jax.random.gumbel(key, logits_array.shape, dtype=jnp.float32))
            scores = logits_array / self.temperature + noise

        num_returned = min(k, scores.shape[0])
        candidates = (
            np.argpartition(-scores, num_returned - 1)[:num_returned]
            if num_returned < scores.shape[0]
            else np.arange(scores.shape[0], dtype=np.int64)
        )
        ordered = candidates[np.lexsort((candidates, -scores[candidates]))]
        return tuple(int(token_id) for token_id in ordered)

    def rank_from_probs(
        self,
        probs: Mapping[int, float],
        vocab_size: int,
        position: int,
        k: int,
    ) -> tuple[int, ...]:
        tokens = np.fromiter(probs.keys(), dtype=np.int64, count=len(probs))
        log_probs = np.log(np.fromiter(probs.values(), dtype=np.float32, count=len(probs)))
        key = jax.random.key(self.seed_for_position(position).value & 0xFFFFFFFF)
        noise = np.asarray(jax.random.gumbel(key, (vocab_size,), dtype=jnp.float32))
        scores = log_probs + noise[tokens]

        num_returned = min(k, len(tokens))
        candidates = (
            np.argpartition(-scores, num_returned - 1)[:num_returned]
            if num_returned < len(tokens)
            else np.arange(len(tokens), dtype=np.int64)
        )
        ordered = candidates[np.lexsort((tokens[candidates], -scores[candidates]))]
        return tuple(int(token) for token in tokens[ordered])
