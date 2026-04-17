from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class GumbelSeed:
    """Seed for Gumbel-noise sampling.

    ``derive(depth)`` branches the seed for a child position; ``advance(ctx_len)``
    advances after a verify step. Both use MurmurHash3-derived mixers. Pin-tested
    for bit-reproducibility — do not change the constants without updating the
    test in ``tests/unit/test_sampler.py``.
    """

    value: int

    def derive(self, depth: int) -> "GumbelSeed":
        h = self.value + depth * 2654435761
        h = ((h >> 16) ^ h) * 0x45D9F3B37197344D & 0xFFFFFFFFFFFFFFFF
        h = ((h >> 16) ^ h) * 0x45D9F3B37197344D & 0xFFFFFFFFFFFFFFFF
        return GumbelSeed(((h >> 16) ^ h) & 0xFFFFFFFFFFFFFFFF)

    def advance(self, context_len: int) -> "GumbelSeed":
        return GumbelSeed((self.value * 2654435761 ^ context_len) & 0xFFFFFFFFFFFFFFFF)

    def sample(self, logits: Float[Array, " vocab"]) -> int:
        """Single-position Gumbel-max sample."""
        (vocab,) = logits.shape
        key = jax.random.key(self.value & 0xFFFFFFFF)
        noise = jax.random.gumbel(key, (vocab,), dtype=jnp.float32)
        return int(jnp.argmax(logits.astype(jnp.float32) + noise))


@dataclass(frozen=True)
class GumbelMaxSampler:
    temperature: float = 1.0

    def sample(
        self,
        logits: Float[Array, "n vocab"],
        seeds: Int[Array, " n"],
    ) -> Int[Array, " n"]:
        if self.temperature < 1e-5:
            return jnp.argmax(logits, axis=-1)
        scaled = logits.astype(jnp.float32) / self.temperature
        _, vocab = scaled.shape
        keys = jax.vmap(jax.random.key)(seeds)
        noise = jax.vmap(lambda k: jax.random.gumbel(k, (vocab,), dtype=jnp.float32))(keys)
        return jnp.argmax(scaled + noise, axis=-1)


def gumbel_rank_from_probs(seed: GumbelSeed, probs: dict[int, float], vocab_size: int, k: int) -> list[int]:
    """Rank ``probs`` candidates by ``log(prob) + gumbel(seed)[token_id]``.

    Uses the same Gumbel noise array the verifier sees (``jax.random.gumbel(key,
    (vocab_size,))``) so shared-seed tie-breaking works, but scores only the
    shortlist — top-k is taken within ``probs.keys()``.
    """
    tokens = np.fromiter(probs.keys(), dtype=np.int32, count=len(probs))
    log_probs = np.log(np.fromiter(probs.values(), dtype=np.float32, count=len(probs)))
    key = jax.random.key(seed.value & 0xFFFFFFFF)
    noise = np.asarray(jax.random.gumbel(key, (vocab_size,), dtype=jnp.float32))
    scores = log_probs + noise[tokens]
    k = min(k, len(tokens))
    top = np.argpartition(-scores, k - 1)[:k] if k < len(tokens) else np.argsort(-scores)
    order = top[np.argsort(-scores[top])]
    return tokens[order].tolist()
