from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = ["GumbelSampler"]


@dataclass(frozen=True)
class GumbelSampler:
    seed: int
    temperature: float = 1.0

    def sample_logits(
        self,
        logits: Float[Array, "batch vocab"],
        positions: Int[Array, " batch"],
    ) -> Int[Array, " batch"]:
        if self.temperature < 1e-5:
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        base_key = jax.random.key(self.seed & 0xFFFFFFFF)
        keys = jax.vmap(lambda position: jax.random.fold_in(base_key, position))(positions)
        noise = jax.vmap(
            lambda key, row_logits: jax.random.gumbel(key, row_logits.shape, dtype=jnp.float32),
        )(keys, logits)
        scores = logits.astype(jnp.float32) / self.temperature + noise
        return jnp.argmax(scores, axis=-1).astype(jnp.int32)
