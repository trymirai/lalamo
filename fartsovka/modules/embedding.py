from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .common import DEFAULT_PRECISION, DType

__all__ = ["Embedding", "EmbeddingFactory"]


class Embedding(eqx.Module):
    vocab_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)

    weights: Float[Array, "token_ids channels"]

    precision: DType = eqx.field(static=True, default=DEFAULT_PRECISION)

    def __init__(self, vocab_dim: int, model_dim: int, precision: DType, *, key: PRNGKeyArray) -> None:
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.precision = precision
        self.weights = jax.random.normal(key, (vocab_dim, model_dim), dtype=precision)

    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens model_dim"]:
        return self.weights[x]

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " token_ids"]:
        return self.weights @ x


@dataclass
class EmbeddingFactory:
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, vocab_dim: int, model_dim: int, *, key: PRNGKeyArray) -> Embedding:
        return Embedding(vocab_dim, model_dim, precision=self.precision, key=key)
