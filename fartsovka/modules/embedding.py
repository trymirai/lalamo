from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

__all__ = ["Embedding", "EmbeddingFactory", "QuantizedEmbedding", "QuantizedEmbeddingFactory"]


class EmbeddingBase(eqx.Module):
    vocab_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)

    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens model_dim"]:
        raise NotImplementedError

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " token_ids"]:
        raise NotImplementedError


@dataclass
class EmbeddingFactoryBase[EmbeddingType: EmbeddingBase]:
    def __call__(self, vocab_dim: int, model_dim: int, *, key: PRNGKeyArray) -> EmbeddingType:
        raise NotImplementedError


class Embedding(EmbeddingBase):
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
class EmbeddingFactory(EmbeddingFactoryBase[Embedding]):
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, vocab_dim: int, model_dim: int, *, key: PRNGKeyArray) -> Embedding:
        return Embedding(vocab_dim, model_dim, precision=self.precision, key=key)


class QuantizedEmbedding(EmbeddingBase):
    weights: Float[Array, "token_ids channels"]

    @property
    def int_weights(self) -> Int[Array, "token_ids channels"]:
        result = quantize_weights(self.weights, self.embedding_quantization_mode)
        return result.astype(self.embedding_quantization_mode.dtype)

    scales: Float[Array, " token_ids"]

    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)
    activation_precision: DType = eqx.field(static=True, default=DEFAULT_PRECISION)

    def __init__(
        self,
        *,
        vocab_dim: int,
        model_dim: int,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: QuantizationMode | None,
        activation_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.embedding_quantization_mode = embedding_quantization_mode
        self.activation_quantization_mode = activation_quantization_mode
        self.activation_precision = activation_precision
        min_val, max_val = embedding_quantization_mode.range
        self.weights = jax.random.uniform(
            key,
            (vocab_dim, model_dim),
            minval=min_val,
            maxval=max_val,
            dtype=activation_precision,
        )
        self.scales = jnp.ones((vocab_dim,), dtype=activation_precision)

    def prepare_weights(self) -> Float[Array, "out_channels in_channels"]:
        quantized_weights = quantize_weights(self.weights, self.embedding_quantization_mode)
        quantized_weights = quantized_weights * self.scales.reshape(-1, 1)
        return quantized_weights

    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens model_dim"]:
        return self.prepare_weights()[x]

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " token_ids"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return self.prepare_weights() @ x


@dataclass
class QuantizedEmbeddingFactory(EmbeddingFactoryBase[QuantizedEmbedding]):
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, vocab_dim: int, model_dim: int, *, key: PRNGKeyArray) -> QuantizedEmbedding:
        return QuantizedEmbedding(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
            activation_precision=self.activation_precision,
            key=key,
        )
