from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.arrays.embedding import CompressedEmbedding, FullPrecisionEmbedding, MLXQuantizedEmbedding
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations

from .common import (
    Initializer,
    LalamoModule,
    register_config_union,
)
from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "EmbeddingQuantConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class EmbeddingQuantConfig:
    group_size: int
    bits: int


def _make_embedding(
    initializer: Initializer,
    vocab_size: int,
    model_dim: int,
    quantization: EmbeddingQuantConfig | None,
) -> CompressedEmbedding:
    if quantization is not None:
        assert model_dim % quantization.group_size == 0
        model_groups = model_dim // quantization.group_size
        return MLXQuantizedEmbedding(
            weights=initializer.zeros((vocab_size, model_dim), initializer.precision),
            scales=initializer.ones((vocab_size, model_groups), initializer.precision),
            biases=initializer.zeros((vocab_size, model_groups), initializer.precision),
            group_size=quantization.group_size,
            bits=quantization.bits,
        )
    return FullPrecisionEmbedding(
        weights=initializer.normal(1.0, (vocab_size, model_dim), initializer.precision),
    )


@dataclass(frozen=True)
class EmbeddingConfigBase:
    input_scale: float | None
    logit_soft_cap: float | None
    activation_quantization_mode: QuantizationMode | None = None

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


class EmbeddingBase[ConfigT: EmbeddingConfigBase](LalamoModule[ConfigT]):
    @abstractmethod
    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]: ...

    @abstractmethod
    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self._prepare_input_weights()[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        logits = self._prepare_output_weights() @ x
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfigBase):
    quantization: EmbeddingQuantConfig | None = None

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "TiedEmbedding":
        embedding = _make_embedding(initializer, vocab_size, model_dim, self.quantization)
        return TiedEmbedding(config=self, embedding=embedding)


class TiedEmbedding(EmbeddingBase[TiedEmbeddingConfig]):
    embedding: CompressedEmbedding

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @property
    def model_dim(self) -> int:
        return self.embedding.model_dim

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.embedding.materialize()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.embedding.materialize()


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfigBase):
    input_quantization: EmbeddingQuantConfig | None = None
    output_quantization: EmbeddingQuantConfig | None = None

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "UntiedEmbedding":
        input_embedding = _make_embedding(initializer, vocab_size, model_dim, self.input_quantization)
        output_embedding = _make_embedding(initializer, vocab_size, model_dim, self.output_quantization)
        return UntiedEmbedding(
            config=self,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
        )


class UntiedEmbedding(EmbeddingBase[UntiedEmbeddingConfig]):
    input_embedding: CompressedEmbedding
    output_embedding: CompressedEmbedding

    @property
    def activation_precision(self) -> DTypeLike:
        return self.input_embedding.activation_precision

    @property
    def model_dim(self) -> int:
        return self.input_embedding.model_dim

    @property
    def vocab_size(self) -> int:
        return self.input_embedding.vocab_size

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.input_embedding.materialize()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.output_embedding.materialize()


EmbeddingConfig = TiedEmbeddingConfig | UntiedEmbeddingConfig

register_config_union(EmbeddingConfig)
