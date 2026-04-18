from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.weight_matrix import EmbeddingMatrix, WeightMatrix

from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class EmbeddingConfigBase(LalamoConfig):
    input_scale: float | None
    logit_soft_cap: float | None

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


class EmbeddingBase[ConfigT: EmbeddingConfigBase](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def embedding_matrix(self) -> EmbeddingMatrix: ...

    @property
    @abstractmethod
    def readout_matrix(self) -> WeightMatrix: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @eqx.filter_jit
    def embed(self, x: int | Int[Array, ""]) -> Float[Array, " channels"]:
        result = self.embedding_matrix.get_embedding(x)
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self.readout_matrix.dot(x)
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
        return self.embedding.dtype

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
        return self.input_embedding.dtype

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
