from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    GradientEstimator,
    MatmulConfig,
    WeightMatrix,
)

from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "EmbeddingForwardPassConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class EmbeddingConfig(LalamoConfig, RegistryABC):
    input_scale: float | None
    logit_soft_cap: float | None
    logit_scale: float = 1.0

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        vocab_size: int,
    ) -> "EmbeddingBase": ...


@dataclass(frozen=True)
class EmbeddingForwardPassConfig:
    activation_dtype: DTypeLike = jnp.bfloat16
    logit_dtype: DTypeLike = jnp.float32
    matmul_config: MatmulConfig = dataclass_field(default_factory=MatmulConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            activation_dtype=jnp.float32,
            matmul_config=MatmulConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(cls, precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT) -> Self:
        return cls(matmul_config=MatmulConfig.for_inference(precision))

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            matmul_config=MatmulConfig.for_training(gradient_estimator, precision),
        )


class EmbeddingBase[ConfigT: EmbeddingConfig](LalamoModule[ConfigT]):
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

    def _readout_logits(
        self,
        x: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig,
    ) -> Float[Array, " vocabulary"]:
        return self.readout_matrix.dot(
            x,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )

    @eqx.filter_jit
    def embed(
        self,
        x: int | Int[Array, "*tokens"],
        *,
        keychain: Keychain,
        forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
    ) -> Float[Array, "*tokens channels"]:
        result = self.embedding_matrix.lookup_embedding(
            x,
            dtype=forward_pass_config.activation_dtype,
            keychain=keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(
        self,
        x: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
    ) -> Float[Array, " vocabulary"]:
        logits = self._readout_logits(
            x,
            keychain=keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        logits = logits.astype(forward_pass_config.logit_dtype)
        logits = logits * jax.lax.stop_gradient(self.config.logit_scale)
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfig):
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        vocab_size: int,
    ) -> "TiedEmbedding":
        embedding = initializer.embedding_matrix(vocab_size, model_dim)
        return TiedEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=embedding,
        )


class TiedEmbedding(EmbeddingBase[TiedEmbeddingConfig]):
    embedding: EmbeddingMatrix

    @property
    def embedding_matrix(self) -> EmbeddingMatrix:
        return self.embedding

    @property
    def readout_matrix(self) -> WeightMatrix:
        return self.embedding

    @property
    def model_dim(self) -> int:
        _, model_dim = self.embedding.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.embedding.shape
        return vocab_size

    def _readout_logits(
        self,
        x: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig,
    ) -> Float[Array, " vocabulary"]:
        return self.embedding.dot(
            x,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
            transposed=True,
        )


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfig):
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        vocab_size: int,
    ) -> "UntiedEmbedding":
        input_embedding = initializer.embedding_matrix(vocab_size, model_dim)
        output_embedding = initializer.weight_matrix(vocab_size, model_dim)
        return UntiedEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
        )


class UntiedEmbedding(EmbeddingBase[UntiedEmbeddingConfig]):
    input_embedding: EmbeddingMatrix
    output_embedding: WeightMatrix

    @property
    def embedding_matrix(self) -> EmbeddingMatrix:
        return self.input_embedding

    @property
    def readout_matrix(self) -> WeightMatrix:
        return self.output_embedding

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_embedding.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_embedding.shape
        return vocab_size
