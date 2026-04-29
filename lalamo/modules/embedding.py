from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import EmbeddingMatrix, FullPrecisionSpec, Layout, MatmulConfig, WeightMatrix

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

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


@dataclass(frozen=True)
class EmbeddingForwardPassConfig:
    activation_dtype: DTypeLike = jnp.bfloat16
    embedding: MatmulConfig = dataclass_field(default_factory=MatmulConfig)
    logit_dtype: DTypeLike = jnp.float32
    readout: MatmulConfig = dataclass_field(default_factory=MatmulConfig)


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

    @eqx.filter_jit
    def embed(
        self,
        x: int | Int[Array, ""],
        *,
        keychain: Keychain,
        forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
    ) -> Float[Array, " channels"]:
        result = self.embedding_matrix.lookup_embedding(
            x,
            dtype=forward_pass_config.activation_dtype,
            keychain=keychain,
            forward_pass_config=forward_pass_config.embedding,
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
        logits = self.readout_matrix.dot(
            x,
            keychain=keychain,
            forward_pass_config=forward_pass_config.readout,
        )
        logits = logits.astype(forward_pass_config.logit_dtype)
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfig):
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "TiedEmbedding":
        embedding = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).init(initializer, (), vocab_size, model_dim)
        return TiedEmbedding(config=self, embedding=embedding)


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
        model_dim, _ = self.embedding.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        _, vocab_size = self.embedding.shape
        return vocab_size


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfig):
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "UntiedEmbedding":
        input_embedding = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).init(initializer, (), vocab_size, model_dim)
        output_embedding = FullPrecisionSpec().init(initializer, (), vocab_size, model_dim)
        return UntiedEmbedding(
            config=self,
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
        model_dim, _ = self.input_embedding.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        _, vocab_size = self.input_embedding.shape
        return vocab_size
