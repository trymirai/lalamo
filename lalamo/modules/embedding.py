from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int, Int32

from lalamo.initializer import Initializer
from lalamo.kernels.low_rank_readout import make_block_topk, make_gathered_dot
from lalamo.module import Keychain, LalamoConfig, LalamoModule, field
from lalamo.sampling import DenseLogits, Logits, SparseLogits
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import (
    EmbeddingMatrix,
    FullPrecisionMatrix,
    GradientEstimator,
    MatmulConfig,
    WeightMatrix,
)

from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "EmbeddingForwardPassConfig",
    "Sparsifier",
    "SparsifierSpec",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class SparsifierSpec(LalamoConfig):
    rank: int = 256
    selection_block_size: int = 1_024
    exact_tokens_per_block: int = 32

    def init(self, initializer: Initializer, model_dim: int, vocab_size: int) -> "Sparsifier":
        if not 0 < self.rank <= model_dim or self.rank % 64:
            raise ValueError("rank must fit model_dim and be divisible by 64.")
        if self.selection_block_size <= 0 or not 0 < self.exact_tokens_per_block <= self.selection_block_size:
            raise ValueError("exact_tokens_per_block must be positive and fit selection_block_size.")
        return Sparsifier(
            config=self,
            sharding_config=initializer.sharding_config,
            hidden_projection=initializer.weight_matrix(
                self.rank,
                model_dim,
                dtype=jnp.bfloat16,
                is_sharded=False,
            ),
            preview=initializer.weight_matrix(
                vocab_size,
                self.rank,
                dtype=jnp.bfloat16,
                is_sharded=False,
            ),
            token_ids=jax.device_put(
                jnp.arange(vocab_size, dtype=jnp.int32),
                initializer.sharding_config.resolve_sharding((None,)),
            ),
        )


class Sparsifier(LalamoModule[SparsifierSpec]):
    hidden_projection: WeightMatrix
    preview: WeightMatrix
    token_ids: Int32[Array, " vocabulary"] = field(trainable=False)

    @property
    def candidate_count(self) -> int:
        full_blocks, tail_size = divmod(self.token_ids.shape[0], self.config.selection_block_size)
        return full_blocks * self.config.exact_tokens_per_block + min(
            tail_size,
            self.config.exact_tokens_per_block,
        )

    def __call__(
        self,
        vector: Float[Array, " channels"],
        readout: WeightMatrix,
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig,
    ) -> SparseLogits:
        projected = self.hidden_projection.dot(
            vector,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        scores = self.preview.dot(
            projected,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        block_size = self.config.selection_block_size
        full_blocks, tail_size = divmod(scores.shape[0], block_size)
        block_count = full_blocks + bool(tail_size)
        scores = jnp.where(jnp.isneginf(scores), jnp.finfo(scores.dtype).min, scores)
        padding = block_count * block_size - scores.shape[0]
        block_scores = jnp.pad(scores, (0, padding), constant_values=-jnp.inf).reshape(block_count, block_size)
        device = self.sharding_config.mesh.devices.flat[0]
        use_mosaic = device.device_kind.startswith("NVIDIA") and float(getattr(device, "compute_capability", 0)) >= 9
        if (
            128 <= block_size <= 4_096
            and block_size & (block_size - 1) == 0
            and block_scores.dtype == jnp.bfloat16
            and use_mosaic
        ):
            local_rows = make_block_topk(block_count, block_size, self.config.exact_tokens_per_block)(block_scores)
        else:
            _, local_rows = jax.lax.top_k(block_scores, self.config.exact_tokens_per_block)
        rows = (local_rows + block_size * jnp.arange(block_count, dtype=jnp.int32)[:, None]).reshape(-1)
        rows = rows[: self.candidate_count]
        token_ids = self.token_ids[rows]

        if (
            isinstance(readout, FullPrecisionMatrix)
            and readout.weights.dtype == jnp.bfloat16
            and vector.dtype == jnp.bfloat16
            and vector.shape[0] % 128 == 0
            and forward_pass_config.precision == DotAlgorithmPreset.DEFAULT
            and use_mosaic
        ):
            values = make_gathered_dot(token_ids.shape[0], vector.shape[0])(readout.weights, token_ids, vector)
        else:
            if not isinstance(readout, EmbeddingMatrix):
                raise TypeError("Sparsified readout requires an embedding-compatible matrix.")
            weights = readout.lookup_embedding(
                token_ids,
                dtype=vector.dtype,
                keychain=keychain,
                forward_pass_config=forward_pass_config,
            )
            with use_dot_algorithm_preset(forward_pass_config.precision):
                values = weights @ vector
        return SparseLogits(values=values, token_ids=token_ids)


@dataclass(frozen=True)
class EmbeddingConfig(LalamoConfig, RegistryABC):
    input_scale: float | None
    logit_soft_cap: float | None
    sparsifier_spec: SparsifierSpec | None = None

    def init_sparsifier(self, initializer: Initializer, model_dim: int, vocab_size: int) -> Sparsifier | None:
        if self.sparsifier_spec is None:
            return None
        return self.sparsifier_spec.init(initializer, model_dim, vocab_size)

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
    sparsifier: Sparsifier | None

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

    def empty_logits(self, batch_size: int) -> Logits:
        if self.sparsifier is None:
            return DenseLogits(values=jnp.zeros((batch_size, self.vocab_size), dtype=jnp.float32))
        return SparseLogits(
            values=jnp.zeros((batch_size, self.sparsifier.candidate_count), dtype=jnp.float32),
            token_ids=jnp.zeros((batch_size, self.sparsifier.candidate_count), dtype=jnp.int32),
        )

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
    ) -> Logits:
        if self.sparsifier is None:
            logits = DenseLogits(
                values=self._readout_logits(
                    x,
                    keychain=keychain,
                    forward_pass_config=forward_pass_config.matmul_config,
                ),
            )
        else:
            logits = self.sparsifier(
                x,
                self.readout_matrix,
                keychain=keychain,
                forward_pass_config=forward_pass_config.matmul_config,
            )
        logits = logits.astype(forward_pass_config.logit_dtype)
        if self.config.logit_soft_cap is not None:
            logits = logits.map_values(lambda values: apply_soft_capping(values, self.config.logit_soft_cap))
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
            sparsifier=self.init_sparsifier(initializer, model_dim, vocab_size),
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
        if self.sparsifier_spec is None:
            output_embedding = initializer.weight_matrix(vocab_size, model_dim)
        else:
            output_embedding = initializer.embedding_matrix(vocab_size, model_dim, dtype=jnp.bfloat16)
        return UntiedEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            sparsifier=self.init_sparsifier(initializer, model_dim, vocab_size),
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
