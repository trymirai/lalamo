from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Int32, Key

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec, IncoherenceProcessingMode
from lalamo.compressed.int import IntSpec
from lalamo.module import Keychain, ParameterNorm, field
from lalamo.preconditioner import Preconditioner
from lalamo.sampling import SparseLogits
from lalamo.utils.dummy_array import dummy_array, is_dummy_array
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, supports_mosaic_gpu
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = ["LowRankPreviewReadoutMatrix", "LowRankPreviewReadoutSpec"]


@dataclass(frozen=True)
class LowRankPreviewReadoutSpec(WeightMatrixSpec):
    rank: int = 256
    selection_block_size: int = 1_024
    exact_tokens_per_block: int = 32

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,  # noqa: ARG002
    ) -> "LowRankPreviewReadoutMatrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise ValueError("LowRankPreviewReadout only supports inference.")
        if not is_dummy_array(weights):
            raise ValueError(
                "Initialization of the low rank arrays is not trivial, "
                "and cannot be done without additional unroll stats."
            )
        vocabulary_size, hidden_size = weights.shape
        if not 0 < self.rank <= hidden_size or self.rank % 64:
            raise ValueError("rank must fit hidden size and be divisible by 64.")
        if self.selection_block_size <= 0 or not 0 < self.exact_tokens_per_block <= self.selection_block_size:
            raise ValueError("exact_tokens_per_block must be positive and fit selection_block_size.")
        matrix_sharding = sharding_config.resolve_sharding((None, None))
        row_sharding = sharding_config.resolve_sharding((None,))
        return LowRankPreviewReadoutMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=False,
            hidden_projection=FullPrecisionSpec().compress(
                dummy_array((self.rank, hidden_size), jnp.bfloat16, matrix_sharding),
                sharding_config=sharding_config,
                is_sharded=False,
            ),
            preview=HybridSpec(
                quantization_spec=IntSpec(bits=4, group_size=64),
                adapter_spec=None,
                incoherence_block_size=32,
                incoherence_processing_mode=IncoherenceProcessingMode.INPUT,
            ).compress(
                dummy_array((vocabulary_size, self.rank), jnp.bfloat16, matrix_sharding),
                key=jax.random.key(0),
                sharding_config=sharding_config,
                is_sharded=False,
            ),
            weights=FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(
                dummy_array((hidden_size, vocabulary_size), jnp.bfloat16, matrix_sharding),
                sharding_config=sharding_config,
                is_sharded=False,
            ),
            preview_token_ids=dummy_array((vocabulary_size,), jnp.int32, row_sharding),
        )


class LowRankPreviewReadoutMatrix(WeightMatrix[LowRankPreviewReadoutSpec]):
    hidden_projection: FullPrecisionMatrix = field(norm=ParameterNorm.SPECTRAL)
    preview: HybridMatrix = field(norm=ParameterNorm.SPECTRAL)
    weights: FullPrecisionMatrix = field(norm=ParameterNorm.SPECTRAL)
    preview_token_ids: Int32[Array, " vocabulary"] = field(trainable=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    @property
    def candidate_count(self) -> int:
        vocabulary_size, _ = self.shape
        full_blocks, tail_size = divmod(vocabulary_size, self.spec.selection_block_size)
        return full_blocks * self.spec.exact_tokens_per_block + min(
            tail_size,
            self.spec.exact_tokens_per_block,
        )

    def astype(self, dtype: DTypeLike) -> "LowRankPreviewReadoutMatrix":
        if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("LowRankPreviewReadout requires BF16 projection, preview scales, and readout rows.")
        return self

    def switch_implementation(
        self,
        implementation: CompressionImplementation,
    ) -> "LowRankPreviewReadoutMatrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise ValueError("LowRankPreviewReadout only supports inference.")
        return self

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> "LowRankPreviewReadoutMatrix":
        if sharding_config != self.sharding_config:
            raise ValueError("LowRankPreviewReadout does not support changing sharding configuration.")
        return self

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=False,
        )

    def decompress(self) -> Float[Array, "vocabulary channels"]:
        return self.weights.weights

    def candidate_logits(
        self,
        vector: Float[Array, " channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
    ) -> SparseLogits:
        self._raise_if_batched()
        if vector.dtype != jnp.bfloat16:
            raise ValueError("LowRankPreviewReadout requires a BF16 input vector.")

        projected_hidden = self.hidden_projection.dot(
            vector,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        preview_scores = self.preview.dot(
            projected_hidden,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        (vocabulary_size,) = preview_scores.shape
        selection_block_size = self.spec.selection_block_size
        full_blocks, tail_size = divmod(vocabulary_size, selection_block_size)
        full_vocabulary_size = full_blocks * selection_block_size
        full_block_scores = preview_scores[:full_vocabulary_size].reshape(full_blocks, selection_block_size)
        supports_mosaic = supports_mosaic_gpu(self.sharding_config.mesh, 9)
        if (
            full_blocks > 0
            and 128 <= selection_block_size <= 4_096
            and selection_block_size & (selection_block_size - 1) == 0
            and full_block_scores.dtype == jnp.bfloat16
            and supports_mosaic
        ):
            from lalamo.kernels.low_rank_readout import make_block_topk  # noqa: PLC0415

            local_rows = make_block_topk(
                full_blocks,
                selection_block_size,
                self.spec.exact_tokens_per_block,
            )(full_block_scores)
        else:
            _, local_rows = jax.lax.top_k(full_block_scores, self.spec.exact_tokens_per_block)
        physical_rows = (
            local_rows + selection_block_size * jnp.arange(full_blocks, dtype=jnp.int32)[:, None]
        ).reshape(-1)
        if tail_size:
            _, tail_rows = jax.lax.top_k(
                preview_scores[full_vocabulary_size:],
                min(tail_size, self.spec.exact_tokens_per_block),
            )
            physical_rows = jnp.concatenate((physical_rows, full_vocabulary_size + tail_rows))
        token_ids = self.preview_token_ids[physical_rows]

        if (
            self.weights.weights.dtype == jnp.bfloat16
            and vector.shape[0] % 128 == 0
            and forward_pass_config.precision == DotAlgorithmPreset.DEFAULT
            and supports_mosaic
        ):
            from lalamo.kernels.low_rank_readout import make_gathered_dot  # noqa: PLC0415

            logits = make_gathered_dot(token_ids.shape[0], vector.shape[0])(
                self.weights.weights,
                token_ids,
                vector,
            )
        else:
            rescore_weights = self.weights.lookup_embedding(
                token_ids,
                dtype=jnp.bfloat16,
                keychain=keychain,
                forward_pass_config=forward_pass_config,
            )
            with use_dot_algorithm_preset(forward_pass_config.precision):
                logits = rescore_weights @ vector
        return SparseLogits(values=logits, token_ids=token_ids)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        if transposed:
            self._raise_if_batched()
            if vector.dtype != jnp.bfloat16:
                raise ValueError("LowRankPreviewReadout requires a BF16 input vector.")
            return self.weights.dot(
                vector,
                keychain=keychain,
                forward_pass_config=forward_pass_config,
            )

        candidates = self.candidate_logits(
            vector,
            keychain=keychain,
            forward_pass_config=forward_pass_config,
        )
        logits = jnp.full(
            (self.shape[0],),
            -jnp.inf,
            dtype=jnp.bfloat16,
        )
        return logits.at[candidates.token_ids].set(candidates.values)
