from dataclasses import dataclass, replace
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.exportable import ExportResults
from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dtype import dtype_to_str, str_to_dtype
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, sharding_of, with_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

# Each tape block opens with its first 16-bit codebook state.
STATE_BITS = 16
COLUMN_CLASSES = 4


@dataclass(frozen=True)
class TrellisCodebook:
    scale: float
    offsets: tuple[float, float, float, float]


def states_to_levels(states: Array) -> Array:
    hashes = states * jnp.uint32(3486300223) + jnp.uint32(1481329315)
    hashes ^= hashes >> 16
    hashes *= jnp.uint32(0x85EBCA6B)
    hashes ^= hashes >> 16
    pairs = (hashes & 0x33333333) + ((hashes >> 2) & 0x33333333)
    pairs = (pairs & 0x0F0F0F0F) + ((pairs >> 4) & 0x0F0F0F0F)
    dither = (3 * (hashes & 0x0F0F0F0F)) & 0x0F0F0F0F
    packed = ((pairs << 3) + dither + 0x4A4A4A4A) ^ jnp.uint32(0x80808080)
    return jax.lax.bitcast_convert_type(packed, jnp.int8)


@dataclass(frozen=True)
class QtipGaussianSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 7, 8]
    restart_columns: Literal[0, 64, 128]
    codebook: TrellisCodebook
    layout: Layout = Layout.OUTPUT_INPUT
    # 2: each block's bits reversed, least significant first.
    tape: Literal[2] = 2

    def __post_init__(self) -> None:
        match self.vector_width, self.transition_bits, self.restart_columns:
            case (2, 4 | 6 | 8, 0) | (4, 8, 0) | (4, 6 | 7 | 8, 64 | 128):
                return
            case unsupported:
                raise ValueError(f"Unsupported QTIP Gaussian layout {unsupported}")

    def code_bytes(self, columns: int) -> int:
        block_columns = self.restart_columns or columns
        assert columns > 0 and columns % block_columns == 0
        assert block_columns % self.vector_width == 0
        steps = columns // self.vector_width
        blocks = columns // block_columns
        # Each block starts with a full state instead of a transition.
        bits = steps * self.transition_bits + blocks * (STATE_BITS - self.transition_bits)
        if self.restart_columns:
            return (bits + 127) // 128 * 16
        return (bits + 7) // 8

    def states(self, codes: UInt8[Array, "*rows bytes"], columns: int) -> Array:
        # State g of an L-bit block is the little-endian 16-bit field at bit L - 16 - g * T.
        step = np.arange(columns // self.vector_width)
        steps_per_block = (self.restart_columns or columns) // self.vector_width
        block_bits = STATE_BITS + (steps_per_block - 1) * self.transition_bits
        offset = (
            (step // steps_per_block + 1) * block_bits - STATE_BITS - step % steps_per_block * self.transition_bits
        )
        byte = offset // 8
        data = jnp.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, 2)]).astype(jnp.uint32)
        word = data[..., byte] | (data[..., byte + 1] << 8) | (data[..., byte + 2] << 16)
        shift = jnp.asarray(offset % 8, dtype=jnp.uint32)
        return (word >> shift) & 0xFFFF

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "QtipGaussianMatrix":
        if not isinstance(weights, Tracer):
            raise NotImplementedError("QTIP Gaussian matrices are loaded from saved parameters, not fitted")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        row_axis, _ = self.layout.weight_partition(0, is_sharded=is_sharded)
        initializer = RandomInitializer(weights.dtype, sharding_config)
        return QtipGaussianMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            dtype_=weights.dtype,
            columns=columns,
            codes=initializer.zeros((rows, self.code_bytes(columns)), (row_axis, None), jnp.uint8),
            scales=initializer.zeros((rows,), (row_axis,), jnp.float32),
        )


class QtipGaussianMatrix(EmbeddingMatrix[QtipGaussianSpec]):
    dtype_: DTypeLike = field(static=True)
    columns: int = field(static=True)
    codes: UInt8[Array, "rows bytes"]
    scales: Float[Array, " rows"]

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.codes.dtype == jnp.uint8
        assert self.scales.shape == (rows,)
        assert jnp.issubdtype(self.scales.dtype, jnp.floating)

    def export(self) -> ExportResults:
        arrays, metadata = super().export()
        return ExportResults(arrays, {**metadata, "dtype": dtype_to_str(self.dtype)})

    def load_exported(self, exported_data: ExportResults, *, prefix: ParameterPath | None = None) -> Self:
        if prefix is None:
            prefix = ParameterPath()
        saved_dtype = exported_data.metadata[prefix / "dtype"]
        assert isinstance(saved_dtype, str)
        dtype = str_to_dtype(saved_dtype)
        assert jnp.issubdtype(dtype, jnp.floating)
        if self.scales.weak_type:
            template = replace(self, dtype_=dtype)
        else:
            template = self
        return cast("Self", WeightMatrix.load_exported(template, exported_data, prefix=prefix))

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.columns

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, dtype_=jnp.dtype(dtype))

    def decode_rows(self, codes: UInt8[Array, "*rows bytes"], scales: Float[Array, "*rows"]) -> Array:
        levels = states_to_levels(self.spec.states(codes, self.columns))[..., : self.spec.vector_width]
        levels = levels.reshape(*codes.shape[:-1], self.columns).astype(jnp.float32)
        codebook = self.spec.codebook
        offsets = jnp.asarray(codebook.offsets, dtype=jnp.float32)
        values = jnp.float32(codebook.scale) * levels + offsets[jnp.arange(self.columns) % COLUMN_CLASSES]
        return (values * scales[..., None].astype(jnp.float32)).astype(self.dtype)

    def decompress(self) -> Array:
        return self.spec.layout.to_output_input(self.decode_rows(self.codes, self.scales))

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(self.spec.layout).compress(
            self.decompress(), sharding_config=self.sharding_config, is_sharded=self.is_sharded
        )

    def lookup_embedding(
        self,
        row_index: int | Int[Array, "*batch"],
        *,
        keychain: Keychain,  # noqa: ARG002
        dtype: DTypeLike | None = None,
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Array:
        if self.spec.layout != Layout.INPUT_OUTPUT:
            raise ValueError("Embedding lookup requires input-output layout")
        rows = self.decode_rows(
            lookup_sharded_indices(self.codes, row_index), lookup_sharded_indices(self.scales, row_index)
        )
        return rows.astype(self.dtype if dtype is None else dtype)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        # A row-by-row dot: a projection's outputs, or the readout (the transposed product) of an embedding table.
        if transposed != (self.spec.layout == Layout.INPUT_OUTPUT):
            raise ValueError("A trellis matrix multiplies row by row, so its rows are the output channels")
        mesh = sharding_of(vector).mesh
        # FSDP shares the batch and matrix axis. Gather packed rows, so token batching owns that axis;
        # only 128 rows are ever decoded at once.
        codes = with_sharding(self.codes, NamedSharding(mesh, PartitionSpec(None, None)))
        scales = with_sharding(self.scales, NamedSharding(mesh, PartitionSpec(None)))

        def row_dot(row: tuple[Array, Array]) -> Array:
            return jax.lax.dot_general(
                self.decode_rows(*row).astype(vector.dtype),
                vector,
                dimension_numbers=(((0,), (0,)), ((), ())),
                precision=forward_pass_config.precision,
                out_sharding=NamedSharding(mesh, PartitionSpec()),
            )

        return jax.lax.map(row_dot, (codes, scales), batch_size=128)
