from dataclasses import dataclass, replace
from math import ceil
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import DotAlgorithmPreset
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int8, Key, UInt8, UInt32

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_or_tracer
from lalamo.utils.sharding import ShardingConfig, sharding_of, with_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

# Every step's 16-bit state hashes to its levels; each tape block opens with its first state in two bytes.
STATE_BITS = 16
# The codebook is [scale, offset of each column class], the class of a column being its index modulo four.
COLUMN_CLASSES = 4


def states_to_levels(states: UInt32[Array, "..."]) -> Int8[Array, "... 4"]:
    # Low 32 bits of SplitMix64 at 0 (forced odd) and 1 with seed 1234, as uzu's trellis_format.rs derives them.
    hashes = states * jnp.uint32(3486300223) + jnp.uint32(1481329315)
    hashes = hashes ^ (hashes >> jnp.uint32(16))
    hashes = hashes * jnp.uint32(0x85EBCA6B)
    hashes = hashes ^ (hashes >> jnp.uint32(16))
    pairs = (hashes & jnp.uint32(0x33333333)) + ((hashes >> jnp.uint32(2)) & jnp.uint32(0x33333333))
    pairs = (pairs & jnp.uint32(0x0F0F0F0F)) + ((pairs >> jnp.uint32(4)) & jnp.uint32(0x0F0F0F0F))
    dither = (jnp.uint32(3) * (hashes & jnp.uint32(0x0F0F0F0F))) & jnp.uint32(0x0F0F0F0F)
    packed = ((pairs << jnp.uint32(3)) + dither + jnp.uint32(0x4A4A4A4A)) ^ jnp.uint32(0x80808080)
    return jax.lax.bitcast_convert_type(packed, jnp.int8)


def codebook_from_table(table: Float[Array, "states width"]) -> Float[Array, " codebook"]:
    # Packages save every state's values; they must be scale * level + the offset of the column class.
    width = table.shape[1]
    values = np.asarray(table, dtype=np.float64)
    levels = np.asarray(states_to_levels(jnp.arange(1 << STATE_BITS, dtype=jnp.uint32)), dtype=np.float64)[:, :width]
    farthest = np.argmax(np.abs(levels[:, 0] - levels[0, 0]))
    scale = (values[farthest, 0] - values[0, 0]) / (levels[farthest, 0] - levels[0, 0])
    offsets = values[0] - scale * levels[0]
    error = np.abs(scale * levels + offsets - values).max()
    if not error <= 1e-5:
        raise ValueError(f"The trellis table is not scale * level + offset (error {error})")
    return jnp.asarray([scale, *offsets[np.arange(COLUMN_CLASSES) % width]], dtype=jnp.float32)


def full_rotation(values: Array, small_q: Array) -> Array:
    # Multiplies by the saved kron(H, Q) without materializing it.
    order = small_q.shape[0]
    width = values.shape[-1] // order
    assert small_q.shape == (order, order)
    assert width > 0 and width & (width - 1) == 0
    result = values.reshape(*values.shape[:-1], width, order)
    stride = 1
    while stride < width:
        grouped = result.reshape(*values.shape[:-1], -1, 2, stride, order)
        left, right = grouped[..., 0, :, :], grouped[..., 1, :, :]
        result = jnp.concatenate((left + right, left - right), axis=-2).reshape(result.shape)
        stride *= 2
    result = result / jnp.sqrt(jnp.asarray(width, dtype=values.dtype))
    if order == 1:
        return (result * small_q[0, 0]).reshape(values.shape)
    return jnp.matmul(result, small_q, precision=DotAlgorithmPreset.F32_F32_F32).reshape(values.shape)


@dataclass(frozen=True)
class QtipGaussianSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 7, 8]
    restart_columns: Literal[0, 64, 128]

    def __post_init__(self) -> None:
        layout = (self.vector_width, self.transition_bits, self.restart_columns)
        restarted = {(4, bits, columns) for bits in (6, 7, 8) for columns in (64, 128)}
        if layout not in {(2, 4, 0), (2, 6, 0), (2, 8, 0), (4, 8, 0), *restarted}:
            raise ValueError(f"Unsupported QTIP Gaussian layout {layout}")

    def tape_shape(self, columns: int) -> tuple[int, int, int]:
        block_columns = self.restart_columns or columns
        if columns <= 0 or columns % block_columns or block_columns % self.vector_width:
            raise ValueError(f"Invalid column count {columns} for {self}")
        steps = block_columns // self.vector_width
        return columns // block_columns, steps, STATE_BITS // 8 + ceil((steps - 1) * self.transition_bits / 8)

    def code_bytes(self, columns: int) -> int:
        blocks, _, block_bytes = self.tape_shape(columns)
        return blocks * block_bytes

    def states(self, codes: UInt8[Array, "*rows bytes"], columns: int) -> Array:
        # Each block is an MSB-first bit stream; state g is its 16-bit window at bit g * transition_bits.
        blocks, steps, block_bytes = self.tape_shape(columns)
        *rows, _ = codes.shape
        tapes = codes.reshape(*rows, blocks, block_bytes)
        tapes = jnp.pad(tapes, [(0, 0)] * (tapes.ndim - 1) + [(0, 2)]).astype(jnp.uint32)
        bit_offsets = np.arange(steps) * self.transition_bits
        windows = sum(tapes[..., bit_offsets // 8 + index] << (16 - 8 * index) for index in range(3))
        shifts = jnp.asarray(8 - bit_offsets % 8, dtype=jnp.uint32)
        return ((windows >> shifts) & jnp.uint32((1 << STATE_BITS) - 1)).reshape(*rows, blocks * steps)

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
        if not is_dummy_or_tracer(weights):
            raise ValueError("QTIP Gaussian matrices must be loaded from saved parameters; fitting is not supported")
        rows, columns = weights.shape
        row_axis, _ = Layout.OUTPUT_INPUT.weight_partition(0, is_sharded=is_sharded)
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return QtipGaussianMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            dtype_=weights.dtype,
            columns=columns,
            codes=initializer.zeros((rows, self.code_bytes(columns)), (row_axis, None), jnp.uint8),
            scales=initializer.zeros((rows,), (row_axis,), jnp.float32),
            codebook=initializer.zeros((1 + COLUMN_CLASSES,), dtype=jnp.float32),
        )


class QtipGaussianMatrix(WeightMatrix[QtipGaussianSpec]):
    dtype_: DTypeLike = field(static=True)
    columns: int = field(static=True)
    codes: UInt8[Array, "rows bytes"]
    scales: Float[Array, " rows"]
    codebook: Float[Array, " codebook"] = field(trainable=False)

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.codes.dtype == jnp.uint8
        assert self.scales.shape == (rows,)
        assert self.codebook.shape == (1 + COLUMN_CLASSES,)
        assert self.scales.dtype == self.codebook.dtype == jnp.float32

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
        offsets = self.codebook[1 + jnp.arange(self.columns) % COLUMN_CLASSES]
        return ((self.codebook[0] * levels + offsets) * scales[..., None]).astype(self.dtype)

    def decompress(self) -> Array:
        return self.decode_rows(self.codes, self.scales)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec().compress(
            self.decompress(), sharding_config=self.sharding_config, is_sharded=self.is_sharded
        )

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        assert not transposed, "QTIP Gaussian matrices are projections, never tied embeddings"
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
