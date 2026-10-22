from dataclasses import dataclass, replace
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
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

# Every step's 16-bit state hashes to its levels; each tape block opens with its first state in 16 bits.
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
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f"The trellis table is not scale * level + offset (error {error})")
    return jnp.asarray([scale, *offsets[np.arange(COLUMN_CLASSES) % width]], dtype=jnp.float32)


@dataclass(frozen=True)
class QtipGaussianSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 7, 8]
    restart_columns: Literal[0, 64, 128]
    # 2: each block's bits reversed, least significant first.
    tape: Literal[2] = 2

    def __post_init__(self) -> None:
        match self.vector_width, self.transition_bits, self.restart_columns:
            case (2, 4 | 6 | 8, 0) | (4, 8, 0) | (4, 6 | 7 | 8, 64 | 128):
                return
            case layout:
                raise ValueError(f"Unsupported QTIP Gaussian layout {layout}")

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

    def block_bits(self, columns: int) -> int:
        return STATE_BITS + ((self.restart_columns or columns) // self.vector_width - 1) * self.transition_bits

    def states(self, codes: UInt8[Array, "*rows bytes"], columns: int) -> Array:
        # State g of an L-bit block is the little-endian 16-bit field at bit L - 16 - g * T.
        step = np.arange(columns // self.vector_width)
        steps_per_block = (self.restart_columns or columns) // self.vector_width
        block_bits = self.block_bits(columns)
        offset = (
            (step // steps_per_block + 1) * block_bits - STATE_BITS - step % steps_per_block * self.transition_bits
        )
        byte = offset // 8
        data = jnp.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, 2)]).astype(jnp.uint32)
        word = data[..., byte] | (data[..., byte + 1] << 8) | (data[..., byte + 2] << 16)
        shift = jnp.asarray(offset % 8, dtype=jnp.uint32)
        return (word >> shift) & 0xFFFF

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
