from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Int8, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_array
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, sharding_of
from lalamo.weight_matrix import (
    CompressionImplementation,
    EmbeddingMatrix,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrixSpec,
)

from .utils.packing import unpack_uint8_to_uint
from .utils.post_gains import GainAxis, apply_post_gains, merge_row_gains, row_gains
from .utils.row_dot import row_batched_dot

# Each 64-column group scales its row by one of 16 ladder values; two 4-bit ladder indices share a byte.
LADDER_INDEX_BITS = 4
COLUMNS_PER_LADDER_INDEX = 64
COLUMNS_PER_LADDER_BYTE = 128
HADAMARD_BLOCK_SIZE = 32


def odd_integer_table(bits: int) -> Int8[Array, "states 1"]:
    states = 1 << bits
    return jnp.arange(1 - states, states, 2, dtype=jnp.int8)[:, None]


class LatticeKind(StrEnum):
    D4 = "d4"
    I3 = "i3"
    I4 = "i4"


@dataclass(frozen=True)
class LatticeSpec(WeightMatrixSpec):
    kind: LatticeKind
    layout: Layout
    post_gain_axes: tuple[GainAxis, ...] = ()

    @property
    def vector_width(self) -> int:
        match self.kind:
            case LatticeKind.D4:
                return 4
            case LatticeKind.I3 | LatticeKind.I4:
                return 1
        raise ValueError(f"Unknown lattice kind: {self.kind}")

    @property
    def code_bits(self) -> int:
        match self.kind:
            case LatticeKind.D4:
                return 8
            case LatticeKind.I3:
                return 3
            case LatticeKind.I4:
                return 4
        raise ValueError(f"Unknown lattice kind: {self.kind}")

    def code_bytes(self, columns: int) -> int:
        if columns <= 0 or columns % COLUMNS_PER_LADDER_BYTE:
            raise ValueError(f"Lattice matrices require a positive multiple of {COLUMNS_PER_LADDER_BYTE} columns")
        return columns * self.code_bits // self.vector_width // 8

    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "LatticeMatrix":
        if not is_dummy_array(weights):
            raise ValueError("Lattice matrices must be loaded from saved parameters; fitting is not supported")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        states = 1 << self.code_bits
        # Vocabulary rows are independently decodable; all transforms stay local.
        row_axis, _ = self.layout.weight_partition(0, is_sharded=is_sharded)
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return LatticeMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, self.code_bytes(columns)), (row_axis, None), jnp.uint8),
            row_scales=initializer.zeros((rows,), (row_axis,)),
            ladder_indices=initializer.zeros((rows, columns // COLUMNS_PER_LADDER_BYTE), (row_axis, None), jnp.uint8),
            ladder=initializer.zeros((1 << LADDER_INDEX_BITS,), dtype=jnp.float16),
            table=initializer.zeros((states, self.vector_width), dtype=jnp.int8),
            signs=initializer.zeros((columns,), dtype=jnp.int32),
            post_gains=tuple(
                initializer.zeros((rows,), (row_axis,), jnp.float32)
                if axis == GainAxis.ROW
                else initializer.zeros((columns,), dtype=jnp.float32)
                for axis in self.post_gain_axes
            ),
        )


class LatticeMatrix(EmbeddingMatrix[LatticeSpec]):
    codes: UInt8[Array, "rows bytes"]
    row_scales: Float[Array, " rows"]
    ladder_indices: UInt8[Array, "rows groups"]
    ladder: Float[Array, " ladder"] = field(trainable=False)
    table: Int8[Array, "states width"] = field(trainable=False)
    signs: Int[Array, " columns"] = field(trainable=False)
    post_gains: tuple[Array, ...] = ()

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.ladder_indices.shape == (rows, columns // COLUMNS_PER_LADDER_BYTE)
        assert self.row_scales.shape == (rows,)
        assert self.codes.dtype == self.ladder_indices.dtype == jnp.uint8
        assert self.ladder.shape == (1 << LADDER_INDEX_BITS,) and self.ladder.dtype == jnp.float16
        states = 1 << self.spec.code_bits
        assert self.table.shape == (states, self.spec.vector_width) and self.table.dtype == jnp.int8
        assert self.signs.dtype == jnp.int32
        for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True):
            assert gain.shape == (rows if axis == GainAxis.ROW else columns,)
            assert gain.dtype == jnp.float32

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.signs.shape[0]

    @property
    def dtype(self) -> DTypeLike:
        return self.row_scales.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, row_scales=self.row_scales.astype(dtype))

    def row_arrays(self) -> tuple[Array, Array, Array, tuple[Array, ...]]:
        return self.codes, self.row_scales, self.ladder_indices, row_gains(self.spec.post_gain_axes, self.post_gains)

    def decode_rows(self, rows: tuple[Array, Array, Array, tuple[Array, ...]], dtype: DTypeLike) -> Array:
        codes, row_scales, ladder_indices, selected_row_gains = rows
        columns = self.signs.shape[0]
        if self.spec.kind == LatticeKind.I4:
            # The original INT4 packer writes the even column in the high nibble.
            indices = jnp.stack((codes >> 4, codes & 15), axis=-1).reshape(*codes.shape[:-1], columns)
        else:
            indices = unpack_uint8_to_uint(
                codes, self.spec.code_bits, unpacked_last_axis_dim=columns // self.spec.vector_width
            )
        row_axes = tuple(sharding_of(codes).spec)[: codes.ndim - 1]
        values = self.table.at[indices].get(out_sharding=PartitionSpec(*row_axes, None, None))
        values = values.reshape(*codes.shape[:-1], columns).astype(jnp.float32)
        groups = unpack_uint8_to_uint(
            ladder_indices, LADDER_INDEX_BITS, unpacked_last_axis_dim=columns // COLUMNS_PER_LADDER_INDEX
        )
        ladder = self.ladder.at[groups].get(out_sharding=PartitionSpec(*row_axes, None))
        scales = row_scales.astype(jnp.float32)[..., None] * ladder.astype(jnp.float32)
        rotated = values * jnp.repeat(scales, COLUMNS_PER_LADDER_INDEX, axis=-1)
        weights = hadamard_transform(rotated, HADAMARD_BLOCK_SIZE) * self.signs.astype(jnp.float32)
        post_gains = merge_row_gains(self.spec.post_gain_axes, self.post_gains, selected_row_gains)
        return apply_post_gains(weights, self.spec.post_gain_axes, post_gains, dtype)

    def decompress(self) -> Array:
        return self.spec.layout.to_output_input(self.decode_rows(self.row_arrays(), self.dtype))

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
        rows = jax.tree.map(lambda array: lookup_sharded_indices(array, row_index), self.row_arrays())
        return self.decode_rows(rows, self.dtype if dtype is None else dtype)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        if transposed or self.spec.layout != Layout.OUTPUT_INPUT:
            weights = self.decompress().astype(vector.dtype)
            layout = Layout.INPUT_OUTPUT if transposed else Layout.OUTPUT_INPUT
            with use_dot_algorithm_preset(forward_pass_config.precision):
                return layout.matmul(weights, vector)

        return row_batched_dot(
            lambda rows: self.decode_rows(rows, self.dtype), self.row_arrays(), vector, forward_pass_config.precision
        )
