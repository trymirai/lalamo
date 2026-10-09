from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Int8, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_or_tracer
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

# Each 64-column group scales its row by one of 16 ladder values; two 4-bit ladder indices share a byte.
LADDER_INDEX_BITS = 4
COLUMNS_PER_LADDER_INDEX = 64
COLUMNS_PER_LADDER_BYTE = 128


class LatticeKind(StrEnum):
    D4 = "d4"
    I3 = "i3"


@dataclass(frozen=True)
class LatticeSpec(WeightMatrixSpec):
    kind: LatticeKind
    layout: Layout

    @property
    def vector_width(self) -> int:
        match self.kind:
            case LatticeKind.D4:
                return 4
            case LatticeKind.I3:
                return 1
        raise ValueError(f"Unknown lattice kind: {self.kind}")

    @property
    def code_bits(self) -> int:
        match self.kind:
            case LatticeKind.D4:
                return 8
            case LatticeKind.I3:
                return 3
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
        if not is_dummy_or_tracer(weights):
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
        )


class LatticeMatrix(EmbeddingMatrix[LatticeSpec]):
    codes: UInt8[Array, "rows bytes"]
    row_scales: Float[Array, " rows"]
    ladder_indices: UInt8[Array, "rows groups"]
    ladder: Float[Array, " ladder"] = field(trainable=False)
    table: Int8[Array, "states width"] = field(trainable=False)

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.ladder_indices.shape == (rows, columns // COLUMNS_PER_LADDER_BYTE)
        assert self.row_scales.shape == (rows,)
        assert self.codes.dtype == self.ladder_indices.dtype == jnp.uint8
        assert self.ladder.shape == (1 << LADDER_INDEX_BITS,) and self.ladder.dtype == jnp.float16
        states = 1 << self.spec.code_bits
        assert self.table.shape == (states, self.spec.vector_width) and self.table.dtype == jnp.int8

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.ladder_indices.shape[1] * COLUMNS_PER_LADDER_BYTE

    @property
    def dtype(self) -> DTypeLike:
        return self.row_scales.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, row_scales=self.row_scales.astype(dtype))

    def row_arrays(self) -> tuple[Array, Array, Array]:
        return self.codes, self.row_scales, self.ladder_indices

    def decode_rows(self, rows: tuple[Array, Array, Array], dtype: DTypeLike) -> Array:
        codes, row_scales, ladder_indices = rows
        columns = self.shape[1]
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
        return (values * jnp.repeat(scales, COLUMNS_PER_LADDER_INDEX, axis=-1)).astype(dtype)

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
        weights = self.decompress().astype(vector.dtype)
        layout = Layout.INPUT_OUTPUT if transposed else Layout.OUTPUT_INPUT
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)
