from dataclasses import dataclass, replace
from typing import Literal, Self

import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Key, UInt8

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
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
# Each code byte picks one row of four int8 values from the saved D4 table.
VECTOR_WIDTH = 4
CODE_BITS = 8
# The D4 code table, one hex digit per coordinate (offset by 3); the ladder is fp16(2^((k - 11) / 2)).
D4_TABLE_DIGITS = (
    "3333423351332433342343235223252335134413531322533243414314432343334242425142244234324332523225323522"
    "4422532262223252451254122352335142511111245134414341524125413531443153312235322545215421232533244224"
    "5124242434144314521425143144044413442244323441341434233433444244514424443434433452342534352444245324"
    "6224325445145414235433534253111324533443434352432543353344335333623336234523542363233362022211226422"
    "3452435212122112354244425342624236324532543263323335423511312435342543251221212131114451131122113245"
    "4541544123453311421151112411344543451241214131314435133122313221412114212321336402241124642434544354"
    "1214211435444444534462443634453454346334733302331133643330230323122321233113445313132213364345435443"
    "6343330202421142644230320332123221323122042213222222321241121412231233224222512224223412431252122512"
    "3142044213422242323241321432233233314231513124313421432152212521351144115311225132414141144123413304"
    "0244114464443034033412342134312404241324222432144114141423143313421351132413304303431243214331330433"
    "133322333223412314232323"
)
D4_TABLE = (np.array([int(digit, 16) for digit in D4_TABLE_DIGITS], np.int8) - 3).reshape(1 << CODE_BITS, VECTOR_WIDTH)
D4_LADDER = np.exp2((np.arange(1 << LADDER_INDEX_BITS, dtype=np.float32) - 11) / 2).astype(np.float16)


@dataclass(frozen=True)
class LatticeSpec(WeightMatrixSpec):
    layout: Layout
    kind: Literal["d4"] = "d4"

    def code_bytes(self, columns: int) -> int:
        if columns <= 0 or columns % COLUMNS_PER_LADDER_BYTE:
            raise ValueError(f"Lattice matrices require a positive multiple of {COLUMNS_PER_LADDER_BYTE} columns")
        return columns // VECTOR_WIDTH

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
    ) -> "LatticeMatrix":
        if not isinstance(weights, Tracer):
            raise NotImplementedError("Lattice matrices are loaded from saved parameters, not fitted")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        row_axis, _ = self.layout.weight_partition(0, is_sharded=is_sharded)
        initializer = RandomInitializer(weights.dtype, sharding_config)
        return LatticeMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, self.code_bytes(columns)), (row_axis, None), jnp.uint8),
            row_scales=initializer.zeros((rows,), (row_axis,)),
            ladder_indices=initializer.zeros((rows, columns // COLUMNS_PER_LADDER_BYTE), (row_axis, None), jnp.uint8),
        )


class LatticeMatrix(EmbeddingMatrix[LatticeSpec]):
    codes: UInt8[Array, "rows bytes"]
    row_scales: Float[Array, " rows"]
    ladder_indices: UInt8[Array, "rows groups"]

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.ladder_indices.shape == (rows, columns // COLUMNS_PER_LADDER_BYTE)
        assert self.row_scales.shape == (rows,)
        assert self.codes.dtype == self.ladder_indices.dtype == jnp.uint8

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.ladder_indices.shape[1] * COLUMNS_PER_LADDER_BYTE

    @property
    def dtype(self) -> DTypeLike:
        return self.row_scales.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, row_scales=self.row_scales.astype(dtype))

    def decode_rows(self, codes: Array, row_scales: Array, ladder_indices: Array) -> Array:
        columns = self.shape[1]
        row_axes = tuple(sharding_of(codes).spec)[: codes.ndim - 1]
        values = jnp.asarray(D4_TABLE).at[codes].get(out_sharding=PartitionSpec(*row_axes, None, None))
        values = values.reshape(*codes.shape[:-1], columns).astype(jnp.float32)
        groups = unpack_uint8_to_uint(
            ladder_indices, LADDER_INDEX_BITS, unpacked_last_axis_dim=columns // COLUMNS_PER_LADDER_INDEX
        )
        ladder = jnp.asarray(D4_LADDER).at[groups].get(out_sharding=PartitionSpec(*row_axes, None))
        scales = row_scales.astype(jnp.float32)[..., None] * ladder.astype(jnp.float32)
        grouped = values.reshape(*scales.shape, COLUMNS_PER_LADDER_INDEX)
        return (grouped * scales[..., None]).reshape(values.shape)

    def decompress(self) -> Array:
        decoded = self.decode_rows(self.codes, self.row_scales, self.ladder_indices)
        return self.spec.layout.to_output_input(decoded).astype(self.dtype)

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
        codes = lookup_sharded_indices(self.codes, row_index)
        row_scales = lookup_sharded_indices(self.row_scales, row_index)
        ladder_indices = lookup_sharded_indices(self.ladder_indices, row_index)
        if dtype is None:
            dtype = self.dtype
        return self.decode_rows(codes, row_scales, ladder_indices).astype(dtype)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Array:
        weights = self.decompress().astype(vector.dtype)
        layout = Layout.OUTPUT_INPUT
        if transposed:
            layout = Layout.INPUT_OUTPUT
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(weights, vector)
