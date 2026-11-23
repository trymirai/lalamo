from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Key, UInt8

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
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

from .utils.packing import unpack_uint8_to_uint


@dataclass(frozen=True)
class I3S4Spec(WeightMatrixSpec):
    layout: Layout = Layout.OUTPUT_INPUT

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
    ) -> "I3S4Matrix":
        if not isinstance(weights, Tracer):
            raise NotImplementedError("I3S4 matrices are loaded from saved parameters, not fitted")
        assert self.layout == Layout.OUTPUT_INPUT
        rows, columns = weights.shape
        assert columns % 128 == 0, f"I3S4 packs 128-column groups, got {columns} input columns"
        row_axis, _ = self.layout.weight_partition(0, is_sharded=is_sharded)
        initializer = RandomInitializer(weights.dtype, sharding_config)
        return I3S4Matrix(
            self,
            sharding_config,
            is_sharded,
            initializer.zeros((rows, columns * 3 // 8), (row_axis, None), jnp.uint8),
            initializer.zeros((rows,), (row_axis,)),
            initializer.zeros((rows, columns // 128), (row_axis, None), jnp.uint8),
            initializer.zeros((16,), (None,), jnp.float16),
        )


class I3S4Matrix(WeightMatrix[I3S4Spec]):
    codes: UInt8[Array, "rows bytes"]
    row_scales: Float[Array, " rows"]
    ladder_indices: UInt8[Array, "rows groups"]
    ladder: Float[Array, "16"]

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.spec.layout == Layout.OUTPUT_INPUT
        assert columns > 0 and columns % 128 == 0
        assert self.codes.shape == (rows, columns * 3 // 8)
        assert self.codes.dtype == self.ladder_indices.dtype == jnp.uint8
        assert self.row_scales.shape == (rows,) and jnp.issubdtype(self.row_scales.dtype, jnp.floating)
        assert self.ladder_indices.shape == (rows, columns // 128)
        assert self.ladder.shape == (16,) and self.ladder.dtype == jnp.float16

    @property
    def shape(self) -> tuple[int, int]:
        rows, code_bytes = self.codes.shape
        return rows, code_bytes * 8 // 3

    @property
    def dtype(self) -> DTypeLike:
        return self.row_scales.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, row_scales=self.row_scales.astype(dtype))

    def decode_rows(self, codes: Array, row_scales: Array, ladder_indices: Array) -> Array:
        levels = 2 * unpack_uint8_to_uint(codes, 3).astype(jnp.float32) - 7
        groups = unpack_uint8_to_uint(ladder_indices, 4)
        ladder = self.ladder.at[groups].get(out_sharding=sharding_of(groups))
        scales = row_scales.astype(jnp.float32)[..., None] * ladder.astype(jnp.float32)
        # The producer's readout kernel rounds decoded weights before the dot product.
        return (levels.reshape(*scales.shape, 64) * scales[..., None]).reshape(levels.shape).astype(self.dtype)

    def decompress(self) -> Array:
        return self.decode_rows(self.codes, self.row_scales, self.ladder_indices)

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
        assert not transposed, "I3S4 matrices are output projections"
        mesh = sharding_of(vector).mesh
        codes = with_sharding(self.codes, NamedSharding(mesh, PartitionSpec(None, None)))
        scales = with_sharding(self.row_scales, NamedSharding(mesh, PartitionSpec(None)))
        indices = with_sharding(self.ladder_indices, NamedSharding(mesh, PartitionSpec(None, None)))

        def row_dot(row: tuple[Array, Array, Array]) -> Array:
            return jax.lax.dot_general(
                self.decode_rows(*row).astype(vector.dtype),
                vector,
                dimension_numbers=(((0,), (0,)), ((), ())),
                precision=forward_pass_config.precision,
                out_sharding=NamedSharding(mesh, PartitionSpec()),
            )

        return jax.lax.map(row_dot, (codes, scales, indices), batch_size=128)
