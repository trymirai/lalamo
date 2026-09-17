from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int, Int8, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.kernels.hadamard import hadamard_transform
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_array
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, lookup_sharded_indices, sharding_of, with_sharding
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
from .utils.row_dot import row_batched_dot
from .utils.s_gains import SScaleAxis, apply_post_gains


class SSurfaceKind(StrEnum):
    D4 = "d4"
    I3 = "i3"
    I4 = "i4"


@dataclass(frozen=True)
class SSurfaceSpec(WeightMatrixSpec):
    kind: SSurfaceKind
    layout: Layout
    post_gain_axes: tuple[SScaleAxis, ...] = ()

    @property
    def vector_width(self) -> int:
        match self.kind:
            case SSurfaceKind.D4:
                return 4
            case SSurfaceKind.I3 | SSurfaceKind.I4:
                return 1
        raise ValueError(f"Unknown S surface kind: {self.kind}")

    @property
    def code_bits(self) -> int:
        match self.kind:
            case SSurfaceKind.D4:
                return 8
            case SSurfaceKind.I3:
                return 3
            case SSurfaceKind.I4:
                return 4
        raise ValueError(f"Unknown S surface kind: {self.kind}")

    def code_bytes(self, columns: int) -> int:
        if columns <= 0 or columns % 128:
            raise ValueError("S surfaces require a positive multiple of 128 columns")
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
    ) -> "SSurfaceMatrix":
        if not is_dummy_array(weights):
            raise ValueError("S surfaces must be loaded from saved parameters; fitting is not supported")
        rows, columns = self.layout.weight_shape((), *weights.shape)
        states = 1 << self.code_bits
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return SSurfaceMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, self.code_bytes(columns)), dtype=jnp.uint8),
            row_scales=initializer.zeros((rows,)),
            ladder_indices=initializer.zeros((rows, columns // 128), dtype=jnp.uint8),
            ladder=initializer.zeros((16,), dtype=jnp.float16),
            table=initializer.zeros((states, self.vector_width), dtype=jnp.int8),
            signs=initializer.zeros((columns,), dtype=jnp.int32),
            post_gains=tuple(
                initializer.zeros((rows if axis == SScaleAxis.ROW else columns,), dtype=jnp.float32)
                for axis in self.post_gain_axes
            ),
        ).switch_sharding_config(sharding_config)


class SSurfaceMatrix(EmbeddingMatrix[SSurfaceSpec]):
    codes: UInt8[Array, "rows bytes"]
    row_scales: Float[Array, " rows"]
    ladder_indices: UInt8[Array, "rows groups"]
    ladder: Float[Array, "16"] = field(trainable=False)
    table: Int8[Array, "states width"] = field(trainable=False)
    signs: Int[Array, " columns"] = field(trainable=False)
    post_gains: tuple[Array, ...] = ()

    def __check_init__(self) -> None:
        rows, columns = self.shape
        assert self.codes.shape == (rows, self.spec.code_bytes(columns))
        assert self.ladder_indices.shape == (rows, columns // 128)
        assert self.row_scales.shape == (rows,)
        assert self.codes.dtype == self.ladder_indices.dtype == jnp.uint8
        assert self.ladder.shape == (16,) and self.ladder.dtype == jnp.float16
        states = 1 << self.spec.code_bits
        assert self.table.shape == (states, self.spec.vector_width) and self.table.dtype == jnp.int8
        assert self.signs.dtype == jnp.int32
        for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True):
            assert isinstance(axis, SScaleAxis)
            assert gain.shape == (rows if axis == SScaleAxis.ROW else columns,)
            assert gain.dtype == jnp.float32

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.signs.shape[0]

    @property
    def dtype(self) -> DTypeLike:
        return self.row_scales.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, row_scales=self.row_scales.astype(dtype))

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> Self:
        # Vocabulary rows are independently decodable; all transforms stay local.
        row_axis, _ = self.spec.layout.weight_partition(0, is_sharded=self.is_sharded)
        return replace(
            self,
            sharding_config=sharding_config,
            codes=with_sharding(self.codes, sharding_config.resolve_sharding((row_axis, None))),
            row_scales=with_sharding(self.row_scales, sharding_config.resolve_sharding((row_axis,))),
            ladder_indices=with_sharding(self.ladder_indices, sharding_config.resolve_sharding((row_axis, None))),
            ladder=with_sharding(self.ladder, sharding_config.resolve_sharding((None,))),
            table=with_sharding(self.table, sharding_config.resolve_sharding((None, None))),
            signs=with_sharding(self.signs, sharding_config.resolve_sharding((None,))),
            post_gains=tuple(
                with_sharding(gain, sharding_config.resolve_sharding((row_axis if axis == SScaleAxis.ROW else None,)))
                for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            ),
        )

    def _decode(
        self, codes: Array, row_scales: Array, ladder_indices: Array, post_gains: tuple[Array, ...], dtype: DTypeLike
    ) -> Array:
        columns = self.signs.shape[0]
        if self.spec.kind == SSurfaceKind.I4:
            # The original INT4 packer writes the even column in the high nibble.
            indices = jnp.stack((codes >> 4, codes & 15), axis=-1).reshape(*codes.shape[:-1], columns)
        else:
            indices = unpack_uint8_to_uint(
                codes, self.spec.code_bits, unpacked_last_axis_dim=columns // self.spec.vector_width
            )
        row_axes = tuple(sharding_of(codes).spec)[:-1]
        values = self.table.at[indices].get(out_sharding=PartitionSpec(*row_axes, None, None))
        values = values.reshape(*codes.shape[:-1], columns).astype(jnp.float32)
        groups = unpack_uint8_to_uint(ladder_indices, 4, unpacked_last_axis_dim=columns // 64)
        ladder = self.ladder.at[groups].get(out_sharding=PartitionSpec(*row_axes, None))
        scales = row_scales.astype(jnp.float32)[..., None] * ladder.astype(jnp.float32)
        rotated = values * jnp.repeat(scales, 64, axis=-1)
        weights = hadamard_transform(rotated, 32) * self.signs.astype(jnp.float32)
        return apply_post_gains(weights, self.spec.post_gain_axes, post_gains, dtype)

    def decompress(self) -> Array:
        stored = self._decode(self.codes, self.row_scales, self.ladder_indices, self.post_gains, self.dtype)
        return self.spec.layout.to_output_input(stored)

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
        return self._decode(
            lookup_sharded_indices(self.codes, row_index),
            lookup_sharded_indices(self.row_scales, row_index),
            lookup_sharded_indices(self.ladder_indices, row_index),
            tuple(
                lookup_sharded_indices(gain, row_index) if axis == SScaleAxis.ROW else gain
                for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            ),
            self.dtype if dtype is None else dtype,
        )

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

        def decode_row(row: tuple[Array, ...]) -> Array:
            codes, scale, indices, *factors = row
            row_gains = iter(factors)
            post_gains = tuple(
                next(row_gains) if axis == SScaleAxis.ROW else gain
                for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            )
            return self._decode(codes, scale, indices, post_gains, self.dtype)

        row_gains = tuple(
            gain
            for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            if axis == SScaleAxis.ROW
        )
        return row_batched_dot(
            decode_row,
            (self.codes, self.row_scales, self.ladder_indices, *row_gains),
            vector,
            forward_pass_config.precision,
        )
