from dataclasses import dataclass, replace
from typing import Literal, Self

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Key, UInt8

from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_evaluation, supports_dummy_arrays
from lalamo.utils.precision import use_dot_algorithm_preset
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


def full_rotation(values: Array, small_q: Array) -> Array:
    """Multiply by the saved kron(H, Q), without materializing the full matrix."""
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
    return jnp.matmul(
        result / jnp.sqrt(jnp.float32(width)), small_q, precision=DotAlgorithmPreset.F32_F32_F32
    ).reshape(values.shape)


@dataclass(frozen=True)
class STrellisSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 8]
    restart_columns: Literal[0, 64]
    layout: Layout = Layout.OUTPUT_INPUT

    def __post_init__(self) -> None:
        if self.layout != Layout.OUTPUT_INPUT:
            raise ValueError("S trellis matrices require output-input layout")
        if (self.vector_width, self.transition_bits, self.restart_columns) not in (
            (2, 4, 0),
            (2, 6, 0),
            (4, 8, 64),
        ):
            raise ValueError("Unsupported S trellis layout")

    def tape_shape(self, columns: int) -> tuple[int, int, int]:
        block_columns = self.restart_columns or columns
        if columns <= 0 or columns % block_columns or block_columns % self.vector_width:
            raise ValueError(f"Invalid column count {columns} for {self}")
        steps = block_columns // self.vector_width
        return columns // block_columns, steps, 2 + ((steps - 1) * self.transition_bits + 7) // 8

    def states(self, codes: UInt8[Array, "rows bytes"], columns: int) -> Array:
        blocks, steps, block_bytes = self.tape_shape(columns)
        tapes = codes.reshape(codes.shape[0], blocks, block_bytes)
        initial = tapes[..., 0].astype(jnp.uint32) | (tapes[..., 1].astype(jnp.uint32) << 8)
        symbols = unpack_uint8_to_uint(
            tapes[..., 2:], self.transition_bits, dtype=jnp.uint32, unpacked_last_axis_dim=steps - 1
        )
        # At most four preceding symbols contribute to a 16-bit state. This is
        # a parallel bit-window decode, independent of the row's sequence length.
        source = jnp.concatenate((initial[..., None], symbols), axis=-1)
        states = source
        for distance in range(1, min(steps, (16 + self.transition_bits - 1) // self.transition_bits)):
            previous = jnp.pad(source[..., :-distance], ((0, 0), (0, 0), (distance, 0)))
            states = states | (previous << (distance * self.transition_bits))
        return (states & jnp.uint32(65535)).reshape(codes.shape[0], blocks * steps)

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
    ) -> "STrellisMatrix":
        if not is_dummy_evaluation():
            raise ValueError("S checkpoints must be loaded from saved parameters; fitting is not supported")
        rows, columns = weights.shape
        blocks, _, block_bytes = self.tape_shape(columns)
        order = columns // (columns & -columns)
        return STrellisMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=jnp.zeros((rows, blocks * block_bytes), jnp.uint8),
            scales=jnp.zeros((rows,), jnp.float16),
            gains=jnp.zeros((rows,), weights.dtype),
            table=jnp.zeros((65536, self.vector_width), jnp.float32),
            signs=jnp.zeros((columns,), jnp.float32),
            small_q=jnp.zeros((order, order), jnp.float32),
        ).switch_sharding_config(sharding_config)


class STrellisMatrix(WeightMatrix[STrellisSpec]):
    codes: UInt8[Array, "rows bytes"]
    scales: Float[Array, " rows"]
    gains: Float[Array, " rows"]
    table: Float[Array, "65536 width"] = field(trainable=False)
    signs: Float[Array, " columns"] = field(trainable=False)
    small_q: Float[Array, "order order"] = field(trainable=False)

    def __check_init__(self) -> None:
        rows, columns = self.shape
        blocks, _, block_bytes = self.spec.tape_shape(columns)
        assert self.codes.shape == (rows, blocks * block_bytes)
        assert self.codes.dtype == jnp.uint8
        assert self.scales.shape == self.gains.shape == (rows,)
        assert self.scales.dtype == jnp.float16
        assert self.table.shape == (65536, self.spec.vector_width)
        assert self.table.dtype == self.signs.dtype == self.small_q.dtype == jnp.float32
        order = columns // (columns & -columns)
        assert self.small_q.shape == (order, order)

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape[0], self.signs.shape[0]

    @property
    def dtype(self) -> DTypeLike:
        return self.gains.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return replace(self, gains=self.gains.astype(dtype))

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> Self:
        row_axis, _ = self.spec.layout.weight_partition(0, is_sharded=self.is_sharded)
        return replace(
            self,
            sharding_config=sharding_config,
            codes=with_sharding(self.codes, sharding_config.resolve_sharding((row_axis, None))),
            scales=with_sharding(self.scales, sharding_config.resolve_sharding((row_axis,))),
            gains=with_sharding(self.gains, sharding_config.resolve_sharding((row_axis,))),
            table=with_sharding(self.table, sharding_config.resolve_sharding((None, None))),
            signs=with_sharding(self.signs, sharding_config.resolve_sharding((None,))),
            small_q=with_sharding(self.small_q, sharding_config.resolve_sharding((None, None))),
        )

    def rotated_weights(self) -> Array:
        return self._rotated_rows(self.codes, self.scales, self.gains)

    def _rotated_rows(self, codes: Array, scales: Array, gains: Array) -> Array:
        states = self.spec.states(codes, self.shape[1])
        row_axis, _ = sharding_of(codes).spec
        values = self.table.at[states].get(out_sharding=PartitionSpec(row_axis, None, None))
        values = values.reshape(codes.shape[0], self.shape[1])
        # The checkpoint's two FP32 multiplies must not be folded into one scale.
        scaled = jax.lax.optimization_barrier(values * scales.astype(jnp.float32)[:, None])
        return scaled * gains.astype(jnp.float32)[:, None]

    def decompress(self) -> Array:
        return (full_rotation(self.rotated_weights(), self.small_q) * self.signs).astype(self.dtype)

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
        if transposed:
            with use_dot_algorithm_preset(forward_pass_config.precision):
                return self.decompress().T.astype(vector.dtype) @ vector
        vector = with_sharding(vector, self.sharding_config.make_sharding((None,)))

        def shard_dot(matrix: STrellisMatrix, inputs: Array) -> Array:
            def row_dot(row: tuple[Array, Array, Array]) -> Array:
                codes, scale, gain = row
                rotated = matrix._rotated_rows(codes[None], scale[None], gain[None])[0]
                weights = (full_rotation(rotated, matrix.small_q) * matrix.signs).astype(matrix.dtype)
                return weights.astype(inputs.dtype) @ inputs

            return jax.lax.map(row_dot, (matrix.codes, matrix.scales, matrix.gains), batch_size=128)

        row_axis, _ = sharding_of(self.codes).spec
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return jax.shard_map(
                shard_dot,
                mesh=self.sharding_config.mesh,
                in_specs=(jax.tree.map(lambda array: sharding_of(array).spec, self), PartitionSpec(None)),
                out_specs=PartitionSpec(row_axis),
            )(self, vector)
