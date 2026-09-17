from dataclasses import dataclass, replace
from typing import Literal, Self

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_array
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
from .utils.row_dot import row_batched_dot
from .utils.s_gains import SScaleAxis, apply_post_gains


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
    result = result / jnp.sqrt(jnp.asarray(width, dtype=values.dtype))
    if order == 1:
        return (result * small_q[0, 0]).reshape(values.shape)
    return jnp.matmul(result, small_q, precision=DotAlgorithmPreset.F32_F32_F32).reshape(values.shape)


@dataclass(frozen=True)
class STrellisSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 8]
    restart_columns: Literal[0, 64]
    layout: Layout = Layout.OUTPUT_INPUT
    scale_dtype: Literal["float16", "float32"] = "float16"
    pre_gain_count: int = 0
    post_gain_axes: tuple[SScaleAxis, ...] = ()

    def __post_init__(self) -> None:
        assert self.scale_dtype in ("float16", "float32")
        assert self.pre_gain_count >= 0
        assert all(isinstance(axis, SScaleAxis) for axis in self.post_gain_axes)
        if self.layout != Layout.OUTPUT_INPUT:
            raise ValueError("S trellis matrices require output-input layout")
        if (self.vector_width, self.transition_bits, self.restart_columns) not in (
            (2, 4, 0),
            (2, 6, 0),
            (2, 8, 0),
            (4, 8, 0),
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
        if not is_dummy_array(weights):
            raise ValueError("S checkpoints must be loaded from saved parameters; fitting is not supported")
        rows, columns = weights.shape
        blocks, _, block_bytes = self.tape_shape(columns)
        order = columns // (columns & -columns)
        initializer = EmptyInitializer(weights.dtype, sharding_config)
        return STrellisMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            codes=initializer.zeros((rows, blocks * block_bytes), dtype=jnp.uint8),
            scales=initializer.zeros((rows,), dtype=jnp.dtype(self.scale_dtype)),
            gains=initializer.zeros((rows,)),
            table=initializer.zeros((65536, self.vector_width), dtype=jnp.float32),
            signs=initializer.zeros((columns,), dtype=jnp.float32),
            small_q=initializer.zeros((order, order), dtype=jnp.float32),
            pre_gains=tuple(initializer.zeros((rows,), dtype=jnp.float32) for _ in range(self.pre_gain_count)),
            post_gains=tuple(
                initializer.zeros((rows if axis == SScaleAxis.ROW else columns,), dtype=jnp.float32)
                for axis in self.post_gain_axes
            ),
        ).switch_sharding_config(sharding_config)


class STrellisMatrix(WeightMatrix[STrellisSpec]):
    codes: UInt8[Array, "rows bytes"]
    scales: Float[Array, " rows"]
    gains: Float[Array, " rows"]
    table: Float[Array, "65536 width"] = field(trainable=False)
    signs: Float[Array, " columns"] = field(trainable=False)
    small_q: Float[Array, "order order"] = field(trainable=False)
    pre_gains: tuple[Array, ...] = ()
    post_gains: tuple[Array, ...] = ()

    def __check_init__(self) -> None:
        rows, columns = self.shape
        blocks, _, block_bytes = self.spec.tape_shape(columns)
        assert self.codes.shape == (rows, blocks * block_bytes)
        assert self.codes.dtype == jnp.uint8
        assert self.scales.shape == self.gains.shape == (rows,)
        assert self.scales.dtype == jnp.dtype(self.spec.scale_dtype)
        assert self.table.shape == (65536, self.spec.vector_width)
        assert self.table.dtype == self.signs.dtype == self.small_q.dtype == jnp.float32
        order = columns // (columns & -columns)
        assert self.small_q.shape == (order, order)
        assert len(self.pre_gains) == self.spec.pre_gain_count
        assert all(gain.shape == (rows,) and gain.dtype == jnp.float32 for gain in self.pre_gains)
        for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True):
            assert gain.shape == (rows if axis == SScaleAxis.ROW else columns,)
            assert gain.dtype == jnp.float32

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
            pre_gains=tuple(
                with_sharding(gain, sharding_config.resolve_sharding((row_axis,))) for gain in self.pre_gains
            ),
            post_gains=tuple(
                with_sharding(gain, sharding_config.resolve_sharding((row_axis if axis == SScaleAxis.ROW else None,)))
                for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            ),
        )

    def rotated_weights(self) -> Array:
        return self._rotated_rows(self.codes, self.scales, self.gains, self.pre_gains)

    def _rotated_rows(self, codes: Array, scales: Array, gains: Array, pre_gains: tuple[Array, ...]) -> Array:
        states = self.spec.states(codes, self.shape[1])
        row_axis, _ = sharding_of(codes).spec
        values = self.table.at[states].get(out_sharding=PartitionSpec(row_axis, None, None))
        values = values.reshape(codes.shape[0], self.shape[1])
        # The checkpoint's two FP32 multiplies must not be folded into one scale.
        scaled = jax.lax.optimization_barrier(values * scales.astype(jnp.float32)[:, None])
        scaled = jax.lax.optimization_barrier(scaled * gains.astype(jnp.float32)[:, None])
        for gain in pre_gains:
            scaled = jax.lax.optimization_barrier(scaled * gain[:, None])
        return scaled

    def decompress(self) -> Array:
        weights = full_rotation(self.rotated_weights(), self.small_q) * self.signs
        return apply_post_gains(weights, self.spec.post_gain_axes, self.post_gains, self.dtype)

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
                return Layout.INPUT_OUTPUT.matmul(self.decompress().astype(vector.dtype), vector)

        def decode_row(row: tuple[Array, ...]) -> Array:
            codes, scale, gain, *factors = row
            pre_gains = tuple(factor[None] for factor in factors[: self.spec.pre_gain_count])
            row_gains = iter(factors[self.spec.pre_gain_count :])
            post_gains = tuple(
                next(row_gains) if axis == SScaleAxis.ROW else factor
                for axis, factor in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            )
            rotated = self._rotated_rows(codes[None], scale[None], gain[None], pre_gains)[0]
            weights = full_rotation(rotated, self.small_q) * self.signs
            return apply_post_gains(weights, self.spec.post_gain_axes, post_gains, self.dtype)

        row_gains = tuple(
            gain
            for axis, gain in zip(self.spec.post_gain_axes, self.post_gains, strict=True)
            if axis == SScaleAxis.ROW
        )
        return row_batched_dot(
            decode_row,
            (self.codes, self.scales, self.gains, *self.pre_gains, *row_gains),
            vector,
            forward_pass_config.precision,
        )
