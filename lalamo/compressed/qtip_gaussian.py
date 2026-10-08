from dataclasses import dataclass, replace
from math import ceil
from typing import Literal, Self

import jax.numpy as jnp
import numpy as np
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, DTypeLike, Float, Key, UInt8

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import is_dummy_or_tracer
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

from .trellis import states_to_levels
from .utils.row_dot import row_batched_dot

# Every step's 16-bit state hashes to its levels; each tape block opens with its first state in two bytes.
STATE_BITS = 16
# The codebook is [scale, offset of each column class], the class of a column being its index modulo four.
COLUMN_CLASSES = 4


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
class QtipGaussianSpec(WeightMatrixSpec):
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 8]
    restart_columns: Literal[0, 64]

    def __post_init__(self) -> None:
        if (self.vector_width, self.transition_bits, self.restart_columns) not in (
            (2, 4, 0),
            (2, 6, 0),
            (2, 8, 0),
            (4, 8, 0),
            (4, 8, 64),
        ):
            raise ValueError("Unsupported QTIP Gaussian trellis layout")

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
        """Each block is an MSB-first bit stream; state g is its 16-bit window at bit g * transition_bits."""
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

    def row_arrays(self) -> tuple[Array, Array]:
        return self.codes, self.scales

    def decode_rows(self, rows: tuple[Array, Array], dtype: DTypeLike) -> Array:
        codes, scales = rows
        columns = self.columns
        levels = states_to_levels(self.spec.states(codes, columns))[..., : self.spec.vector_width]
        levels = levels.reshape(*codes.shape[:-1], columns).astype(jnp.float32)
        scale = self.codebook[0]
        offsets = self.codebook[1 + jnp.arange(columns) % COLUMN_CLASSES]
        return ((scale * levels + offsets) * scales[..., None]).astype(dtype)

    def decompress(self) -> Array:
        return self.decode_rows(self.row_arrays(), self.dtype)

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
        return row_batched_dot(
            lambda rows: self.decode_rows(rows, self.dtype), self.row_arrays(), vector, forward_pass_config.precision
        )
