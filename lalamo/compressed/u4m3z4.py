from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.core import Tracer
from jaxtyping import Array, DTypeLike, Float, Float32, Key, UInt8

from lalamo.compressed.utils.packing import unpack_uint8_to_uint
from lalamo.exportable import ExportResults
from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, with_sharding
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = ["U4M3Z4Matrix", "U4M3Z4Spec"]


@dataclass(frozen=True)
class U4M3Z4Spec(WeightMatrixSpec):
    bits: Literal[4] = 4
    group_size: Literal[32, 64] = 64
    multiplier_bits: Literal[3] = 3
    zero_point_bits: Literal[4] = 4
    layout: Literal[Layout.OUTPUT_INPUT] = Layout.OUTPUT_INPUT

    def from_packed_parameters(
        self,
        *,
        packed_weights: UInt8[Array, "*components rows packed_cols"],
        base_scales: Float32[Array, "*components rows"],
        mz_plane: UInt8[Array, "*components rows groups"],
        dtype: DTypeLike = jnp.float32,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "U4M3Z4Matrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise NotImplementedError("U4M3Z4 supports inference matrices only")
        dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("U4M3Z4 compute dtype must be floating point")
        if packed_weights.ndim < 2:
            raise ValueError("packed_weights must have shape [..., rows, packed_cols]")
        *leading_dims, rows, packed_cols = packed_weights.shape
        packed_group_size = self.group_size // 2
        if rows == 0 or packed_cols == 0 or packed_cols % packed_group_size:
            raise ValueError("packed_weights must have positive rows and columns divisible by group_size / 2")
        group_count = packed_cols // packed_group_size
        if base_scales.shape != (*leading_dims, rows):
            raise ValueError("base_scales must have shape [..., rows]")
        if mz_plane.shape != (*leading_dims, rows, group_count):
            raise ValueError("mz_plane must have shape [..., rows, groups]")
        if packed_weights.dtype != jnp.uint8 or mz_plane.dtype != jnp.uint8:
            raise ValueError("packed_weights and mz_plane must be uint8")
        if base_scales.dtype != jnp.float32:
            raise ValueError("base_scales must be float32")

        if not any(isinstance(value, (ShapeDtypeStruct, Tracer)) for value in (packed_weights, base_scales, mz_plane)):
            if bool(jnp.any((mz_plane & jnp.uint8(0x80)) != 0)):
                raise ValueError("mz_plane reserved bit 7 must be zero")
            if bool(jnp.any(~jnp.isfinite(base_scales) | (base_scales <= 0))):
                raise ValueError("base_scales must be finite and positive")

        weight_axes = self.layout.weight_partition(len(leading_dims), is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        return U4M3Z4Matrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            packed_weights=with_sharding(packed_weights, weight_sharding),
            base_scales=with_sharding(base_scales, sharding_config.resolve_sharding(weight_axes[:-1])),
            mz_plane=with_sharding(mz_plane, weight_sharding),
            dtype_=dtype,
        )

    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "U4M3Z4Matrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise NotImplementedError("U4M3Z4 supports inference matrices only")
        if not isinstance(weights, ShapeDtypeStruct):
            raise NotImplementedError(
                "U4M3Z4 dense-weight quantization is not implemented; "
                "construct an inference matrix with from_packed_parameters"
            )
        if weights.ndim < 2:
            raise ValueError("weights must have shape [..., rows, cols]")
        if not jnp.issubdtype(weights.dtype, jnp.floating):
            raise ValueError("weights must use a floating-point dtype")

        *leading_dims, rows, cols = weights.shape
        weight_axes = self.layout.weight_partition(len(leading_dims), is_sharded=is_sharded)
        weight_sharding = sharding_config.resolve_sharding(weight_axes)
        matrix = self.from_packed_parameters(
            packed_weights=dummy_array(
                (*leading_dims, rows, cols // 2),
                jnp.uint8,
                weight_sharding,
            ),
            base_scales=dummy_array(
                (*leading_dims, rows),
                jnp.float32,
                sharding_config.resolve_sharding(weight_axes[:-1]),
            ),
            mz_plane=dummy_array(
                (*leading_dims, rows, cols // self.group_size),
                jnp.uint8,
                weight_sharding,
            ),
            dtype=weights.dtype,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )
        if matrix.shape != weights.shape:
            raise ValueError("weights columns must be positive and divisible by group_size")
        return matrix


class U4M3Z4Matrix(WeightMatrix[U4M3Z4Spec]):
    packed_weights: UInt8[Array, "*components rows packed_cols"]
    base_scales: Float32[Array, "*components rows"]
    mz_plane: UInt8[Array, "*components rows groups"]
    dtype_: DTypeLike = field(static=True)

    @property
    def shape(self) -> tuple[int, ...]:
        *leading_dims, rows, packed_cols = self.packed_weights.shape
        return (*leading_dims, rows, packed_cols * 2)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    def astype(self, dtype: DTypeLike) -> "U4M3Z4Matrix":
        dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("U4M3Z4 compute dtype must be floating point")
        return replace(self, dtype_=dtype)

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "*components out_channels in_channels"]:
        codes = unpack_uint8_to_uint(
            self.packed_weights,
            bits=4,
            dtype=jnp.int16,
            unpacked_last_axis_dim=self.shape[-1],
        )
        grouped_codes = codes.reshape(*self.shape[:-1], -1, self.spec.group_size)
        signed_delta = grouped_codes - (self.mz_plane & jnp.uint8(0x0F)).astype(jnp.int16)[..., None]
        multipliers = ((self.mz_plane >> jnp.uint8(4)) & jnp.uint8(0x07)) + jnp.uint8(1)
        result = (
            self.base_scales[..., :, None, None]
            * multipliers.astype(jnp.float32)[..., None]
            * signed_delta.astype(jnp.float32)
        )
        return result.reshape(self.shape).astype(self.dtype_)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        del keychain
        self._raise_if_batched()
        layout = self.spec.layout
        if transposed:
            layout = layout.transpose()
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(self.decompress().astype(vector.dtype), vector)

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={"weights": self.packed_weights, "scales": self.base_scales, "mz_plane": self.mz_plane},
            metadata={"spec": self.spec.to_json()},
        )

    def load_exported(
        self,
        exported_data: ExportResults,
        *,
        prefix: ParameterPath | None = None,
    ) -> "U4M3Z4Matrix":
        if prefix is None:
            prefix = ParameterPath()
        loaded_spec = WeightMatrixSpec.from_json(exported_data.metadata[prefix / "spec"])
        if loaded_spec != self.spec:
            raise ValueError(f"WeightMatrix spec mismatch: expected {self.spec}, got {loaded_spec}")
        matrix = self.spec.from_packed_parameters(
            packed_weights=exported_data.arrays[prefix / "weights"],
            base_scales=exported_data.arrays[prefix / "scales"],
            mz_plane=exported_data.arrays[prefix / "mz_plane"],
            dtype=self.dtype,
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )
        return load_as(self, matrix)

    def switch_implementation(self, implementation: CompressionImplementation) -> "U4M3Z4Matrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise NotImplementedError("U4M3Z4 supports inference matrices only")
        return self

    def switch_sharding_config(self, sharding_config: ShardingConfig) -> "U4M3Z4Matrix":
        if sharding_config == self.sharding_config:
            return self
        weight_axes = self.spec.layout.weight_partition(self.packed_weights.ndim - 2, is_sharded=self.is_sharded)
        return replace(
            self,
            sharding_config=sharding_config,
            packed_weights=with_sharding(self.packed_weights, sharding_config.resolve_sharding(weight_axes)),
            base_scales=with_sharding(self.base_scales, sharding_config.resolve_sharding(weight_axes[:-1])),
            mz_plane=with_sharding(self.mz_plane, sharding_config.resolve_sharding(weight_axes)),
        )
