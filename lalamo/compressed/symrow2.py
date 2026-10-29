from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array, BFloat16, DTypeLike, Float, Key, UInt8

from lalamo.module import Keychain, field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import ShardingConfig, with_sharding
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)

__all__ = ["SymRow2Matrix", "SymRow2Spec"]

# Dequantization is scale * (code - 1.5); the scale is searched over fractions of max|w| / 1.5.
_CODE_ZERO_POINT = 1.5
_SCALE_SEARCH_FRACTIONS = (0.6, 0.7, 0.8, 0.9, 1.0)
_MIN_SCALE = 1e-6

# Bit offset of the code of (jj, cc) inside a packed byte.
_BYTE_SHIFTS = 4 * np.arange(2, dtype=np.uint8)[:, None] + 2 * np.arange(2, dtype=np.uint8)[None, :]


def _fit_row_codes(
    weights: Float[Array, "rows cols"],
) -> tuple[UInt8[Array, "rows cols"], BFloat16[Array, " rows"]]:
    weights = weights.astype(jnp.float32)
    max_abs = jnp.max(jnp.abs(weights), axis=-1)
    best_error = jnp.full_like(max_abs, jnp.inf)
    best_scales = jnp.ones_like(max_abs)
    best_codes = jnp.zeros(weights.shape, dtype=jnp.uint8)
    for fraction in _SCALE_SEARCH_FRACTIONS:
        scales = jnp.maximum(jnp.float32(fraction) * max_abs / _CODE_ZERO_POINT, _MIN_SCALE)
        scales = scales.astype(jnp.bfloat16).astype(jnp.float32)
        # Full-shape divisor: XLA turns a broadcast divisor into a reciprocal multiply, which flips exact ties.
        quotient = jax.lax.div(weights, jnp.broadcast_to(scales[:, None], weights.shape))
        codes = jnp.clip(jnp.rint(quotient + _CODE_ZERO_POINT), 0, 3).astype(jnp.uint8)
        error = jnp.sum(
            jnp.square(scales[:, None] * (codes.astype(jnp.float32) - _CODE_ZERO_POINT) - weights), axis=-1
        )
        improved = error < best_error
        best_error = jnp.where(improved, error, best_error)
        best_scales = jnp.where(improved, scales, best_scales)
        best_codes = jnp.where(improved[:, None], codes, best_codes)
    return best_codes, best_scales.astype(jnp.bfloat16)


def pack_layout16(codes: UInt8[Array, "rows cols"]) -> UInt8[Array, "rows packed_cols"]:
    # [K/128][N/16][32 lanes][16 B] flattened to [N, K/4]; the activation order of each 16-code run is baked in.
    rows, cols = codes.shape
    shifted = rearrange(
        codes,
        "(t cc nl2 nl01) (g u jj q1 q0 i) -> jj cc t nl2 nl01 g u q1 q0 i",
        cc=2,
        nl2=2,
        nl01=4,
        u=4,
        jj=2,
        q1=2,
        q0=2,
        i=4,
    )
    shifted = shifted << _BYTE_SHIFTS[:, :, *(None,) * 8]
    blocks = jnp.sum(shifted, axis=(0, 1), dtype=jnp.uint8)
    return rearrange(blocks, "t nl2 nl01 g u q1 q0 i -> (g t nl2 q1 nl01 q0 u i)").reshape(rows, cols // 4)


def unpack_layout16(packed: UInt8[Array, "rows packed_cols"]) -> UInt8[Array, "rows cols"]:
    rows, packed_cols = packed.shape
    lanes = rearrange(
        packed.reshape(-1),
        "(g t nl2 q1 nl01 q0 u i) -> g t nl2 q1 nl01 q0 u i",
        t=rows // 16,
        nl2=2,
        q1=2,
        nl01=4,
        q0=2,
        u=4,
        i=4,
    )
    codes = (lanes[..., None, None] >> _BYTE_SHIFTS) & jnp.uint8(3)
    return rearrange(
        codes,
        "g t nl2 q1 nl01 q0 u i jj cc -> (t cc nl2 nl01) (g u jj q1 q0 i)",
    ).reshape(rows, packed_cols * 4)


@dataclass(frozen=True)
class SymRow2Spec(WeightMatrixSpec):
    layout: Literal[Layout.OUTPUT_INPUT] = Layout.OUTPUT_INPUT

    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "*components out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,  # noqa: ARG002
        preconditioner: Preconditioner | None = None,  # noqa: ARG002
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "SymRow2Matrix":
        if implementation != CompressionImplementation.INFERENCE:
            raise NotImplementedError("SymRow2 supports inference matrices only")
        if weights.ndim != 2 or weights.shape[0] % 16 or weights.shape[1] % 128:
            raise ValueError(f"SymRow2 needs a 2D [N % 16 == 0, K % 128 == 0] matrix, got {weights.shape}")
        # Packed rows are not output rows, so nothing is sharded along them.
        replicated = sharding_config.resolve_sharding((None, None))
        codes, scales = _fit_row_codes(with_sharding(weights, replicated))
        return SymRow2Matrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            weights=with_sharding(pack_layout16(codes), replicated),
            scales=with_sharding(scales, sharding_config.resolve_sharding((None,))),
            dtype_=weights.dtype,
        )


class SymRow2Matrix(WeightMatrix[SymRow2Spec]):
    weights: UInt8[Array, "rows packed_cols"]
    scales: BFloat16[Array, " rows"]
    dtype_: DTypeLike = field(static=True)

    @property
    def shape(self) -> tuple[int, ...]:
        rows, packed_cols = self.weights.shape
        return (rows, packed_cols * 4)

    @property
    def dtype(self) -> DTypeLike:
        return self.dtype_

    def astype(self, dtype: DTypeLike) -> "SymRow2Matrix":
        return replace(self, dtype_=jnp.dtype(dtype))

    def to_full_precision(self) -> FullPrecisionMatrix:
        return FullPrecisionSpec(layout=self.spec.layout).compress(
            self.decompress(),
            sharding_config=self.sharding_config,
            is_sharded=self.is_sharded,
        )

    def decompress(self) -> Float[Array, "out_channels in_channels"]:
        codes = unpack_layout16(self.weights).astype(jnp.float32)
        return (self.scales.astype(jnp.float32)[:, None] * (codes - _CODE_ZERO_POINT)).astype(self.dtype_)

    def dot(
        self,
        vector: Float[Array, " source_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),
        transposed: bool = False,
    ) -> Float[Array, " target_channels"]:
        layout = self.spec.layout.transpose() if transposed else self.spec.layout
        with use_dot_algorithm_preset(forward_pass_config.precision):
            return layout.matmul(self.decompress().astype(vector.dtype), vector)
