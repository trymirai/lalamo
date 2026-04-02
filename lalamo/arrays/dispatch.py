from __future__ import annotations

from collections.abc import Mapping

from jaxtyping import Array, DTypeLike

from .awq import AWQQuantArray
from .base import CompressedArray
from .full_precision import FullPrecisionArray
from .mlx import MLXQuantArray
from .quant_format import QuantFormat


def quant_array_import_weights(
    weights_map: Mapping[str, Array],
    *,
    quant_format: QuantFormat,
    precision: DTypeLike,
    group_size: int | None = None,
    bits: int | None = None,
) -> CompressedArray:
    match quant_format:
        case QuantFormat.FULL_PRECISION:
            return FullPrecisionArray.import_weights(weights_map)
        case QuantFormat.AWQ:
            if group_size is None or bits is None:
                raise ValueError("group_size and bits are required for AWQ format")
            return AWQQuantArray.import_weights(weights_map, precision=precision, group_size=group_size, bits=bits)
        case QuantFormat.MLX:
            if group_size is None or bits is None:
                raise ValueError("group_size and bits are required for MLX format")
            return MLXQuantArray.import_weights(weights_map, precision=precision, group_size=group_size, bits=bits)
