from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lalamo.common import ParameterPath

from .awq import AWQQuantArray
from .full_precision import FullPrecisionArray
from .mlx import MLXQuantArray
from .quant_format import QuantFormat

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike

    from .base import CompressedArray


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


def quant_array_from_torch(
    state_dict: Mapping[str, Array],
    *,
    quant_format: QuantFormat,
    prefix: ParameterPath | str,
    precision: DTypeLike,
    group_size: int | None = None,
    bits: int | None = None,
) -> CompressedArray:
    match quant_format:
        case QuantFormat.FULL_PRECISION:
            return FullPrecisionArray.from_weight(state_dict[ParameterPath(prefix) / "weight"], dtype=precision)
        case QuantFormat.AWQ:
            if group_size is None or bits is None:
                raise ValueError("group_size and bits are required for AWQ format")
            path = ParameterPath(prefix)
            return AWQQuantArray.from_packed(
                state_dict[path / "qweight"],
                state_dict[path / "qzeros"],
                state_dict[path / "scales"],
                dtype=precision,
                group_size=group_size,
                bits=bits,
            )
        case QuantFormat.MLX:
            if group_size is None or bits is None:
                raise ValueError("group_size and bits are required for MLX format")
            path = ParameterPath(prefix)
            scales = state_dict[path / "scales"]
            return MLXQuantArray.from_packed(
                state_dict[path / "weight"],
                scales,
                state_dict[path / "biases"],
                dtype=precision,
                expected_in_channels=scales.shape[-1] * group_size,
                group_size=group_size,
                bits=bits,
            )
