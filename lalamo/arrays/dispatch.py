from __future__ import annotations

from collections.abc import Mapping

from jax import lax
from jaxtyping import Array, ArrayLike, DTypeLike

import quax

from .awq import AWQQuantArray
from .base import QuantArray
from .composite import CompositeArray
from .full_precision import FullPrecisionArray
from .lora import LoraArray
from .mlx import MLXQuantArray
from .quant_format import QuantFormat


def quant_array_import_weights(
    weights_map: Mapping[str, Array],
    *,
    quant_format: QuantFormat,
    precision: DTypeLike,
    group_size: int | None = None,
    bits: int | None = None,
) -> QuantArray:
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


def _resolve_value(x: Array | quax.Value) -> Array:
    if isinstance(x, (QuantArray, LoraArray, CompositeArray)):
        return x.value
    return x


def _materialise_dot(
    lhs: Array | quax.Value,
    rhs: Array | quax.Value,
    dimension_numbers: lax.DotDimensionNumbers,
    **kwargs: object,
) -> Array:
    return lax.dot_general(_resolve_value(lhs), _resolve_value(rhs), dimension_numbers, **kwargs)


@quax.register(lax.dot_general_p)
def _(
    lhs: CompositeArray, rhs: ArrayLike | quax.Value, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object
) -> Array:
    result = lax.dot_general(lhs.base.value, rhs, dimension_numbers, **kwargs)
    for lora in lhs.loras:
        up_x = lax.dot_general(lora.up, rhs, dimension_numbers, **kwargs)
        ((_lhs_contract, _rhs_contract), (lhs_batch, _rhs_batch)) = dimension_numbers
        down_contract = (lora.down.ndim - 1,)
        up_x_contract = (len(lhs_batch),)
        down_dn = ((down_contract, up_x_contract), (lhs_batch, tuple(range(len(lhs_batch)))))
        result = result + lora.scale * lax.dot_general(lora.down, up_x, down_dn, **kwargs)
    return result


@quax.register(lax.dot_general_p)
def _(
    lhs: ArrayLike | quax.Value, rhs: CompositeArray, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object
) -> Array:
    return _materialise_dot(lhs, rhs, dimension_numbers, **kwargs)


@quax.register(lax.dot_general_p)
def _(
    lhs: CompositeArray, rhs: CompositeArray, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object
) -> Array:
    return _materialise_dot(lhs, rhs, dimension_numbers, **kwargs)


@quax.register(lax.dot_general_p)
def _(
    lhs: QuantArray, rhs: ArrayLike | quax.Value, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object
) -> Array:
    return _materialise_dot(lhs, rhs, dimension_numbers, **kwargs)


@quax.register(lax.dot_general_p)
def _(
    lhs: ArrayLike | quax.Value, rhs: QuantArray, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object
) -> Array:
    return _materialise_dot(lhs, rhs, dimension_numbers, **kwargs)


@quax.register(lax.dot_general_p)
def _(lhs: QuantArray, rhs: QuantArray, *, dimension_numbers: lax.DotDimensionNumbers, **kwargs: object) -> Array:
    return _materialise_dot(lhs, rhs, dimension_numbers, **kwargs)
