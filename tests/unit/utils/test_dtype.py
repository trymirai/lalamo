import jax.numpy as jnp
import pytest

from lalamo.utils.dtype import cast_if_float, dtype_to_str, str_to_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.int8,
        jnp.int32,
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        jnp.complex64,
    ],
)
def test_dtype_roundtrips_through_string(dtype: jnp.dtype) -> None:
    dtype_str = dtype_to_str(dtype)

    assert str_to_dtype(dtype_str) == dtype


def test_dtype_to_str_handles_jax_array_dtype() -> None:
    array = jnp.ones((2,), dtype=jnp.float32)

    assert dtype_to_str(array.dtype) == "float32"


def test_dtype_to_str_uses_plain_bfloat16_name() -> None:
    assert dtype_to_str(jnp.bfloat16) == "bfloat16"


def test_str_to_dtype_rejects_unknown_dtype() -> None:
    with pytest.raises(KeyError):
        str_to_dtype("not-a-dtype")


def test_cast_if_float_casts_floating_arrays() -> None:
    array = jnp.ones((2,), dtype=jnp.float32)

    result = cast_if_float(array, jnp.bfloat16)

    assert result.dtype == jnp.bfloat16
    assert result.shape == array.shape


def test_cast_if_float_keeps_integer_arrays_unchanged() -> None:
    array = jnp.ones((2,), dtype=jnp.int32)

    result = cast_if_float(array, jnp.bfloat16)

    assert result.dtype == jnp.int32
    assert result.shape == array.shape
