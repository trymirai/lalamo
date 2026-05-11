import jax.numpy as jnp
import pytest

from lalamo.utils.dtype import cast_if_float, dtype_to_str, str_to_dtype


@pytest.mark.parametrize(
    ("dtype", "name"),
    [
        (jnp.int8, "int8"),
        (jnp.int32, "int32"),
        (jnp.bfloat16, "bfloat16"),
        (jnp.float16, "float16"),
        (jnp.float32, "float32"),
        (jnp.float64, "float64"),
        (jnp.complex64, "complex64"),
    ],
)
def test_dtype_roundtrips_through_string(dtype: jnp.dtype, name: str) -> None:
    assert dtype_to_str(dtype) == name
    assert str_to_dtype(name) == dtype


def test_str_to_dtype_rejects_unknown_dtype() -> None:
    with pytest.raises(KeyError):
        str_to_dtype("not-a-dtype")


@pytest.mark.parametrize(("input_dtype", "expected_dtype"), [(jnp.float32, jnp.bfloat16), (jnp.int32, jnp.int32)])
def test_cast_if_float_only_casts_floating_arrays(input_dtype: jnp.dtype, expected_dtype: jnp.dtype) -> None:
    result = cast_if_float(jnp.ones((2,), dtype=input_dtype), jnp.bfloat16)

    assert result.dtype == expected_dtype
    assert result.shape == (2,)
