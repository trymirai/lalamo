from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import MatmulConfig, WeightMatrix, WeightMatrixSpec


@dataclass(frozen=True)
class SurgeryWeightMatrixSpec(WeightMatrixSpec):
    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> WeightMatrix:
        raise NotImplementedError


class SurgeryWeightMatrixBase(WeightMatrix[SurgeryWeightMatrixSpec]):
    weights: Float[Array, "out_channels in_channels"]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def astype(self, dtype: DTypeLike) -> Self:
        return type(self)(spec=self.spec, weights=self.weights.astype(dtype))

    def decompress(self) -> Float[Array, "out_channels in_channels"]:
        return self.weights

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, " out_channels"]:
        return self.weights @ vector


class TemplateWeightMatrix(SurgeryWeightMatrixBase):
    pass


class ValueWeightMatrix(SurgeryWeightMatrixBase):
    pass


def _template_matrix(shape: tuple[int, int] = (2, 3), dtype: DTypeLike = jnp.float32) -> TemplateWeightMatrix:
    return TemplateWeightMatrix(spec=SurgeryWeightMatrixSpec(), weights=jnp.zeros(shape, dtype=dtype))


def _value_matrix(shape: tuple[int, int] = (2, 3), dtype: DTypeLike = jnp.float32) -> ValueWeightMatrix:
    return ValueWeightMatrix(spec=SurgeryWeightMatrixSpec(), weights=jnp.ones(shape, dtype=dtype))


def test_load_as_reports_shape_mismatch_inside_non_array_pytree() -> None:
    template = {"weights": jnp.zeros((2, 3), dtype=jnp.float32), "name": "linear"}
    value = {"weights": jnp.ones((2, 4), dtype=jnp.float32), "name": "linear"}

    with pytest.raises(ValueError, match="shape"):
        load_as(template, value)


def test_load_as_reports_non_array_leaf_type_mismatch() -> None:
    template = {"name": "linear"}
    value = {"name": 1}

    with pytest.raises(TypeError, match="type"):
        load_as(template, value)


def test_load_as_casts_array_dtype_when_allowed() -> None:
    template = jnp.zeros((2, 3), dtype=jnp.float32)
    value = jnp.ones((2, 3), dtype=jnp.float16)

    result = load_as(template, value, allow_dtype_cast=True)

    assert result.dtype == jnp.float32
    assert jnp.all(result == 1)


def test_load_as_rejects_array_shape_mismatch() -> None:
    template = jnp.zeros((2, 3), dtype=jnp.float32)
    value = jnp.ones((2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="shape"):
        load_as(template, value)


def test_load_as_rejects_array_sharding_mismatch() -> None:
    template = ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32, sharding=None)
    value = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="sharding"):
        load_as(template, value)


def test_load_as_accepts_different_weight_matrix_subclasses() -> None:
    template = _template_matrix()
    value = _value_matrix()

    result = load_as(template, value)

    assert result is value


def test_load_as_casts_weight_matrix_dtype_when_allowed() -> None:
    template = _template_matrix(dtype=jnp.float32)
    value = _value_matrix(dtype=jnp.float16)

    result = load_as(template, value, allow_dtype_cast=True)

    assert isinstance(result, ValueWeightMatrix)
    assert result.dtype == jnp.float32
    assert jnp.all(result.weights == 1)


def test_load_as_rejects_weight_matrix_dtype_mismatch() -> None:
    template = _template_matrix(dtype=jnp.float32)
    value = _value_matrix(dtype=jnp.float16)

    with pytest.raises(ValueError, match="dtype"):
        load_as(template, value)


def test_load_as_rejects_weight_matrix_shape_mismatch() -> None:
    template = _template_matrix(shape=(2, 3))
    value = _value_matrix(shape=(2, 4))

    with pytest.raises(ValueError, match="shape"):
        load_as(template, value)
