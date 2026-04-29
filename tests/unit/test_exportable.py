from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from lalamo.exportable import Exportable, ExportResults
from lalamo.module import field
from lalamo.utils.parameter_path import ParameterPath


class Leaf(Exportable, eqx.Module):
    weight: jax.Array
    bias: jax.Array | None


class Container(Exportable, eqx.Module):
    leaf: Leaf
    nested: tuple[jax.Array, ...]
    optional: jax.Array | None
    opaque: object
    name: str = field(static=True)


class CustomLeaf(Exportable, eqx.Module):
    value: jax.Array
    scale: float = field(static=True)

    def export(self) -> ExportResults:
        return ExportResults(
            arrays={"scaled": self.value * self.scale},
            metadata={"scale": self.scale},
        )

    def load_exported(
        self,
        expored_data: ExportResults,
        allow_dtype_cast: bool = False,  # noqa: ARG002
        *,
        prefix: ParameterPath | None = None,
    ) -> Self:
        if prefix is None:
            prefix = ParameterPath()
        scale = expored_data.metadata[prefix / "scale"]
        assert isinstance(scale, float)
        return type(self)(value=expored_data.arrays[prefix / "scaled"] / scale, scale=scale)


class CustomContainer(Exportable, eqx.Module):
    child: CustomLeaf
    residual: jax.Array


def _single_device_sharding() -> NamedSharding:
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    return NamedSharding(mesh, PartitionSpec())


def test_export_flattens_arrays_and_skips_none_and_static_fields() -> None:
    container = Container(
        leaf=Leaf(weight=jnp.ones((2, 3)), bias=None),
        nested=(jnp.arange(2),),
        optional=None,
        opaque=object(),
        name="not-exported",
    )

    arrays, metadata = container.export()

    assert set(arrays) == {"leaf.weight", "nested.0"}
    assert jnp.array_equal(arrays["leaf.weight"], container.leaf.weight)
    assert jnp.array_equal(arrays["nested.0"], container.nested[0])
    assert metadata == {}


def test_export_prefixes_nested_exportable_arrays_and_metadata() -> None:
    container = CustomContainer(
        child=CustomLeaf(value=jnp.array([1.0, 2.0]), scale=3.0),
        residual=jnp.array([4.0]),
    )

    arrays, metadata = container.export()

    assert set(arrays) == {"child.scaled", "residual"}
    assert jnp.array_equal(arrays["child.scaled"], jnp.array([3.0, 6.0]))
    assert jnp.array_equal(arrays["residual"], container.residual)
    assert metadata == {"child.scale": 3.0}


def test_load_exported_restores_arrays_into_skeleton() -> None:
    original = Container(
        leaf=Leaf(weight=jnp.arange(6, dtype=jnp.float32).reshape(2, 3), bias=jnp.array([1.0, 2.0])),
        nested=(jnp.array([3.0]),),
        optional=None,
        opaque=object(),
        name="original",
    )
    skeleton_opaque = object()
    skeleton = Container(
        leaf=Leaf(weight=jnp.zeros((2, 3), dtype=jnp.float32), bias=jnp.zeros((2,), dtype=jnp.float32)),
        nested=(jnp.zeros((1,), dtype=jnp.float32),),
        optional=None,
        opaque=skeleton_opaque,
        name="skeleton",
    )

    restored = skeleton.load_exported(original.export())

    assert jnp.array_equal(restored.leaf.weight, original.leaf.weight)
    assert restored.leaf.bias is not None
    assert original.leaf.bias is not None
    assert jnp.array_equal(restored.leaf.bias, original.leaf.bias)
    assert jnp.array_equal(restored.nested[0], original.nested[0])
    assert restored.optional is None
    assert restored.opaque is skeleton_opaque
    assert restored.name == "skeleton"


def test_load_exported_uses_prefix() -> None:
    skeleton = Leaf(weight=jnp.zeros((2,), dtype=jnp.float32), bias=None)
    exported = ExportResults(
        arrays={"prefix.weight": jnp.ones((2,), dtype=jnp.float32)},
        metadata={},
    )

    restored = skeleton.load_exported(exported, prefix=ParameterPath("prefix"))

    assert jnp.array_equal(restored.weight, jnp.ones((2,), dtype=jnp.float32))
    assert restored.bias is None


def test_load_exported_reports_missing_array() -> None:
    skeleton = Leaf(weight=jnp.zeros((2,), dtype=jnp.float32), bias=None)

    with pytest.raises(KeyError, match="weight"):
        skeleton.load_exported(ExportResults(arrays={}, metadata={}))


def test_load_exported_rejects_shape_mismatch() -> None:
    skeleton = Leaf(weight=jnp.zeros((2,), dtype=jnp.float32), bias=None)
    exported = ExportResults(arrays={"weight": jnp.ones((3,), dtype=jnp.float32)}, metadata={})

    with pytest.raises(ValueError, match="shape"):
        skeleton.load_exported(exported)


def test_load_exported_rejects_dtype_mismatch_by_default() -> None:
    skeleton = Leaf(weight=jnp.zeros((2,), dtype=jnp.float32), bias=None)
    exported = ExportResults(arrays={"weight": jnp.ones((2,), dtype=jnp.float16)}, metadata={})

    with pytest.raises(ValueError, match="dtype"):
        skeleton.load_exported(exported)


def test_load_exported_casts_dtype_when_allowed() -> None:
    skeleton = Leaf(weight=jnp.zeros((2,), dtype=jnp.float32), bias=None)
    exported = ExportResults(arrays={"weight": jnp.ones((2,), dtype=jnp.float16)}, metadata={})

    restored = skeleton.load_exported(exported, allow_dtype_cast=True)

    assert restored.weight.dtype == jnp.float32
    assert jnp.array_equal(restored.weight, jnp.ones((2,), dtype=jnp.float32))


def test_load_exported_preserves_template_sharding() -> None:
    sharding = _single_device_sharding()
    skeleton = Leaf(
        weight=jax.device_put(jnp.zeros((2,), dtype=jnp.float32), sharding),
        bias=None,
    )
    exported = ExportResults(arrays={"weight": jnp.ones((2,), dtype=jnp.float32)}, metadata={})

    restored = skeleton.load_exported(exported)

    assert restored.weight.sharding == sharding
    assert jnp.array_equal(restored.weight, jnp.ones((2,), dtype=jnp.float32))


def test_nested_exportable_load_uses_child_override_with_prefixed_metadata() -> None:
    original = CustomContainer(
        child=CustomLeaf(value=jnp.array([1.0, 2.0]), scale=4.0),
        residual=jnp.array([5.0]),
    )
    skeleton = CustomContainer(
        child=CustomLeaf(value=jnp.zeros((2,)), scale=1.0),
        residual=jnp.zeros((1,)),
    )

    restored = skeleton.load_exported(original.export())

    assert restored.child.scale == 4.0
    assert jnp.array_equal(restored.child.value, original.child.value)
    assert jnp.array_equal(restored.residual, original.residual)
