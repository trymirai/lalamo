import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array

from lalamo.module import LogicalAxis
from lalamo.utils.dummy_array import contains_dummy_arrays, dummy_array, is_dummy_array, supports_dummy_arrays
from tests.helpers import make_sharding


def test_dummy_array_predicates() -> None:
    value = dummy_array((2,), jnp.float32, make_sharding((None,)))

    assert is_dummy_array(value)
    assert contains_dummy_arrays({"value": value})
    assert not is_dummy_array(jnp.zeros((2,), dtype=jnp.float32))
    assert not contains_dummy_arrays({"value": jnp.zeros((2,), dtype=jnp.float32)})


def test_supports_dummy_arrays_leaves_concrete_arrays_concrete() -> None:
    @supports_dummy_arrays()
    def add_one(values: Array) -> Array:
        return values + 1

    values = jnp.arange(4, dtype=jnp.float32)

    result = add_one(values)

    assert isinstance(result, jax.Array)
    assert jnp.array_equal(result, values + 1)


def test_supports_dummy_arrays_evaluates_shape_for_dummy_arrays() -> None:
    @supports_dummy_arrays()
    def flatten(values: Array) -> Array:
        return jnp.reshape(values, (-1,))

    values = dummy_array((2, 3), jnp.float16, make_sharding((None, None)))

    result = flatten(values)

    assert isinstance(result, ShapeDtypeStruct)
    assert result.shape == (6,)
    assert result.dtype == jnp.dtype(jnp.float16)


def test_supports_dummy_arrays_preserves_inferred_named_sharding(fake_mesh: Mesh) -> None:
    @supports_dummy_arrays()
    def add_one(values: Array) -> Array:
        return values + 1

    values = dummy_array((8,), jnp.float32, make_sharding((LogicalAxis.BATCH,)))

    result = add_one(values)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert tuple(result.sharding.spec) == (LogicalAxis.BATCH,)
    assert result.sharding.mesh.axis_names == fake_mesh.axis_names


def test_supports_dummy_arrays_applies_out_sharding_rule(fake_mesh: Mesh) -> None:
    def shard_like_tensor_axis(input_shardings: tuple[NamedSharding, ...]) -> NamedSharding:
        (source_sharding,) = input_shardings
        return NamedSharding(source_sharding.mesh, PartitionSpec(LogicalAxis.MIXTURE))

    @supports_dummy_arrays(out_sharding_rule=shard_like_tensor_axis)
    def add_one(values: Array) -> Array:
        return values + 1

    values = dummy_array((8,), jnp.float32, make_sharding((LogicalAxis.BATCH,)))

    result = add_one(values)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding.mesh == fake_mesh
    assert tuple(result.sharding.spec) == (LogicalAxis.MIXTURE,)


def test_supports_dummy_arrays_passes_concrete_input_named_sharding_to_rule(fake_mesh: Mesh) -> None:
    def reference_sharding(input_shardings: tuple[NamedSharding, ...]) -> NamedSharding:
        _, reference_sharding = input_shardings
        return reference_sharding

    @supports_dummy_arrays(out_sharding_rule=reference_sharding)
    def add_one(values: Array, _reference: Array) -> Array:
        return values + 1

    values = dummy_array((8,), jnp.float32, make_sharding((LogicalAxis.BATCH,)))
    reference = jax.device_put(jnp.zeros((8,), dtype=jnp.float32), make_sharding((LogicalAxis.MIXTURE,)))

    result = add_one(values, reference)

    assert isinstance(result, ShapeDtypeStruct)
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding.mesh == fake_mesh
    assert tuple(result.sharding.spec) == (LogicalAxis.MIXTURE,)
