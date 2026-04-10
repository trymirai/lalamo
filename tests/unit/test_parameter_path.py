import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from lalamo.common import ParameterPath


@pytest.mark.fast
def test_empty_path_is_empty_string() -> None:
    assert ParameterPath("") == ""
    assert not ParameterPath("")


@pytest.mark.fast
def test_path_is_str_subclass() -> None:
    p = ParameterPath("a.b.c")
    assert isinstance(p, str)
    assert isinstance(p, ParameterPath)


@pytest.mark.fast
def test_div_str_from_empty() -> None:
    assert ParameterPath("") / "weight" == "weight"


@pytest.mark.fast
def test_div_str_appends_with_dot() -> None:
    assert ParameterPath("layer") / "weight" == "layer.weight"


@pytest.mark.fast
def test_div_str_compound_allowed() -> None:
    """Compound strings (containing dots) are allowed — needed for path composition."""
    assert ParameterPath("a") / "b.c" == "a.b.c"


@pytest.mark.fast
def test_div_str_returns_parameter_path() -> None:
    result = ParameterPath("a") / "b"
    assert isinstance(result, ParameterPath)


@pytest.mark.fast
def test_div_int_from_empty() -> None:
    assert ParameterPath("") / 0 == "0"


@pytest.mark.fast
def test_div_int_appends() -> None:
    assert ParameterPath("layers") / 3 == "layers.3"


@pytest.mark.fast
def test_div_get_attr_key() -> None:
    assert ParameterPath("") / jtu.GetAttrKey("weight") == "weight"
    assert ParameterPath("layer") / jtu.GetAttrKey("bias") == "layer.bias"


@pytest.mark.fast
def test_div_sequence_key() -> None:
    assert ParameterPath("") / jtu.SequenceKey(0) == "0"
    assert ParameterPath("layers") / jtu.SequenceKey(5) == "layers.5"


@pytest.mark.fast
def test_div_dict_key() -> None:
    assert ParameterPath("") / jtu.DictKey("name") == "name"
    assert ParameterPath("config") / jtu.DictKey("lr") == "config.lr"


@pytest.mark.fast
def test_div_dict_key_with_dots_raises() -> None:
    with pytest.raises(ValueError, match="contains dots"):
        ParameterPath("") / jtu.DictKey("bad.key")


@pytest.mark.fast
def test_div_dict_key_with_int_key() -> None:
    assert ParameterPath("layers") / jtu.DictKey(0) == "layers.0"


@pytest.mark.fast
def test_div_flattened_index_key() -> None:
    assert ParameterPath("") / jtu.FlattenedIndexKey(7) == "7"
    assert ParameterPath("flat") / jtu.FlattenedIndexKey(0) == "flat.0"


@pytest.mark.fast
def test_div_tuple_single_element() -> None:
    path = (jtu.GetAttrKey("weight"),)
    assert ParameterPath("") / path == "weight"


@pytest.mark.fast
def test_div_tuple_mixed_keys() -> None:
    path = (jtu.GetAttrKey("layers"), jtu.SequenceKey(2), jtu.GetAttrKey("weight"))
    assert ParameterPath("") / path == "layers.2.weight"


@pytest.mark.fast
def test_div_tuple_with_prefix() -> None:
    path = (jtu.GetAttrKey("attention"), jtu.GetAttrKey("q_proj"))
    assert ParameterPath("model.layers.0") / path == "model.layers.0.attention.q_proj"


@pytest.mark.fast
def test_div_empty_tuple() -> None:
    assert ParameterPath("prefix") / () == "prefix"
    assert ParameterPath("") / () == ""


@pytest.mark.fast
def test_chaining_multiple_str() -> None:
    result = ParameterPath("") / "model" / "layers" / "0" / "weight"
    assert result == "model.layers.0.weight"


@pytest.mark.fast
def test_chaining_mixed_types() -> None:
    result = ParameterPath("") / "layers" / 0 / jtu.GetAttrKey("weight")
    assert result == "layers.0.weight"


@pytest.mark.fast
def test_div_parameter_path() -> None:
    """ParameterPath is a str subclass, so composing two paths should work via str dispatch."""
    a = ParameterPath("model.layers")
    b = ParameterPath("0.weight")
    assert a / b == "model.layers.0.weight"
    assert isinstance(a / b, ParameterPath)


@pytest.mark.fast
def test_invariant_simple_module() -> None:
    """For any equinox module, ParameterPath("") / path should produce correct flat keys."""

    class Inner(eqx.Module):
        weight: jax.Array
        bias: jax.Array

    class Outer(eqx.Module):
        inner: Inner
        scale: jax.Array

    module = Outer(
        inner=Inner(weight=jnp.ones(3), bias=jnp.zeros(3)),
        scale=jnp.array(1.0),
    )

    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(module)
    paths = [str(ParameterPath("") / path) for path, _ in flat_with_path]

    assert set(paths) == {"inner.weight", "inner.bias", "scale"}


@pytest.mark.fast
def test_invariant_module_with_list() -> None:
    class Model(eqx.Module):
        layers: list[jax.Array]

    module = Model(layers=[jnp.ones(2), jnp.zeros(2)])

    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(module)
    paths = [str(ParameterPath("") / path) for path, _ in flat_with_path]

    assert paths == ["layers.0", "layers.1"]


@pytest.mark.fast
def test_invariant_nested_module() -> None:
    class Layer(eqx.Module):
        w: jax.Array

    class Block(eqx.Module):
        layers: list[Layer]

    class Model(eqx.Module):
        block: Block
        head: jax.Array

    module = Model(
        block=Block(layers=[Layer(w=jnp.ones(2)), Layer(w=jnp.zeros(2))]),
        head=jnp.array(1.0),
    )

    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(module)
    paths = [str(ParameterPath("") / path) for path, _ in flat_with_path]

    assert set(paths) == {"block.layers.0.w", "block.layers.1.w", "head"}


@pytest.mark.fast
def test_invariant_dict_pytree() -> None:
    tree = {"model": {"layers": [jnp.ones(2), jnp.zeros(2)], "bias": jnp.array(1.0)}}

    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths = [str(ParameterPath("") / path) for path, _ in flat_with_path]

    assert set(paths) == {"model.bias", "model.layers.0", "model.layers.1"}


@pytest.mark.fast
def test_invariant_prefix_composition() -> None:
    """Prefixed path should equal manual string concatenation."""

    class Module(eqx.Module):
        weight: jax.Array

    module = Module(weight=jnp.ones(3))
    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(module)

    prefix = ParameterPath("decoder.layers.0")
    paths = [str(prefix / path) for path, _ in flat_with_path]

    assert paths == ["decoder.layers.0.weight"]
