import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jaxtyping import TypeCheckError

from lalamo.utils.parameter_path import ParameterPath


@pytest.mark.parametrize(
    ("prefix", "key", "expected"),
    [
        ("", "weight", "weight"),
        ("decoder", "weight", "decoder.weight"),
        ("layers", 3, "layers.3"),
        ("", jtu.GetAttrKey("bias"), "bias"),
        ("layers", jtu.SequenceKey(2), "layers.2"),
        ("config", jtu.DictKey("lr"), "config.lr"),
        ("flat", jtu.FlattenedIndexKey(5), "flat.5"),
        (
            "decoder",
            (jtu.GetAttrKey("layers"), jtu.SequenceKey(0), jtu.GetAttrKey("weight")),
            "decoder.layers.0.weight",
        ),
    ],
)
def test_parameter_path_appends_supported_keys(
    prefix: str,
    key: object,
    expected: str,
) -> None:
    result = ParameterPath(prefix) / key

    assert result == expected
    assert isinstance(result, ParameterPath)


def test_parameter_path_rejects_ambiguous_dict_keys() -> None:
    with pytest.raises(ValueError, match="contains dots"):
        ParameterPath("model") / jtu.DictKey("bad.key")


def test_parameter_path_rejects_unknown_key_types() -> None:
    with pytest.raises((TypeCheckError, TypeError)):
        ParameterPath("model") / object()


def test_parameter_path_matches_flattened_equinox_module_paths() -> None:
    class Attention(eqx.Module):
        query: jax.Array
        key: jax.Array

    class Decoder(eqx.Module):
        attention: Attention
        layers: list[jax.Array]

    decoder = Decoder(
        attention=Attention(query=jnp.ones((2, 3)), key=jnp.zeros((2, 3))),
        layers=[jnp.ones((3,)), jnp.zeros((3,))],
    )

    leaves_with_paths, _tree_def = jtu.tree_flatten_with_path(decoder)
    paths = {str(ParameterPath("") / path) for path, _leaf in leaves_with_paths}

    assert paths == {
        "attention.query",
        "attention.key",
        "layers.0",
        "layers.1",
    }


def test_parameter_path_can_extend_existing_path_with_flattened_path_tuple() -> None:
    tree = {"weights": [jnp.ones((2,)), jnp.zeros((2,))], "bias": jnp.array(1)}

    leaves_with_paths, _tree_def = jtu.tree_flatten_with_path(tree)
    paths = {str(ParameterPath("decoder") / path) for path, _leaf in leaves_with_paths}

    assert paths == {
        "decoder.bias",
        "decoder.weights.0",
        "decoder.weights.1",
    }
