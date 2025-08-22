from typing import Any

import jax.numpy as jnp
import pytest

from lalamo.common import unflatten_parameters
from lalamo.utils import MapDictValues, MapSequence


def test_unflatten_simple_dict() -> None:
    flat = {"a": jnp.array([1]), "b": jnp.array([2])}
    result = unflatten_parameters(flat)
    assert isinstance(result, MapDictValues)
    assert "a" in result and "b" in result
    assert jnp.array_equal(result["a"], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result["b"], jnp.array([2]))  # type: ignore


def test_unflatten_nested_dict() -> None:
    flat = {
        "layer.weight": jnp.array([1]),
        "layer.bias": jnp.array([2]),
    }
    result = unflatten_parameters(flat)
    assert isinstance(result, MapDictValues)
    assert isinstance(result["layer"], MapDictValues)  # type: ignore
    assert jnp.array_equal(result["layer"]["weight"], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result["layer"]["bias"], jnp.array([2]))  # type: ignore


def test_unflatten_to_list() -> None:
    flat = {"0": jnp.array([1]), "1": jnp.array([2])}
    result = unflatten_parameters(flat)
    assert isinstance(result, MapSequence)
    assert len(result) == 2
    assert jnp.array_equal(result[0], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result[1], jnp.array([2]))  # type: ignore


def test_unflatten_mixed_nested() -> None:
    flat = {
        "layers.0.weight": jnp.array([1]),
        "layers.0.bias": jnp.array([2]),
        "layers.1.weight": jnp.array([3]),
        "layers.1.bias": jnp.array([4]),
    }
    result = unflatten_parameters(flat)
    assert isinstance(result, MapDictValues)
    assert isinstance(result["layers"], MapSequence)  # type: ignore
    assert len(result["layers"]) == 2  # type: ignore
    assert isinstance(result["layers"][0], MapDictValues)  # type: ignore
    assert isinstance(result["layers"][1], MapDictValues)  # type: ignore
    assert jnp.array_equal(result["layers"][0]["weight"], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result["layers"][0]["bias"], jnp.array([2]))  # type: ignore
    assert jnp.array_equal(result["layers"][1]["weight"], jnp.array([3]))  # type: ignore
    assert jnp.array_equal(result["layers"][1]["bias"], jnp.array([4]))  # type: ignore


def test_unflatten_empty_dict() -> None:
    flat: dict[str, Any] = {}
    result = unflatten_parameters(flat)
    assert isinstance(result, MapDictValues)
    assert len(result) == 0


def test_unflatten_single_level_numeric() -> None:
    flat = {
        "0.weight": jnp.array([1]),
        "1.weight": jnp.array([2]),
    }
    result = unflatten_parameters(flat)
    assert isinstance(result, MapSequence)
    assert len(result) == 2
    assert isinstance(result[0], MapDictValues)
    assert isinstance(result[1], MapDictValues)
    assert jnp.array_equal(result[0]["weight"], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result[1]["weight"], jnp.array([2]))  # type: ignore


def test_unflatten_deep_nesting() -> None:
    flat = {
        "model.layers.0.attention.weight": jnp.array([1]),
        "model.layers.0.attention.bias": jnp.array([2]),
        "model.layers.1.attention.weight": jnp.array([3]),
        "model.layers.1.attention.bias": jnp.array([4]),
    }
    result = unflatten_parameters(flat)
    assert isinstance(result, MapDictValues)
    assert isinstance(result["model"], MapDictValues)  # type: ignore
    assert isinstance(result["model"]["layers"], MapSequence)  # type: ignore
    assert len(result["model"]["layers"]) == 2  # type: ignore
    assert isinstance(result["model"]["layers"][0]["attention"], MapDictValues)  # type: ignore
    assert isinstance(result["model"]["layers"][1]["attention"], MapDictValues)  # type: ignore
    assert jnp.array_equal(result["model"]["layers"][0]["attention"]["weight"], jnp.array([1]))  # type: ignore
    assert jnp.array_equal(result["model"]["layers"][0]["attention"]["bias"], jnp.array([2]))  # type: ignore
    assert jnp.array_equal(result["model"]["layers"][1]["attention"]["weight"], jnp.array([3]))  # type: ignore
    assert jnp.array_equal(result["model"]["layers"][1]["attention"]["bias"], jnp.array([4]))  # type: ignore


def test_unflatten_invalid_numeric_sequence() -> None:
    flat = {"0": jnp.array([1]), "2": jnp.array([2])}  # Missing "1"
    with pytest.raises(AssertionError):
        unflatten_parameters(flat)
