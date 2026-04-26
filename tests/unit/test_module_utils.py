from typing import Any

import jax
import jax.numpy as jnp
import pytest

from lalamo.module import Keychain
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from tests.common import assert_close


def test_call_vmapped_matches_jax_vmap_with_multiple_inputs_and_custom_axes() -> None:
    left = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    right = jnp.arange(24, 48, dtype=jnp.float32).reshape(3, 2, 4)
    bias = jnp.arange(4, dtype=jnp.float32)

    def combine(left_slice: jax.Array, right_slice: jax.Array, bias_vector: jax.Array) -> jax.Array:
        return left_slice + 2.0 * right_slice + bias_vector

    expected = jax.vmap(combine, in_axes=(1, 0, None), out_axes=-1)(left, right, bias)
    result = call_vmapped(combine, left, right, bias, in_axes=(1, 0, None), out_axes=-1)

    assert_close(result=result, reference=expected)


def test_call_vmapped_preserves_single_tuple_input_compatibility() -> None:
    def combine_pair(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        left, right = pair
        return left - right

    inputs = (
        jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        jnp.arange(12, 24, dtype=jnp.float32).reshape(3, 4),
    )

    expected = jax.vmap(combine_pair, in_axes=((0, 0),), out_axes=0)(inputs)
    result = call_vmapped(combine_pair, inputs, in_axes=((0, 0),), out_axes=0)

    assert_close(result=result, reference=expected)


def test_call_vmapped_supports_multi_arg_pytrees_and_unmapped_inputs() -> None:
    def project(left: jax.Array, metadata: dict[str, jax.Array], scale: jax.Array) -> tuple[jax.Array, jax.Array]:
        right = metadata["right"]
        bias = metadata["bias"]
        mapped = left + right + bias * scale
        repeated_scalar = jnp.asarray((bias * scale).sum(), dtype=left.dtype)
        return mapped, repeated_scalar

    left = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    metadata = {
        "right": jnp.arange(10, dtype=jnp.float32).reshape(2, 5),
        "bias": jnp.array(3.0, dtype=jnp.float32),
    }
    scale = jnp.array(2.0, dtype=jnp.float32)

    expected = jax.vmap(
        project,
        in_axes=(0, {"right": 1, "bias": None}, None),
        out_axes=(0, 0),
    )(left, metadata, scale)
    result = call_vmapped(
        project,
        left,
        metadata,
        scale,
        in_axes=(0, {"right": 1, "bias": None}, None),
        out_axes=(0, 0),
    )

    assert_close(result=result[0], reference=expected[0])
    assert_close(result=result[1], reference=expected[1])


def test_call_vmapped_preserves_forward_pass_config_and_vmaps_keychain() -> None:
    left = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4)
    right = jnp.arange(24, 48, dtype=jnp.float32).reshape(3, 2, 4)
    forward_pass_config = 3
    base_keychain = Keychain.init(0)
    vmapped_keys = base_keychain.broadcast((3,)).vmapped_keys
    batch_key = base_keychain.batch_key

    def transform(
        left_slice: jax.Array,
        right_slice: jax.Array,
        *,
        forward_pass_config: int,
        keychain: Keychain,
    ) -> jax.Array:
        token_bias = jax.random.uniform(keychain.vmapped_keys, shape=(), dtype=left_slice.dtype)
        batch_bias = jnp.asarray(jax.random.key_data(keychain.batch_key).sum(), dtype=left_slice.dtype)
        return left_slice + right_slice + forward_pass_config + token_bias + batch_bias

    expected = jax.vmap(
        lambda left_slice, right_slice, vmapped_keys: transform(
            left_slice,
            right_slice,
            forward_pass_config=forward_pass_config,
            keychain=Keychain(vmapped_keys=vmapped_keys, batch_key=batch_key),
        ),
        in_axes=(0, 0, 0),
    )(
        left,
        right,
        vmapped_keys,
    )
    result = call_vmapped(
        transform,
        left,
        right,
        forward_pass_config=forward_pass_config,
        keychain=base_keychain,
    )

    assert_close(result=result, reference=expected)


def test_call_vmapped_twice_vmaps_keychain_over_both_batch_dimensions() -> None:
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    base_keychain = Keychain.init(2)
    vmapped_keys = base_keychain.broadcast((2, 3)).vmapped_keys
    batch_key = base_keychain.batch_key

    def add_keychain_bias(x: jax.Array, *, keychain: Keychain) -> jax.Array:
        return x + jax.random.uniform(keychain.vmapped_keys, shape=(), dtype=x.dtype)

    expected = jax.vmap(
        jax.vmap(
            lambda x, vmapped_keys: add_keychain_bias(
                x,
                keychain=Keychain(vmapped_keys=vmapped_keys, batch_key=batch_key),
            ),
        ),
    )(
        inputs,
        vmapped_keys,
    )
    result = call_vmapped_twice(
        add_keychain_bias,
        inputs,
        keychain=base_keychain,
    )

    assert_close(result=result, reference=expected)


def test_call_vmapped_twice_matches_nested_vmap_with_distinct_axes_for_multiple_inputs() -> None:
    left = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    right = jnp.arange(24, 48, dtype=jnp.float32).reshape(3, 2, 4)

    def combine_rows(left_row: jax.Array, right_row: jax.Array) -> jax.Array:
        return left_row - right_row

    expected = jax.vmap(
        jax.vmap(combine_rows, in_axes=(0, 0), out_axes=-1),
        in_axes=(1, 0),
        out_axes=0,
    )(left, right)
    result = call_vmapped_twice(
        combine_rows,
        left,
        right,
        in_axes=((1, 0), (0, 0)),
        out_axes=(0, -1),
    )

    assert_close(result=result, reference=expected)


def test_call_vmapped_twice_supports_multi_arg_pytree_axis_pairs() -> None:
    def add_tree(left_tree: dict[str, jax.Array], right_tree: dict[str, jax.Array]) -> jax.Array:
        return left_tree["lhs"] + right_tree["rhs"] + left_tree["bias"]

    left_tree = {
        "lhs": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "bias": jnp.array(1.0, dtype=jnp.float32),
    }
    right_tree = {"rhs": jnp.arange(6, dtype=jnp.float32).reshape(3, 2)}

    expected = jax.vmap(
        jax.vmap(
            add_tree,
            in_axes=({"lhs": 0, "bias": None}, {"rhs": 0}),
            out_axes=0,
        ),
        in_axes=({"lhs": 1, "bias": None}, {"rhs": 0}),
        out_axes=1,
    )(left_tree, right_tree)
    result = call_vmapped_twice(
        add_tree,
        left_tree,
        right_tree,
        in_axes=(
            ({"lhs": 1, "bias": None}, {"rhs": 0}),
            ({"lhs": 0, "bias": None}, {"rhs": 0}),
        ),
        out_axes=(1, 0),
    )

    assert_close(result=result, reference=expected)


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("in_axes", ((0, 0),)),
        ("out_axes", (0, 0, 0)),
    ],
)
def test_call_vmapped_twice_rejects_malformed_axis_pairs(
    argument_name: str,
    argument_value: tuple[Any, ...],
) -> None:
    kwargs = {argument_name: argument_value}

    with pytest.raises(ValueError, match="must contain exactly two axis specs"):
        call_vmapped_twice(
            lambda left, right: left + right,
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            jnp.arange(6, 12, dtype=jnp.float32).reshape(2, 3),
            **kwargs,
        )
