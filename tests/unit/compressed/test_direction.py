from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.direction import DirectionMatrix, DirectionSpec
from lalamo.module import Keychain
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def test_direction_rows_match_the_producer_through_decode_lookup_and_dot() -> None:
    # Eight Muse vocabulary rows packed by the independent producer, with the bfloat16 weights it decoded them to.
    with np.load(Path(__file__).parent / "data/direction_muse.npz") as data:
        vocabulary = DirectionMatrix(
            spec=DirectionSpec(Layout.INPUT_OUTPUT),
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data["codes"]),
            levels=jnp.asarray(data["levels"]),
            unit_scale=jnp.asarray(data["unit_scale"]),
            mean_norm=jnp.asarray(data["mean_norm"]),
            tail=jnp.asarray(data["tail_bits"].view(jnp.bfloat16)),
        )
        expected = data["expected_bits"].view(jnp.bfloat16).astype(np.float32)
    readout = replace(vocabulary, spec=DirectionSpec(Layout.OUTPUT_INPUT))
    keychain = Keychain.init(0, sharding_config=vocabulary.sharding_config)
    rows = jnp.array([5, 0, 3], dtype=jnp.int32)
    vector = jnp.linspace(-1, 1, expected.shape[1], dtype=jnp.float32)

    np.testing.assert_array_equal(jax.jit(lambda m: m.decompress())(readout).astype(jnp.float32), expected)
    np.testing.assert_array_equal(
        jax.jit(lambda m: m.lookup_embedding(rows, keychain=keychain, dtype=jnp.float32))(vocabulary), expected[rows]
    )
    np.testing.assert_allclose(readout.dot(vector, keychain=keychain), expected @ np.asarray(vector), atol=1e-4)
