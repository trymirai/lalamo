import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.row_stack import RowStackMatrix
from lalamo.compressed.s_surface import SSurfaceKind, SSurfaceMatrix
from lalamo.compressed.s_trellis import STrellisMatrix
from lalamo.model_import.loaders.s_checkpoint import load_s_checkpoint
from lalamo.models.language_model import LanguageModel
from lalamo.module import Keychain
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.safetensors import safe_read
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import Layout, WeightMatrix

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        "S_CHECKPOINT_DIRECTORY" not in os.environ, reason="requires the original physical HYB036 S checkpoint"
    ),
]


@pytest.fixture(scope="module")
def model() -> Iterator[LanguageModel]:
    config = ShardingConfig.replicated()
    with jax.set_mesh(config.mesh):
        yield load_s_checkpoint(Path(os.environ["S_CHECKPOINT_DIRECTORY"]), config)


def test_s_checkpoint_preserves_every_packed_weight(model: LanguageModel) -> None:
    path = Path(os.environ["S_CHECKPOINT_DIRECTORY"]) / "model.safetensors"
    matrices = []
    for jax_path, leaf in jax.tree_util.tree_leaves_with_path(model, is_leaf=lambda x: isinstance(x, WeightMatrix)):
        prefix = ParameterPath() / jax_path
        if isinstance(leaf, RowStackMatrix):
            parent = prefix.removesuffix("qkvg_projection.weights")
            matrices.extend(
                (ParameterPath(parent + name + ".weights"), part)
                for name, part in zip(("qkv_projection", "gate_projection"), leaf.parts, strict=True)
            )
        elif isinstance(leaf, WeightMatrix):
            matrices.append((prefix, leaf))
    assert Counter(type(matrix).__name__ for _, matrix in matrices) == {"STrellisMatrix": 272, "SSurfaceMatrix": 2}
    with path.open("rb") as stream:
        _, saved = safe_read(stream)
        for prefix, matrix in matrices:
            assert matrix.dtype == jnp.bfloat16
            if isinstance(matrix, STrellisMatrix):
                parameters = {"codes": matrix.codes, "scales": matrix.scales, "gains": matrix.gains}
                shared = {
                    f"codebook_v{matrix.spec.vector_width}": matrix.table,
                    f"signs_{matrix.shape[1]}": matrix.signs,
                    f"q_{matrix.shape[1]}": matrix.small_q,
                }
                for key, value in shared.items():
                    np.testing.assert_array_equal(value, saved[f"qtip_shared.{key}"])
            else:
                assert isinstance(matrix, SSurfaceMatrix)
                sign_name = (
                    "output_hadamard_factors"
                    if matrix.spec.layout == Layout.INPUT_OUTPUT
                    else "input_hadamard_factors"
                )
                parameters = {
                    "codes": matrix.codes,
                    "row_scales": matrix.row_scales,
                    "ladder_indices": matrix.ladder_indices,
                    "ladder": matrix.ladder,
                    sign_name: matrix.signs,
                }
                if matrix.spec.kind == SSurfaceKind.D4:
                    parameters["table"] = matrix.table
            for key, value in parameters.items():
                original = saved[prefix / key]
                assert value.dtype == original.dtype
                np.testing.assert_array_equal(value, original)


def test_s_checkpoint_runs_a_complete_forward_pass(model: LanguageModel) -> None:
    tokens = jnp.array([[1]], dtype=jnp.int32)
    state = model.decoder.init_static_state(batch_size=1, capacity=32, dtype=jnp.bfloat16)
    result = model.decoder(
        tokens,
        jnp.zeros_like(tokens),
        state=state,
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
    )
    assert result.logits.shape == (1, 1, 248320)
    assert bool(jnp.all(jnp.isfinite(result.logits)))
