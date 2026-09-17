import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.s_direction import SDirectionMatrix, SDirectionSpec
from lalamo.initializer import RandomInitializer
from lalamo.models.language_model import LanguageModel
from lalamo.module import Keychain
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.safetensors import safe_write
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of, with_sharding
from lalamo.weight_matrix import Layout, ShapeDtypeSpec
from tests.conftest import RunLalamo
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.fixture(params=list(Layout))
def matrix(request: pytest.FixtureRequest) -> SDirectionMatrix:
    with np.load(Path(__file__).parent / "data/s_direction_muse.npz") as data:
        return SDirectionMatrix(
            spec=SDirectionSpec(request.param),
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data["codes"]),
            levels=jnp.asarray(data["levels"]),
            unit_scale=jnp.asarray(data["unit_scale"]),
            mean_norm=jnp.asarray(data["mean_norm"]),
            tail=jax.lax.bitcast_convert_type(jnp.asarray(data["tail_bits"]), jnp.bfloat16),
        ).switch_sharding_config(make_test_sharding_config())


def test_s_direction_matches_producer_and_native_reload(matrix: SDirectionMatrix) -> None:
    with np.load(Path(__file__).parent / "data/s_direction_muse.npz") as data:
        expected = data["expected_bits"]
        if matrix.spec.layout == Layout.INPUT_OUTPUT:
            expected = expected.T
    actual = jax.jit(lambda m: m.decompress())(matrix)
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(actual, jnp.uint16), expected)
    np.testing.assert_array_equal(matrix.astype(jnp.float32).decompress(), actual.astype(jnp.float32))
    template = ShapeDtypeSpec(matrix.spec.layout).compress(
        dummy_array(matrix.decompress().shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    assert isinstance(restored, SDirectionMatrix)
    assert restored.dtype == jnp.bfloat16
    for name, value in matrix.export().arrays.items():
        loaded = restored.export().arrays[name]
        assert loaded.dtype == value.dtype
        np.testing.assert_array_equal(loaded, value)
    np.testing.assert_array_equal(restored.decompress(), actual)


def test_s_direction_lookup_and_transposed_dot(matrix: SDirectionMatrix) -> None:
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    if matrix.spec.layout == Layout.INPUT_OUTPUT:
        rows = jnp.array([5, 0, 3], dtype=jnp.int32)
        with np.load(Path(__file__).parent / "data/s_direction_muse.npz") as data:
            expected = (data["expected_bits"][[5, 0, 3]].astype(np.uint32) << 16).view(np.float32)
        actual = jax.jit(lambda m: m.lookup_embedding(rows, keychain=keychain, dtype=jnp.float32))(matrix)
        np.testing.assert_array_equal(actual, expected)
    else:
        with pytest.raises(ValueError, match="input-output"):
            matrix.lookup_embedding(0, keychain=keychain)
    weights = matrix.decompress().astype(jnp.float32)
    vector = jnp.linspace(-1, 1, weights.shape[0], dtype=jnp.float32)
    np.testing.assert_allclose(
        matrix.dot(vector, transposed=True, keychain=keychain), weights.T @ vector, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("matrix", [Layout.OUTPUT_INPUT], indirect=True)
@pytest.mark.parametrize("mode", ["fully_sharded_data_parallel", "tensor_parallel", "data_parallel"])
def test_s_direction_batched_dot_and_sharding(matrix: SDirectionMatrix, mode: str) -> None:
    config = getattr(ShardingConfig, mode)(jax.devices("cpu")[:4])
    with jax.set_mesh(config.mesh):
        matrix = replace(
            matrix,
            codes=jnp.asarray(np.tile(np.asarray(matrix.codes), (20, 1))),
            tail=jnp.asarray(np.tile(np.asarray(matrix.tail), (20, 1))),
        ).switch_sharding_config(config)
        inputs = with_sharding(
            jnp.linspace(-1, 1, 4 * matrix.shape[1], dtype=jnp.float32).reshape(4, -1),
            config.resolve_sharding((LogicalAxis.BATCH, None)),
        )
        keychain = Keychain.init(0, sharding_config=config)
        actual = jax.jit(jax.vmap(lambda x: matrix.dot(x, keychain=keychain)))(inputs)
        expected = np.asarray(inputs, dtype=np.float64) @ np.asarray(matrix.decompress(), dtype=np.float64).T
        # FP32 reductions over 6656 columns differ across sharding strategies.
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=2e-5)
        assert sharding_of(actual).spec == sharding_of(inputs).spec


@pytest.mark.parametrize("matrix", [Layout.INPUT_OUTPUT], indirect=True)
def test_s_direction_checkpoint_chat_and_native_resave(
    matrix: SDirectionMatrix, model: LanguageModel, tmp_path: Path, run_lalamo: RunLalamo
) -> None:
    decoder = model.config.decoder_config
    config = replace(
        model.config,
        decoder_config=replace(
            decoder,
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            transformer_config=replace(decoder.transformer_config, model_dim=matrix.shape[1]),
        ),
    )
    with jax.set_mesh(model.sharding_config.mesh):
        model = config.init(
            model.token_codec.tokenizer,
            RandomInitializer(jnp.bfloat16, model.sharding_config, key=jax.random.key(12)),
        )
        matrix = replace(
            matrix,
            codes=jnp.asarray(np.tile(np.asarray(matrix.codes), (4, 1))),
            tail=jnp.asarray(np.tile(np.asarray(matrix.tail), (4, 1))),
        ).switch_sharding_config(model.sharding_config)
        exported = model.export()
        arrays, metadata = dict(exported.arrays), dict(exported.metadata)
        for name, layout in (("input_embedding", Layout.INPUT_OUTPUT), ("output_embedding", Layout.OUTPUT_INPUT)):
            prefix = f"decoder.embedding.{name}."
            arrays.pop(prefix + "weights")
            arrays.update({prefix + key: value for key, value in matrix.export().arrays.items()})
            metadata[prefix + "spec"] = SDirectionSpec(layout).to_json()
        legacy = config.to_json()
        assert isinstance(legacy, dict) and isinstance(legacy["decoder_config"], dict)
        legacy["decoder_config"]["pard_token"] = None
        (tmp_path / "config.json").write_text(json.dumps(legacy))
        model.token_codec.tokenizer.save(str(tmp_path / "tokenizer.json"))
        with (tmp_path / "model.safetensors").open("wb") as stream:
            safe_write(stream, arrays, metadata={name: json.dumps(value) for name, value in metadata.items()})
        restored = LanguageModel.load(tmp_path, model.sharding_config)
        restored.save(tmp_path / "native")
    with jax.set_mesh(None):
        output = run_lalamo("chat", str(tmp_path), "--message", "token1", "--max-tokens", "2", "--temperature", "0")
        native = run_lalamo(
            "chat", str(tmp_path / "native"), "--message", "token1", "--max-tokens", "2", "--temperature", "0"
        )
    assert "token" in output
    assert output == native
