from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import Keychain, ShardingAxis
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig, UntiedEmbedding, UntiedEmbeddingConfig
from lalamo.modules.utils import apply_soft_capping, call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import make_sharding
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, Layout
from tests.common import assert_close

MODEL_DIM = 4
VOCAB_SIZE = 4


def _weights(*, offset: int = 0) -> jax.Array:
    shape = (VOCAB_SIZE, MODEL_DIM)
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _input_embedding_matrix(*, offset: int = 0) -> FullPrecisionMatrix:
    return FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(_weights(offset=offset))


def _output_embedding_matrix(*, offset: int = 100) -> FullPrecisionMatrix:
    return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(_weights(offset=offset))


def _tied_embedding() -> TiedEmbedding:
    return TiedEmbedding(
        config=TiedEmbeddingConfig(input_scale=1.5, logit_soft_cap=2.0),
        embedding=_input_embedding_matrix(),
    )


def _untied_embedding() -> UntiedEmbedding:
    return UntiedEmbedding(
        config=UntiedEmbeddingConfig(input_scale=1.5, logit_soft_cap=2.0),
        input_embedding=_input_embedding_matrix(),
        output_embedding=_output_embedding_matrix(),
    )


def _embedding(tied: bool) -> TiedEmbedding | UntiedEmbedding:
    if tied:
        return _tied_embedding()
    return _untied_embedding()


def _embed_reference(module: TiedEmbedding | UntiedEmbedding, token_id: Array | int) -> Array:
    result = module.embedding_matrix.lookup_embedding(token_id, keychain=Keychain.init(10))
    if module.config.input_scale is not None:
        result = result * jnp.array(module.config.input_scale, dtype=result.dtype)
    return result


def _readout_reference(module: TiedEmbedding | UntiedEmbedding, inputs: Array) -> Array:
    logits = module.readout_matrix.dot(inputs, keychain=Keychain.init(11))
    if module.config.logit_soft_cap is not None:
        logits = apply_soft_capping(logits, module.config.logit_soft_cap)
    return logits


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((ShardingAxis.DATA,)))


def _sharded_vectors(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None, ShardingAxis.DATA)))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_matches_reference_under_jit(fake_mesh: Mesh, tied: bool) -> None:
    module = _embedding(tied)
    token_id = jnp.array(2, dtype=jnp.int32)

    result = eqx.filter_jit(lambda module, token_id: module.embed(token_id, keychain=Keychain.init(0)))(
        module,
        token_id,
    )

    _assert_close(result=result, reference=_embed_reference(module, token_id))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_vmapped_over_tokens_preserves_token_and_feature_sharding(
    fake_mesh: Mesh,
    tied: bool,
) -> None:
    module = _embedding(tied)
    token_ids = jax.device_put(jnp.array([0, 2], dtype=jnp.int32), make_sharding((ShardingAxis.DATA,)))

    result = jax.vmap(lambda token_id: module.embed(token_id, keychain=Keychain.init(1)))(token_ids)
    reference = jax.vmap(lambda token_id: _embed_reference(module, token_id))(token_ids)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_call_vmapped_over_tokens_preserves_token_and_feature_sharding(
    fake_mesh: Mesh,
    tied: bool,
) -> None:
    module = _embedding(tied)
    token_ids = jax.device_put(jnp.array([1, 3], dtype=jnp.int32), make_sharding((ShardingAxis.DATA,)))

    result = call_vmapped(
        module.embed,
        token_ids,
        keychain=Keychain.init(1),
        added_sharding_axis=ShardingAxis.DATA,
    )
    reference = jax.vmap(lambda token_id: _embed_reference(module, token_id))(token_ids)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((ShardingAxis.DATA, None))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_readout_matches_reference_under_jit_and_preserves_input_sharding(
    fake_mesh: Mesh,
    tied: bool,
) -> None:
    module = _embedding(tied)
    inputs = _sharded_vector(jnp.array([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float32))

    result = eqx.filter_jit(lambda module, values: module.readout(values, keychain=Keychain.init(2)))(module, inputs)

    _assert_close(result=result, reference=_readout_reference(module, inputs))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_readout_vmapped_over_inputs_preserves_input_sharding(fake_mesh: Mesh, tied: bool) -> None:
    module = _embedding(tied)
    inputs = _sharded_vectors(jnp.arange(2 * MODEL_DIM, dtype=jnp.float32).reshape(2, MODEL_DIM) / 10)

    result = jax.vmap(lambda values: module.readout(values, keychain=Keychain.init(3)))(inputs)
    reference = jax.vmap(lambda values: _readout_reference(module, values))(inputs)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


def test_tied_embedding_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _tied_embedding()
    template = TiedEmbedding(
        config=original.config,
        embedding=FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(
            dummy_array(_weights().shape, jnp.float32),
        ),
    )
    inputs = _sharded_vector(jnp.array([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float32))

    restored = template.load_exported(original.export())
    result = restored.readout(inputs, keychain=Keychain.init(4))

    assert isinstance(restored.embedding, FullPrecisionMatrix)
    assert isinstance(template.embedding, FullPrecisionMatrix)
    assert restored.embedding.weights.sharding == template.embedding.weights.sharding
    _assert_named_sharding(restored.embedding.weights.sharding, fake_mesh)
    _assert_close(result=result, reference=original.readout(inputs, keychain=Keychain.init(5)))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


def test_untied_embedding_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _untied_embedding()
    template = UntiedEmbedding(
        config=original.config,
        input_embedding=FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(
            dummy_array(_weights().shape, jnp.float32),
        ),
        output_embedding=FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(
            dummy_array(_weights(offset=100).shape, jnp.float32),
        ),
    )
    inputs = _sharded_vector(jnp.array([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float32))

    restored = template.load_exported(original.export())
    result = restored.readout(inputs, keychain=Keychain.init(6))

    assert isinstance(restored.input_embedding, FullPrecisionMatrix)
    assert isinstance(restored.output_embedding, FullPrecisionMatrix)
    assert isinstance(template.input_embedding, FullPrecisionMatrix)
    assert isinstance(template.output_embedding, FullPrecisionMatrix)
    assert restored.input_embedding.weights.sharding == template.input_embedding.weights.sharding
    assert restored.output_embedding.weights.sharding == template.output_embedding.weights.sharding
    _assert_named_sharding(restored.input_embedding.weights.sharding, fake_mesh)
    _assert_named_sharding(restored.output_embedding.weights.sharding, fake_mesh)
    _assert_close(result=result, reference=original.readout(inputs, keychain=Keychain.init(7)))
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding
