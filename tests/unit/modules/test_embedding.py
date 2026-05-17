from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.embedding import (
    EmbeddingForwardPassConfig,
    TiedEmbedding,
    TiedEmbeddingConfig,
    UntiedEmbedding,
    UntiedEmbeddingConfig,
)
from lalamo.modules.utils import apply_soft_capping, call_vmapped
from lalamo.utils.sharding import is_sharded
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, Layout, ShapeDtypeMatrix
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

MODEL_DIM = 6
VOCAB_SIZE = 10

ACTIVATION_DTYPES = [
    pytest.param(jnp.bfloat16, id="bf16"),
    pytest.param(jnp.float32, id="fp32"),
]

LOGIT_DTYPES = [
    pytest.param(jnp.float32, id="fp32"),
    pytest.param(jnp.bfloat16, id="bf16"),
]


def _weights(*, offset: int = 0) -> jax.Array:
    shape = (VOCAB_SIZE, MODEL_DIM)
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 10) - 1


def _input_embedding_matrix(*, offset: int = 0) -> FullPrecisionMatrix:
    return FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(
        jnp.matrix_transpose(_weights(offset=offset)),
        sharding_config=make_test_sharding_config().replicated_with_same_mesh(),
    )


def _output_embedding_matrix(*, offset: int = 100) -> FullPrecisionMatrix:
    return FullPrecisionSpec(layout=Layout.OUTPUT_INPUT).compress(
        _weights(offset=offset), sharding_config=make_test_sharding_config()
    )


def _tied_embedding() -> TiedEmbedding:
    return TiedEmbedding(
        config=TiedEmbeddingConfig(input_scale=1.5, logit_soft_cap=2.0),
        sharding_config=make_test_sharding_config(),
        embedding=_input_embedding_matrix(),
    )


def _untied_embedding() -> UntiedEmbedding:
    return UntiedEmbedding(
        config=UntiedEmbeddingConfig(input_scale=1.5, logit_soft_cap=2.0),
        sharding_config=make_test_sharding_config(),
        input_embedding=_input_embedding_matrix(),
        output_embedding=_output_embedding_matrix(),
    )


def _embedding(tied: bool) -> TiedEmbedding | UntiedEmbedding:
    if tied:
        return _tied_embedding()
    return _untied_embedding()


def _embed_reference(
    module: TiedEmbedding | UntiedEmbedding,
    token_id: Array | int,
    forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
) -> Array:
    result = module.embedding_matrix.lookup_embedding(
        token_id,
        dtype=forward_pass_config.activation_dtype,
        keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
        forward_pass_config=forward_pass_config.matmul_config,
    )
    if module.config.input_scale is not None:
        result = result * jnp.array(module.config.input_scale, dtype=result.dtype)
    return result


def _readout_reference(
    module: TiedEmbedding | UntiedEmbedding,
    inputs: Array,
    forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
) -> Array:
    if isinstance(module, TiedEmbedding):
        logits = module.embedding.dot(
            inputs,
            keychain=Keychain.init(11, sharding_config=make_test_sharding_config()),
            forward_pass_config=forward_pass_config.matmul_config,
            transposed=True,
        )
    else:
        logits = module.readout_matrix.dot(
            inputs,
            keychain=Keychain.init(11, sharding_config=make_test_sharding_config()),
            forward_pass_config=forward_pass_config.matmul_config,
        )
    logits = logits.astype(forward_pass_config.logit_dtype)
    if module.config.logit_soft_cap is not None:
        logits = apply_soft_capping(logits, module.config.logit_soft_cap)
    return logits


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _sharded_vector(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None,)))


def _sharded_vectors(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None)))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("activation_dtype", ACTIVATION_DTYPES)
def test_embedding_embed_matches_reference_under_jit(
    fake_mesh: Mesh,
    tied: bool,
    activation_dtype: DTypeLike,
) -> None:
    module = _embedding(tied)
    token_id = jnp.array(2, dtype=jnp.int32)
    forward_pass_config = EmbeddingForwardPassConfig(activation_dtype=activation_dtype)

    def call(module: TiedEmbedding | UntiedEmbedding, token_id: Array) -> Array:
        return module.embed(
            token_id,
            keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
            forward_pass_config=forward_pass_config,
        )

    result = eqx.filter_jit(call)(module, token_id)

    _assert_close(result=result, reference=_embed_reference(module, token_id, forward_pass_config))
    assert result.dtype == jnp.dtype(forward_pass_config.activation_dtype)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None,))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_vmapped_over_tokens_keeps_token_and_feature_axes_unsharded(
    fake_mesh: Mesh,
    tied: bool,
) -> None:
    module = _embedding(tied)
    token_ids = jnp.array([0, 2], dtype=jnp.int32)

    result = jax.vmap(
        lambda token_id: module.embed(token_id, keychain=Keychain.init(1, sharding_config=make_test_sharding_config()))
    )(token_ids)
    reference = jax.vmap(lambda token_id: _embed_reference(module, token_id))(token_ids)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None, None))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_call_vmapped_over_tokens_keeps_token_and_feature_axes_unsharded(
    fake_mesh: Mesh,
    tied: bool,
) -> None:
    module = _embedding(tied)
    token_ids = jnp.array([1, 3], dtype=jnp.int32)

    result = call_vmapped(
        module.embed,
        token_ids,
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    )
    reference = jax.vmap(lambda token_id: _embed_reference(module, token_id))(token_ids)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == make_sharding((None, None))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("logit_dtype", LOGIT_DTYPES)
def test_embedding_readout_matches_reference_under_jit_and_preserves_input_sharding(
    fake_mesh: Mesh,
    tied: bool,
    logit_dtype: DTypeLike,
) -> None:
    module = _embedding(tied)
    inputs = _sharded_vector(jnp.linspace(-1.0, 1.25, MODEL_DIM, dtype=jnp.float32))
    forward_pass_config = EmbeddingForwardPassConfig(logit_dtype=logit_dtype)

    def call(module: TiedEmbedding | UntiedEmbedding, values: Array) -> Array:
        return module.readout(
            values,
            keychain=Keychain.init(2, sharding_config=make_test_sharding_config()),
            forward_pass_config=forward_pass_config,
        )

    result = eqx.filter_jit(call)(module, inputs)

    _assert_close(result=result, reference=_readout_reference(module, inputs, forward_pass_config))
    assert result.dtype == jnp.dtype(forward_pass_config.logit_dtype)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_embed_dtype_can_be_overridden(fake_mesh: Mesh, tied: bool) -> None:
    module = _embedding(tied)
    token_id = jnp.array(2, dtype=jnp.int32)
    forward_pass_config = EmbeddingForwardPassConfig(activation_dtype=jnp.float32)

    result = module.embed(
        token_id,
        keychain=Keychain.init(12, sharding_config=make_test_sharding_config()),
        forward_pass_config=forward_pass_config,
    )

    assert result.dtype == jnp.float32
    _assert_named_sharding(result.sharding, fake_mesh)
    _assert_close(result=result, reference=_embed_reference(module, token_id, forward_pass_config))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_readout_defaults_to_float32_with_bfloat16_inputs(fake_mesh: Mesh, tied: bool) -> None:
    module = _embedding(tied)
    inputs = jnp.linspace(-1.0, 1.25, MODEL_DIM, dtype=jnp.bfloat16)

    result = module.readout(inputs, keychain=Keychain.init(13, sharding_config=make_test_sharding_config()))

    assert result.dtype == jnp.float32
    _assert_named_sharding(result.sharding, fake_mesh)
    _assert_close(result=result, reference=_readout_reference(module, inputs))


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_embedding_readout_vmapped_over_inputs_preserves_input_sharding(fake_mesh: Mesh, tied: bool) -> None:
    module = _embedding(tied)
    inputs = _sharded_vectors(jnp.arange(2 * MODEL_DIM, dtype=jnp.float32).reshape(2, MODEL_DIM) / 10)

    result = jax.vmap(
        lambda values: module.readout(values, keychain=Keychain.init(3, sharding_config=make_test_sharding_config()))
    )(inputs)
    reference = jax.vmap(lambda values: _readout_reference(module, values))(inputs)

    _assert_close(result=result, reference=reference)
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


def test_tied_embedding_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _tied_embedding()
    template = original.config.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
        model_dim=MODEL_DIM,
        vocab_size=VOCAB_SIZE,
    )
    inputs = _sharded_vector(jnp.linspace(-1.0, 1.25, MODEL_DIM, dtype=jnp.float32))

    restored = template.load_exported(original.export())
    result = restored.readout(inputs, keychain=Keychain.init(4, sharding_config=make_test_sharding_config()))

    assert isinstance(restored.embedding, FullPrecisionMatrix)
    assert isinstance(template.embedding, ShapeDtypeMatrix)
    assert template.embedding.shape == (VOCAB_SIZE, MODEL_DIM)
    assert template.embedding.decompress().shape == (MODEL_DIM, VOCAB_SIZE)
    assert restored.embedding.sharding_config == make_test_sharding_config().replicated_with_same_mesh()
    assert not is_sharded(restored.embedding.weights.sharding)
    _assert_close(
        result=result,
        reference=original.readout(inputs, keychain=Keychain.init(5, sharding_config=make_test_sharding_config())),
    )
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding


def test_untied_embedding_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _untied_embedding()
    template = original.config.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
        model_dim=MODEL_DIM,
        vocab_size=VOCAB_SIZE,
    )
    inputs = _sharded_vector(jnp.linspace(-1.0, 1.25, MODEL_DIM, dtype=jnp.float32))

    restored = template.load_exported(original.export())
    result = restored.readout(inputs, keychain=Keychain.init(6, sharding_config=make_test_sharding_config()))

    assert isinstance(restored.input_embedding, FullPrecisionMatrix)
    assert isinstance(restored.output_embedding, FullPrecisionMatrix)
    assert isinstance(template.input_embedding, ShapeDtypeMatrix)
    assert isinstance(template.output_embedding, ShapeDtypeMatrix)
    assert template.input_embedding.shape == (VOCAB_SIZE, MODEL_DIM)
    assert template.input_embedding.decompress().shape == (MODEL_DIM, VOCAB_SIZE)
    assert template.output_embedding.shape == (VOCAB_SIZE, MODEL_DIM)
    assert restored.input_embedding.sharding_config == make_test_sharding_config().replicated_with_same_mesh()
    assert restored.output_embedding.sharding_config == make_test_sharding_config()
    assert not is_sharded(restored.input_embedding.weights.sharding)
    _assert_named_sharding(restored.output_embedding.weights.sharding, fake_mesh)
    _assert_close(
        result=result,
        reference=original.readout(inputs, keychain=Keychain.init(7, sharding_config=make_test_sharding_config())),
    )
    _assert_named_sharding(result.sharding, fake_mesh)
    assert result.sharding == inputs.sharding
