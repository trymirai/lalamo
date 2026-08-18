from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import Mesh, NamedSharding, Sharding
from jaxtyping import Array

from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer
from lalamo.modules.utils import call_vmapped
from lalamo.utils.dummy_array import dummy_array
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec
from tests.common import assert_close
from tests.helpers import make_sharding, make_test_sharding_config

MODEL_DIM = 4
NUM_HEADS = 2
NUM_GROUPS = 2
HEAD_DIM = 2


def _weights(shape: tuple[int, ...], *, offset: int = 0) -> jax.Array:
    return (jnp.arange(offset, offset + prod(shape), dtype=jnp.float32).reshape(shape) / 20) - 0.5


def _linear(weights: Array, output_dims: tuple[int, ...]) -> Linear:
    return Linear(
        config=LinearConfig(),
        sharding_config=make_test_sharding_config(),
        weights=FullPrecisionSpec().compress(weights, sharding_config=make_test_sharding_config()),
        biases=None,
        output_dims=output_dims,
    )


def _attention(
    *,
    logit_soft_cap: float | None = None,
    has_gate: bool = False,
    is_kv_sharing: bool = False,
) -> Attention:
    qkv_dim = NUM_HEADS * HEAD_DIM
    output_dims = (qkv_dim,) if is_kv_sharing else (qkv_dim, qkv_dim, qkv_dim)
    if has_gate:
        output_dims = (*output_dims, qkv_dim)
    return Attention(
        config=AttentionConfig(
            qkvg_projection_config=LinearConfig(),
            out_projection_config=LinearConfig(),
            query_norm_config=None,
            key_norm_config=None,
            num_heads=NUM_HEADS,
            num_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=logit_soft_cap,
            has_sinks=False,
            has_qkvg_biases=False,
            has_out_biases=False,
            has_gate=has_gate,
            is_kv_sharing=is_kv_sharing,
        ),
        sharding_config=make_test_sharding_config(),
        qkvg_projection=_linear(_weights((sum(output_dims), MODEL_DIM)), output_dims),
        out_projection=_linear(_weights((MODEL_DIM, qkv_dim), offset=100), (MODEL_DIM,)),
        query_norm=None,
        key_norm=None,
        sinks=None,
    )


def _linear_reference(linear: Linear, inputs: Array) -> tuple[Array, ...]:
    weights = linear.weights.decompress()
    outputs = jnp.einsum("...i,oi->...o", inputs, weights)
    return tuple(jnp.split(outputs, Linear.get_split_points(linear.output_dims), axis=-1))


def _reference(module: Attention, inputs: Array) -> Array:
    inputs = jnp.asarray(jax.device_get(inputs))
    projections = _linear_reference(module.qkvg_projection, inputs)
    queries, keys, values, *_ = projections
    queries = rearrange(queries, "tokens (heads channels) -> tokens heads channels", heads=NUM_HEADS)
    keys = rearrange(keys, "tokens (groups channels) -> tokens groups channels", groups=NUM_GROUPS)
    values = rearrange(values, "tokens (groups channels) -> tokens groups channels", groups=NUM_GROUPS)

    logits = jnp.einsum("thc,shc->hts", queries, keys) * (HEAD_DIM**-0.5)
    causal_mask = jnp.arange(inputs.shape[0])[:, None] >= jnp.arange(inputs.shape[0])[None, :]
    logits = jnp.where(causal_mask[None, :, :], logits, jnp.array(float("-inf"), dtype=logits.dtype))
    attention_weights = jax.nn.softmax(logits, axis=-1)
    attention_output = jnp.einsum("hts,shc->thc", attention_weights, values)
    attention_output = rearrange(attention_output, "tokens heads channels -> tokens (heads channels)")
    if module.config.has_gate:
        attention_output = attention_output * jax.nn.sigmoid(projections[-1])
    (outputs,) = _linear_reference(module.out_projection, attention_output)
    return outputs


def _assert_named_sharding(sharding: Sharding, mesh: Mesh) -> None:
    assert isinstance(sharding, NamedSharding)
    assert sharding.mesh == mesh


def _assert_close(result: Array, reference: Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def _inputs() -> Array:
    return jnp.arange(5 * MODEL_DIM, dtype=jnp.float32).reshape(5, MODEL_DIM) / 10


def _sharded_sequence(values: Array) -> Array:
    return jax.device_put(values, make_sharding((None, None)))


def _sharded_sequences(values: Array) -> Array:
    return jax.device_put(values, make_sharding((LogicalAxis.BATCH, None, None)))


def test_attention_matches_reference_and_preserves_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(_inputs())

    result = module(
        inputs, positional_embeddings=None, keychain=Keychain.init(0, sharding_config=make_test_sharding_config())
    )

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
    assert result.state is None


def test_gated_attention_matches_reference(fake_mesh: Mesh) -> None:
    module = _attention(has_gate=True)
    inputs = _sharded_sequence(_inputs())

    result = module(
        inputs,
        positional_embeddings=None,
        keychain=Keychain.init(11, sharding_config=make_test_sharding_config()),
    )

    assert module.qkvg_projection.output_dims == (4, 4, 4, 4)
    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)


def test_gated_kv_sharing_matches_equivalent_full_projection(fake_mesh: Mesh) -> None:
    source = _attention()
    query_gate = _attention(has_gate=True, is_kv_sharing=True)
    query_weights, gate_weights = jnp.split(query_gate.qkvg_projection.weights.decompress(), 2)
    _, key_weights, value_weights = jnp.split(source.qkvg_projection.weights.decompress(), 3)
    full = _attention(has_gate=True)
    full_projection = _linear(
        jnp.concatenate((query_weights, key_weights, value_weights, gate_weights)),
        (NUM_HEADS * HEAD_DIM,) * 4,
    )
    full = eqx.tree_at(lambda module: module.qkvg_projection, full, full_projection)
    inputs = _sharded_sequence(_inputs())

    source_result = source(
        inputs,
        positional_embeddings=None,
        return_updated_state=True,
        keychain=Keychain.init(12, sharding_config=make_test_sharding_config()),
    )
    assert source_result.state is not None
    shared_result = query_gate(
        inputs,
        positional_embeddings=None,
        state=source_result.state,
        reuse_cache=True,
        keychain=Keychain.init(13, sharding_config=make_test_sharding_config()),
    )
    full_result = full(
        inputs,
        positional_embeddings=None,
        keychain=Keychain.init(14, sharding_config=make_test_sharding_config()),
    )

    assert query_gate.qkvg_projection.output_dims == (4, 4)
    _assert_close(result=shared_result.outputs, reference=full_result.outputs)
    _assert_named_sharding(shared_result.outputs.sharding, fake_mesh)


def test_attention_returns_dynamic_state_with_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(_inputs())

    result = module(
        inputs,
        positional_embeddings=None,
        return_updated_state=True,
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    )

    assert result.state is not None
    _assert_named_sharding(result.state.keys.sharding, fake_mesh)
    _assert_named_sharding(result.state.values.sharding, fake_mesh)
    assert result.state.keys.sharding == make_sharding((None, None, None))
    assert result.state.values.sharding == make_sharding((None, None, None))


def test_paged_decode_matches_dense_decode_and_updates_physical_page(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(_inputs()[:3])
    prefix_result = module(
        inputs[:2],
        positional_embeddings=None,
        return_updated_state=True,
        keychain=Keychain.init(20, sharding_config=make_test_sharding_config()),
    )
    assert prefix_result.state is not None
    dense_result = module(
        inputs[2:],
        positional_embeddings=None,
        state=prefix_result.state,
        return_updated_state=True,
        keychain=Keychain.init(21, sharding_config=make_test_sharding_config()),
    )
    assert dense_result.state is not None

    key_pages = jnp.zeros((NUM_GROUPS, 4, 2, HEAD_DIM), dtype=jnp.float32)
    value_pages = jnp.zeros_like(key_pages)
    key_pages = key_pages.at[:, 2].set(
        rearrange(prefix_result.state.keys, "tokens groups channels -> groups tokens channels")
    )
    key_pages = key_pages.at[:, 1].set(
        rearrange(prefix_result.state.keys, "tokens groups channels -> groups tokens channels")
    )
    value_pages = value_pages.at[:, 2].set(
        rearrange(prefix_result.state.values, "tokens groups channels -> groups tokens channels")
    )
    value_pages = value_pages.at[:, 1].set(
        rearrange(prefix_result.state.values, "tokens groups channels -> groups tokens channels")
    )
    paged_inputs = _sharded_sequences(jnp.repeat(inputs[None, 2:], 2, axis=0))
    paged_result = module.paged_decode(
        paged_inputs,
        positional_embeddings=None,
        state=PagedKVCacheLayer(
            keys=jax.device_put(key_pages, make_sharding((None, None, None, None))),
            values=jax.device_put(value_pages, make_sharding((None, None, None, None))),
            block_tables=jax.device_put(
                jnp.array([[2, 0], [1, 3]], dtype=jnp.int32),
                make_sharding((None, None)),
            ),
            lengths=jax.device_put(
                jnp.array([2, 2], dtype=jnp.int32),
                make_sharding((None,)),
            ),
        ),
        return_updated_state=True,
        forward_pass_config=MixerForwardPassConfig(),
        keychain=Keychain.init(21, shape=(2,), sharding_config=make_test_sharding_config()),
    )
    assert paged_result.state is not None

    paged_outputs = jnp.asarray(jax.device_get(paged_result.outputs))
    _assert_close(result=paged_outputs[0], reference=dense_result.outputs)
    _assert_close(result=paged_outputs[1], reference=dense_result.outputs)
    _assert_named_sharding(paged_result.outputs.sharding, fake_mesh)
    assert jnp.array_equal(
        paged_result.state.keys[:, 0, 0],
        rearrange(dense_result.state.keys[-1], "groups channels -> groups channels"),
    )
    assert jnp.array_equal(
        paged_result.state.values[:, 0, 0],
        rearrange(dense_result.state.values[-1], "groups channels -> groups channels"),
    )
    assert jnp.array_equal(paged_result.state.keys[:, 3, 0], paged_result.state.keys[:, 0, 0])
    assert jnp.array_equal(paged_result.state.values[:, 3, 0], paged_result.state.values[:, 0, 0])


def test_attention_output_dtype_matches_input_dtype(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(jnp.arange(5 * MODEL_DIM, dtype=jnp.bfloat16).reshape(5, MODEL_DIM) / 10)

    result = module(
        inputs, positional_embeddings=None, keychain=Keychain.init(6, sharding_config=make_test_sharding_config())
    )

    assert result.outputs.dtype == inputs.dtype
    _assert_named_sharding(result.outputs.sharding, fake_mesh)


def test_attention_under_jit_matches_reference_and_preserves_tensor_sharding(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(_inputs())

    result = eqx.filter_jit(
        lambda module, values: module(
            values, positional_embeddings=None, keychain=Keychain.init(2, sharding_config=make_test_sharding_config())
        ),
    )(module, inputs)

    _assert_close(result=result.outputs, reference=_reference(module, inputs))
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))


def test_attention_implementations_match(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequence(_inputs())

    standard = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.STANDARD,
        ),
        keychain=Keychain.init(7, sharding_config=make_test_sharding_config()),
    )
    stable_reduction = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.STABLE_REDUCTION,
            attention_tile_size=2,
        ),
        keychain=Keychain.init(8, sharding_config=make_test_sharding_config()),
    )

    _assert_close(result=stable_reduction.outputs, reference=standard.outputs)
    _assert_named_sharding(stable_reduction.outputs.sharding, fake_mesh)


def test_soft_capped_attention_implementations_match(fake_mesh: Mesh) -> None:
    module = _attention(logit_soft_cap=0.5)
    inputs = _sharded_sequence(_inputs())

    standard = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.STANDARD,
        ),
        keychain=Keychain.init(9, sharding_config=make_test_sharding_config()),
    )
    stable_reduction = module(
        inputs,
        positional_embeddings=None,
        forward_pass_config=MixerForwardPassConfig(
            attention_implementation=AttentionImplementation.STABLE_REDUCTION,
            attention_tile_size=2,
        ),
        keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
    )

    _assert_close(result=stable_reduction.outputs, reference=standard.outputs)
    _assert_named_sharding(stable_reduction.outputs.sharding, fake_mesh)


def test_attention_vmapped_over_inputs_matches_reference_and_keeps_data_sharding(fake_mesh: Mesh) -> None:
    module = _attention()
    inputs = _sharded_sequences(jnp.arange(2 * 5 * MODEL_DIM, dtype=jnp.float32).reshape(2, 5, MODEL_DIM) / 10)

    result = call_vmapped(
        lambda values, *, keychain: module(values, positional_embeddings=None, keychain=keychain),
        inputs,
        keychain=Keychain.init(3, sharding_config=make_test_sharding_config()),
        added_sharding_axis=make_test_sharding_config().resolve_axis(LogicalAxis.BATCH),
    )
    reference = jnp.stack([_reference(module, values) for values in jnp.asarray(jax.device_get(inputs))])

    _assert_close(result=result.outputs, reference=reference)
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((LogicalAxis.BATCH, None, None))


def test_attention_export_load_roundtrips_and_preserves_template_sharding(fake_mesh: Mesh) -> None:
    original = _attention()
    weight_sharding = make_sharding((None, None))
    template = Attention(
        config=original.config,
        sharding_config=make_test_sharding_config(),
        qkvg_projection=Linear(
            config=LinearConfig(),
            sharding_config=make_test_sharding_config(),
            weights=FullPrecisionSpec().compress(
                dummy_array(original.qkvg_projection.weights.shape, jnp.float32, weight_sharding),
                sharding_config=make_test_sharding_config(),
            ),
            biases=None,
            output_dims=original.qkvg_projection.output_dims,
        ),
        out_projection=Linear(
            config=LinearConfig(),
            sharding_config=make_test_sharding_config(),
            weights=FullPrecisionSpec().compress(
                dummy_array(original.out_projection.weights.shape, jnp.float32, weight_sharding),
                sharding_config=make_test_sharding_config(),
            ),
            biases=None,
            output_dims=original.out_projection.output_dims,
        ),
        query_norm=None,
        key_norm=None,
        sinks=None,
    )
    inputs = _sharded_sequence(_inputs())

    restored = template.load_exported(original.export())
    result = restored(
        inputs, positional_embeddings=None, keychain=Keychain.init(4, sharding_config=make_test_sharding_config())
    )

    assert isinstance(restored.qkvg_projection.weights, FullPrecisionMatrix)
    assert isinstance(restored.out_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.qkvg_projection.weights, FullPrecisionMatrix)
    assert isinstance(template.out_projection.weights, FullPrecisionMatrix)
    assert restored.qkvg_projection.weights.weights.sharding == template.qkvg_projection.weights.weights.sharding
    assert restored.out_projection.weights.weights.sharding == template.out_projection.weights.weights.sharding
    _assert_close(
        result=result.outputs,
        reference=original(
            inputs, positional_embeddings=None, keychain=Keychain.init(5, sharding_config=make_test_sharding_config())
        ).outputs,
    )
    _assert_named_sharding(result.outputs.sharding, fake_mesh)
    assert result.outputs.sharding == make_sharding((None, None))
