import jax.numpy as jnp
import pytest
from frozendict import frozendict

from lalamo.modules import (
    Decoder,
    DecoderForwardPassConfig,
    DynamicKVCacheLayer,
    EmbeddingForwardPassConfig,
    ForwardPassMode,
    Keychain,
    StaticKVCacheLayer,
    TransformerForwardPassConfig,
)
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import build_tree_attention_mask, tree_ancestor_mask
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> Decoder:
    num_layers, kv_reuse_map = getattr(request, "param", (1, frozendict()))
    return build_tiny_attention_decoder(num_layers, kv_reuse_map)


def test_tree_ancestor_mask_chain() -> None:
    parent_indices = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)
    mask = tree_ancestor_mask(parent_indices)

    expected = jnp.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=jnp.bool,
    )
    assert jnp.array_equal(mask, expected)


def test_tree_ancestor_mask_fork() -> None:
    parent_indices = jnp.array([-1, 0, 0], dtype=jnp.int32)
    mask = tree_ancestor_mask(parent_indices)

    expected = jnp.array(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ],
        dtype=jnp.bool,
    )
    assert jnp.array_equal(mask, expected)


def test_build_tree_attention_mask_prefix_plus_draft() -> None:
    parent_indices = jnp.array([-1, 0, 0], dtype=jnp.int32)
    mask = build_tree_attention_mask(
        prefix_length=2,
        parent_indices=parent_indices,
        has_sinks=False,
        key_positions=jnp.arange(5, dtype=jnp.int32),
    )

    expected = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, False, True],
        ],
        dtype=jnp.bool,
    )
    assert jnp.array_equal(mask, expected)


@pytest.mark.parametrize(
    "decoder",
    [(1, frozendict()), (2, frozendict({1: 0}))],
    indirect=True,
    ids=["plain", "borrowed_kv"],
)
def test_tree_attention_matches_sequential_chain(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)
    forward_pass_config = DecoderForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
    )
    single_token_forward_pass_config = DecoderForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
        transformer_forward_pass_config=TransformerForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
    )

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
        forward_pass_config=forward_pass_config,
        keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
    )
    assert prefix_result.updated_state is not None

    sequential_logits = []
    state = prefix_result.updated_state
    for i, token_id in enumerate(chain_token_ids):
        step_result = decoder(
            token_id[None, None],
            jnp.array([[3 + i]], dtype=jnp.int32),
            state=state,
            return_updated_state=True,
            forward_pass_config=single_token_forward_pass_config,
            keychain=Keychain.init(20 + i, sharding_config=make_test_sharding_config()),
        )
        assert step_result.updated_state is not None
        state = step_result.updated_state
        sequential_logits.append(step_result.logits[0, -1])

    tree_result = decoder(
        chain_token_ids[None, :],
        jnp.array([[3, 4, 5]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        tree_ancestor_indices=jnp.array([[-1, 0, 1]], dtype=jnp.int32),
        forward_pass_config=single_token_forward_pass_config,
        keychain=Keychain.init(30, sharding_config=make_test_sharding_config()),
    )
    assert_close(
        result=tree_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="tree attention chain vs sequential logits",
    )


@pytest.mark.parametrize("decoder", [(2, frozendict({1: 0}))], indirect=True, ids=["borrowed_kv"])
def test_kv_borrowing_layer_projects_queries_only(decoder: Decoder) -> None:
    owner = decoder.transformer.layers[0].mixer
    borrower = decoder.transformer.layers[1].mixer
    assert isinstance(owner, Attention)
    assert isinstance(borrower, Attention)
    assert not owner.borrows_kv_cache
    assert borrower.borrows_kv_cache
    assert owner.qkv_projection.output_dims == (8, 8, 8)
    assert borrower.qkv_projection.output_dims == (8,)
    assert borrower.key_norm is None
    assert borrower.value_norm is None


@pytest.mark.parametrize("decoder", [(2, frozendict())], indirect=True)
def test_layers_have_distinct_ropes(decoder: Decoder) -> None:
    first, second = decoder.transformer.layers
    assert first.rope is not None
    assert second.rope is not None
    assert first.rope is not second.rope


def test_tree_attention_sibling_isolation(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1]], dtype=jnp.int32)

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
        keychain=Keychain.init(40, sharding_config=make_test_sharding_config()),
    )
    assert prefix_result.updated_state is not None

    state_a = prefix_result.updated_state
    result_a = decoder(
        jnp.array([[10]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_a,
        forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(50, sharding_config=make_test_sharding_config()),
    )
    logits_a = result_a.logits[0, 0]

    state_b = prefix_result.updated_state
    result_b = decoder(
        jnp.array([[20]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_b,
        forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(60, sharding_config=make_test_sharding_config()),
    )
    logits_b = result_b.logits[0, 0]

    tree_result = decoder(
        jnp.array([[10, 20]], dtype=jnp.int32),
        jnp.array([[2, 2]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        tree_ancestor_indices=jnp.array([[-1, -1]], dtype=jnp.int32),
        forward_pass_config=DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(70, sharding_config=make_test_sharding_config()),
    )

    assert_close(
        result=tree_result.logits[0, 0],
        reference=logits_a,
        operation_name="tree sibling A matches isolated A",
    )
    assert_close(
        result=tree_result.logits[0, 1],
        reference=logits_b,
        operation_name="tree sibling B matches isolated B",
    )


def test_tree_attention_mask_static_matches_dynamic() -> None:
    prefix_length = 3
    draft_length = 3

    dynamic_cache = DynamicKVCacheLayer.init(
        has_sinks=False,
        keys=jnp.zeros((prefix_length + draft_length, 1, 1), dtype=jnp.float32),
        values=jnp.zeros((prefix_length + draft_length, 1, 1), dtype=jnp.float32),
    )
    static_cache = StaticKVCacheLayer(
        has_sinks=False,
        keys=jnp.zeros((prefix_length + draft_length, 1, 1), dtype=jnp.float32),
        values=jnp.zeros((prefix_length + draft_length, 1, 1), dtype=jnp.float32),
        current_length=jnp.array(prefix_length + draft_length, dtype=jnp.int32),
    )

    parent_indices = jnp.array([-1, 0, 1], dtype=jnp.int32)
    dynamic_mask = dynamic_cache.tree_attention_mask(
        prefix_length=prefix_length,
        parent_indices=parent_indices,
    )
    static_mask = static_cache.tree_attention_mask(
        prefix_length=prefix_length,
        parent_indices=parent_indices,
    )
    assert jnp.array_equal(dynamic_mask, static_mask)


def test_reused_dynamic_cache_preserves_sparse_padding_mask() -> None:
    keys = jnp.zeros((5, 1, 1), dtype=jnp.float32)
    cache = DynamicKVCacheLayer.init(has_sinks=False, keys=keys, values=keys, length=jnp.array(3, dtype=jnp.int32))
    cache = cache.extend(jnp.ones((1, 1, 1), dtype=jnp.float32), jnp.ones((1, 1, 1), dtype=jnp.float32))
    mask = cache.attention_mask(suffix_length=1, is_causal=True, suffix_length_without_padding=1)

    assert jnp.array_equal(mask, jnp.array([[True, True, True, False, False, True]], dtype=jnp.bool))


def test_reused_tree_mask_keeps_sparse_suffix() -> None:
    keys = jnp.zeros((5, 1, 1), dtype=jnp.float32)
    cache = DynamicKVCacheLayer.init(has_sinks=False, keys=keys, values=keys, length=jnp.array(3, dtype=jnp.int32))
    cache = cache.extend(jnp.ones((1, 1, 1), dtype=jnp.float32), jnp.ones((1, 1, 1), dtype=jnp.float32))
    prefix_length = cache.current_prefix_length() - 1
    mask = cache.tree_attention_mask(prefix_length, jnp.array([-1], dtype=jnp.int32))

    assert jnp.array_equal(mask, jnp.array([[True, True, True, False, False, True]], dtype=jnp.bool))
