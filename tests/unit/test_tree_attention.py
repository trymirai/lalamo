import jax
import jax.numpy as jnp
import pytest

from lalamo.modules import (
    Decoder,
    DynamicKVCacheLayer,
    EmbeddingForwardPassConfig,
    ForwardPassMode,
    Keychain,
    ModelForwardPassConfig,
    StaticKVCacheLayer,
)
from lalamo.modules.flow import FlowForwardPassConfig, leaves
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.kv_cache import build_tree_attention_mask, tree_ancestor_mask
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


@pytest.mark.parametrize("capacity", [None, 16], ids=["dynamic", "static"])
def test_padded_shared_kv_tree_matches_each_unpadded_row(capacity: int | None) -> None:
    model = build_tiny_attention_decoder((None, None, 0, 1))
    sharding = model.sharding_config
    with jax.set_mesh(sharding.mesh):
        tokens = jax.random.randint(jax.random.key(0), (2, 9), 0, model.vocab_size)
        positions = jnp.broadcast_to(jnp.arange(9, dtype=jnp.int32), tokens.shape)
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()
        state = None
        if capacity is not None:
            state = model.init_static_state(2, capacity, jnp.float32)
        prefix = model(
            tokens[:, :4],
            positions[:, :4],
            state,
            return_updated_state=True,
            forward_pass_config=forward,
            keychain=keychain,
        ).updated_state
        assert prefix is not None
        parents = jnp.array([[-1, 0, 0, 2, 1], [-1, 0, 0, 2, 1]], dtype=jnp.int32)
        lengths = jnp.array([5, 3], dtype=jnp.int32)
        batch = model(
            tokens[:, 4:],
            positions[:, 4:],
            prefix,
            lengths_without_padding=lengths,
            attention_parent_indices=parents,
            forward_pass_config=forward,
            keychain=keychain,
        )
        for row, length in enumerate(lengths.tolist()):
            row_state = jax.tree.map(lambda array, row=row: array[row : row + 1], prefix)
            single = model(
                tokens[row : row + 1, 4 : 4 + length],
                positions[row : row + 1, 4 : 4 + length],
                row_state,
                attention_parent_indices=parents[row : row + 1, :length],
                forward_pass_config=forward,
                keychain=keychain,
            )
            assert_close(
                result=batch.logits[row, :length],
                reference=single.logits[0],
                operation_name="padded shared-KV tree",
            )


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> Decoder:
    kv_source_layer_indices: tuple[int | None, ...] = getattr(request, "param", (None,))
    return build_tiny_attention_decoder(kv_source_layer_indices)


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
        total_capacity=5,
        prefix_length=2,
        parent_indices=parent_indices,
        has_sinks=False,
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


@pytest.mark.parametrize("decoder", [(None,), (None, 0)], indirect=True, ids=["plain", "shared_kv"])
def test_tree_attention_matches_sequential_chain(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)
    forward_pass_config = ModelForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
    )
    single_token_forward_pass_config = ModelForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
        flow_forward_pass_config=FlowForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
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
        attention_parent_indices=jnp.array([[-1, 0, 1]], dtype=jnp.int32),
        forward_pass_config=single_token_forward_pass_config,
        keychain=Keychain.init(30, sharding_config=make_test_sharding_config()),
    )
    assert_close(
        result=tree_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="tree attention chain vs sequential logits",
    )


@pytest.mark.parametrize("decoder", [(None, 0)], indirect=True, ids=["shared_kv"])
def test_kv_sharing_layer_projects_queries_only(decoder: Decoder) -> None:
    owner, shared = (step for step in leaves(decoder.program) if isinstance(step, Attention))
    assert shared.config.kv_source is owner.config
    assert owner.qkvg_projection.output_dims == (8, 8, 8)
    assert shared.qkvg_projection.output_dims == (8,)
    assert shared.key_norm is None


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
        forward_pass_config=ModelForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(50, sharding_config=make_test_sharding_config()),
    )
    logits_a = result_a.logits[0, 0]

    state_b = prefix_result.updated_state
    result_b = decoder(
        jnp.array([[20]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_b,
        forward_pass_config=ModelForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(60, sharding_config=make_test_sharding_config()),
    )
    logits_b = result_b.logits[0, 0]

    tree_result = decoder(
        jnp.array([[10, 20]], dtype=jnp.int32),
        jnp.array([[2, 2]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        attention_parent_indices=jnp.array([[-1, -1]], dtype=jnp.int32),
        forward_pass_config=ModelForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
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
