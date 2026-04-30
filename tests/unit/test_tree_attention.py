import jax
import jax.numpy as jnp
import pytest

from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DenseMLPConfig,
    DynamicKVCacheLayer,
    ForwardPassMode,
    FullPrecisionLinearConfig,
    Identity,
    NormalizationConfig,
    StaticKVCacheLayer,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.state.kv_cache import build_tree_attention_mask, tree_ancestor_mask
from tests.common import assert_close


@pytest.fixture
def decoder() -> Decoder:
    precision = jnp.float32
    model_dim = 8
    hidden_dim = 16
    vocab_size = 32
    context_length = 16

    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    attention_config = AttentionConfig(
        qkv_projection_config=FullPrecisionLinearConfig(precision=precision),
        out_projection_config=FullPrecisionLinearConfig(precision=precision),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=4,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )
    mlp_config = DenseMLPConfig(
        linear_config=FullPrecisionLinearConfig(precision=precision),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    rope_config = UnscaledRoPEConfig(
        precision=precision,
        base=10_000.0,
        max_sequence_length=context_length,
        head_dim=4,
    )
    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=norm_config,
        mixer_config=attention_config,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
        rope_config=rope_config,
    )
    transformer_config = TransformerConfig(
        layer_configs=(layer_config,),
        output_norm_config=norm_config,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )
    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=precision,
        ),
        transformer_config=transformer_config,
        vocab_size=vocab_size,
    )
    return decoder_config.random_init(key=jax.random.key(4))


def test_tree_ancestor_mask_chain() -> None:
    """Linear chain: each node should attend to itself and all ancestors."""
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
    """Forked tree: siblings must not see each other."""
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
    """Draft nodes attend to all prefix tokens plus their own ancestor chain."""
    parent_indices = jnp.array([-1, 0, 0], dtype=jnp.int32)
    mask = build_tree_attention_mask(
        total_capacity=5,
        prefix_length=2,
        parent_indices=parent_indices,
        has_sinks=False,
    )

    # Columns: [prefix0, prefix1, draft0, draft1, draft2]
    expected = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, False, True],
        ],
        dtype=jnp.bool,
    )
    assert jnp.array_equal(mask, expected)


def test_tree_attention_matches_sequential_chain(decoder: Decoder) -> None:
    """Tree attention on a linear chain should match sequential single-token decode."""
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
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
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        )
        assert step_result.updated_state is not None
        state = step_result.updated_state
        sequential_logits.append(step_result.logits[0, -1])

    # Tree attention: chain is also a tree (each node has single ancestor)
    tree_result = decoder(
        chain_token_ids[None, :],
        jnp.array([[3, 4, 5]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        attention_parent_indices=jnp.array([[-1, 0, 1]], dtype=jnp.int32),
    )

    assert_close(
        result=tree_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="tree attention chain vs sequential logits",
    )


def test_tree_attention_sibling_isolation(decoder: Decoder) -> None:
    """Sibling draft tokens must not see each other in tree attention."""
    prefix_token_ids = jnp.array([[1, 2]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1]], dtype=jnp.int32)

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    # Compute sibling A alone
    state_a = prefix_result.updated_state
    result_a = decoder(
        jnp.array([[10]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_a,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
    )
    logits_a = result_a.logits[0, 0]

    # Compute sibling B alone
    state_b = prefix_result.updated_state
    result_b = decoder(
        jnp.array([[20]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_b,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
    )
    logits_b = result_b.logits[0, 0]

    # Tree attention: both siblings in same batch, parent_indices=[-1, -1]
    tree_result = decoder(
        jnp.array([[10, 20]], dtype=jnp.int32),
        jnp.array([[2, 2]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        attention_parent_indices=jnp.array([[-1, -1]], dtype=jnp.int32),
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
    """Static and dynamic KV caches should produce identical tree masks."""
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
