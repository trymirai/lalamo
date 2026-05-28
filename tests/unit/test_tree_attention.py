from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    AttentionImplementation,
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    DenseMLPConfig,
    DynamicKVCacheLayer,
    EmbeddingForwardPassConfig,
    ForwardPassMode,
    Identity,
    Keychain,
    LinearConfig,
    NormalizationConfig,
    StaticKVCacheLayer,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerForwardPassConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig, AttentionProjectionMode
from lalamo.modules.token_mixers.kv_cache import build_tree_attention_mask, tree_ancestor_mask
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.helpers import make_test_sharding_config


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> Decoder:
    kv_source_per_layer: tuple[int, ...] = getattr(request, "param", (0,))
    precision = jnp.float32
    model_dim = 16
    hidden_dim = 32
    vocab_size = 32
    context_length = 16

    norm_config = NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    rope_config = UnscaledRoPEConfig(
        base=10_000.0,
        max_sequence_length=context_length,
        head_dim=8,
    )

    layer_configs = []
    for layer_index, kv_source in enumerate(kv_source_per_layer):
        if kv_source == layer_index:
            projection_mode = AttentionProjectionMode.QKV
        else:
            projection_mode = AttentionProjectionMode.BORROWED_Q
        attention_config = AttentionConfig(
            qkv_projection_config=LinearConfig(),
            out_projection_config=LinearConfig(),
            query_norm_config=None,
            key_norm_config=None,
            num_heads=2,
            num_groups=2,
            head_dim=8,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
            projection_mode=projection_mode,
        )
        layer_configs.append(
            TransformerLayerConfig(
                pre_mixer_norm_config=norm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
                rope_config=rope_config,
            )
        )

    transformer_config = TransformerConfig(
        layer_configs=tuple(layer_configs),
        output_norm_config=norm_config,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        kv_source_per_layer=kv_source_per_layer,
    )
    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        transformer_config=transformer_config,
        vocab_size=vocab_size,
    )
    return decoder_config.init(
        RandomInitializer(
            default_dtype=precision,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(4),
        ),
    )


def _standard_single_token_config() -> DecoderForwardPassConfig:
    transformer_config = TransformerForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)
    return DecoderForwardPassConfig(
        transformer_forward_pass_config=replace(
            transformer_config,
            mixer_forward_pass_config=replace(
                transformer_config.mixer_forward_pass_config,
                attention_implementation=AttentionImplementation.STANDARD,
            ),
        ),
    )


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


def test_tree_attention_rejects_invalid_layouts() -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="parent_indices"):
        tree_ancestor_mask(jnp.array([-1, 1], dtype=jnp.int32)).block_until_ready()

    with pytest.raises(eqx.EquinoxRuntimeError, match="total_capacity"):
        build_tree_attention_mask(
            total_capacity=2,
            prefix_length=1,
            parent_indices=jnp.array([-1, 0], dtype=jnp.int32),
            has_sinks=False,
        ).block_until_ready()


def test_transformer_config_derives_kv_cache_layout(decoder: Decoder) -> None:
    layer_config = decoder.config.transformer_config.layer_configs[0]
    attention_config = layer_config.mixer_config
    assert isinstance(attention_config, AttentionConfig)
    borrowed_layer_config = replace(
        layer_config,
        mixer_config=replace(
            attention_config,
            key_norm_config=None,
            projection_mode=AttentionProjectionMode.BORROWED_Q,
        ),
    )
    transformer_config = replace(
        decoder.config.transformer_config,
        layer_configs=(layer_config, borrowed_layer_config, layer_config, borrowed_layer_config),
        kv_source_per_layer=(0, 0, 2, 2),
    )

    assert transformer_config.kv_source_per_layer == (0, 0, 2, 2)
    assert transformer_config.kv_cache_source_layers == (0, 2)
    assert transformer_config.kv_cache_width == 2


@pytest.mark.parametrize("kv_source_per_layer", [(1, 1), (0, 0, 1)])
def test_transformer_config_rejects_invalid_kv_cache_layout(
    decoder: Decoder,
    kv_source_per_layer: tuple[int, ...],
) -> None:
    layer_config = decoder.config.transformer_config.layer_configs[0]
    with pytest.raises(ValueError):
        replace(
            decoder.config.transformer_config,
            layer_configs=(layer_config,) * len(kv_source_per_layer),
            kv_source_per_layer=kv_source_per_layer,
        )


def test_transformer_config_rejects_invalid_projection_modes(decoder: Decoder) -> None:
    layer_config = decoder.config.transformer_config.layer_configs[0]
    attention_config = layer_config.mixer_config
    assert isinstance(attention_config, AttentionConfig)

    with pytest.raises(ValueError, match="must use borrowed_q"):
        replace(
            decoder.config.transformer_config,
            layer_configs=(layer_config, layer_config),
            kv_source_per_layer=(0, 0),
        )

    borrowed_layer_config = replace(
        layer_config,
        mixer_config=replace(
            attention_config,
            key_norm_config=None,
            projection_mode=AttentionProjectionMode.BORROWED_Q,
        ),
    )
    with pytest.raises(ValueError, match="owns its KV cache"):
        replace(
            decoder.config.transformer_config,
            layer_configs=(borrowed_layer_config,),
            kv_source_per_layer=(0,),
        )


@pytest.mark.parametrize("decoder", [(0,), (0, 0)], indirect=True, ids=["plain", "shared_kv"])
def test_tree_attention_matches_sequential_chain(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)
    forward_pass_config = DecoderForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
    )
    single_token_forward_pass_config = replace(
        _standard_single_token_config(),
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
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


@pytest.mark.parametrize("decoder", [(0, 0)], indirect=True, ids=["shared_kv"])
def test_kv_sharing_layer_projects_queries_only(decoder: Decoder) -> None:
    owner = decoder.transformer.layers[0].mixer
    shared = decoder.transformer.layers[1].mixer
    assert isinstance(owner, Attention)
    assert isinstance(shared, Attention)
    assert owner.qkv_projection.output_dims == (16, 16, 16)
    assert shared.qkv_projection.output_dims == (16,)
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
        forward_pass_config=_standard_single_token_config(),
        keychain=Keychain.init(50, sharding_config=make_test_sharding_config()),
    )
    logits_a = result_a.logits[0, 0]

    state_b = prefix_result.updated_state
    result_b = decoder(
        jnp.array([[20]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=state_b,
        forward_pass_config=_standard_single_token_config(),
        keychain=Keychain.init(60, sharding_config=make_test_sharding_config()),
    )
    logits_b = result_b.logits[0, 0]

    tree_result = decoder(
        jnp.array([[10, 20]], dtype=jnp.int32),
        jnp.array([[2, 2]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        attention_parent_indices=jnp.array([[-1, -1]], dtype=jnp.int32),
        forward_pass_config=_standard_single_token_config(),
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


def test_shared_kv_runtime_uses_compact_state(decoder: Decoder) -> None:
    layer_config = decoder.config.transformer_config.layer_configs[0]
    attention_config = layer_config.mixer_config
    assert isinstance(attention_config, AttentionConfig)
    borrowed_layer_config = replace(
        layer_config,
        mixer_config=replace(
            attention_config,
            key_norm_config=None,
            projection_mode=AttentionProjectionMode.BORROWED_Q,
        ),
    )
    decoder_config = replace(
        decoder.config,
        transformer_config=replace(
            decoder.config.transformer_config,
            layer_configs=(layer_config, borrowed_layer_config, layer_config, borrowed_layer_config),
            kv_source_per_layer=(0, 0, 2, 2),
        ),
    )
    shared_decoder = decoder_config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(80),
        ),
    )

    result = shared_decoder(
        jnp.array([[1, 2]], dtype=jnp.int32),
        jnp.array([[0, 1]], dtype=jnp.int32),
        return_updated_state=True,
        keychain=Keychain.init(81, sharding_config=make_test_sharding_config()),
    )
    assert result.updated_state is not None
    assert len(result.updated_state) == 2

    static_state = shared_decoder.init_static_state(batch_size=2, capacity=8, dtype=jnp.bfloat16)
    assert len(static_state) == 2
    static_followup = shared_decoder(
        jnp.array([[3], [4]], dtype=jnp.int32),
        jnp.array([[0], [0]], dtype=jnp.int32),
        state=static_state,
        return_updated_state=True,
        forward_pass_config=_standard_single_token_config(),
        keychain=Keychain.init(83, sharding_config=make_test_sharding_config()),
    )
    assert static_followup.updated_state is not None
    assert len(static_followup.updated_state) == 2

    followup = shared_decoder(
        jnp.array([[3]], dtype=jnp.int32),
        jnp.array([[2]], dtype=jnp.int32),
        state=result.updated_state,
        return_updated_state=True,
        forward_pass_config=_standard_single_token_config(),
        keychain=Keychain.init(82, sharding_config=make_test_sharding_config()),
    )
    assert followup.updated_state is not None
    assert len(followup.updated_state) == 2


def test_static_kv_cache_rejects_capacity_overrun() -> None:
    cache = StaticKVCacheLayer(
        has_sinks=False,
        keys=jnp.zeros((5, 1, 1), dtype=jnp.float32),
        values=jnp.zeros((5, 1, 1), dtype=jnp.float32),
        current_length=jnp.array(4, dtype=jnp.int32),
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="capacity"):
        cache.extend(
            jnp.ones((3, 1, 1), dtype=jnp.float32),
            jnp.ones((3, 1, 1), dtype=jnp.float32),
        ).current_length.block_until_ready()
