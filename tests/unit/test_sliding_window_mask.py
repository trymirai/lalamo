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
from tests.common import assert_close


@pytest.fixture
def decoder() -> Decoder:
    precision = jnp.float32
    model_dim = 8
    hidden_dim = 16
    vocab_size = 32
    context_length = 128

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
        sliding_window_size=2,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        use_rope=True,
    )
    mlp_config = DenseMLPConfig(
        linear_config=FullPrecisionLinearConfig(precision=precision),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=norm_config,
        mixer_config=attention_config,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
    )
    transformer_config = TransformerConfig(
        global_rope_config=UnscaledRoPEConfig(
            precision=precision,
            base=10_000.0,
            max_sequence_length=context_length,
        ),
        local_rope_config=None,
        layer_configs=(layer_config, layer_config),
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


def test_dynamic_kv_cache_sliding_window_mask_matches_static_mask() -> None:
    prefix_length = 3
    suffix_length = 4
    sliding_window_size = 2

    dynamic_cache = DynamicKVCacheLayer.init(
        has_sinks=False,
        keys=jnp.zeros((prefix_length + suffix_length, 1, 1), dtype=jnp.float32),
        values=jnp.zeros((prefix_length + suffix_length, 1, 1), dtype=jnp.float32),
    )
    static_cache = StaticKVCacheLayer.init(
        has_sinks=False,
        capacity=prefix_length + suffix_length,
        num_groups=1,
        head_dim=1,
        dtype=jnp.float32,
    )
    static_cache = StaticKVCacheLayer(
        has_sinks=static_cache.has_sinks,
        keys=static_cache.keys,
        values=static_cache.values,
        current_length=jnp.array(prefix_length + suffix_length, dtype=jnp.int32),
    )

    dynamic_mask = dynamic_cache.attention_mask(
        suffix_length=suffix_length,
        is_causal=True,
        sliding_window_size=sliding_window_size,
    )
    static_mask = static_cache.attention_mask(
        suffix_length=suffix_length,
        is_causal=True,
        sliding_window_size=sliding_window_size,
    )

    assert jnp.array_equal(dynamic_mask, static_mask)


def test_sliding_window_dynamic_state_multi_token_matches_single_token_sequential(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    suffix_token_ids = jnp.arange(4, 68, dtype=jnp.int32)[None, :]
    suffix_positions = jnp.arange(3, 67, dtype=jnp.int32)[None, :]

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    sequential_logits = []
    sequential_hidden = []
    state = prefix_result.updated_state
    for token_id, token_position in zip(suffix_token_ids[0], suffix_positions[0], strict=True):
        step_result = decoder(
            token_id[None, None],
            token_position[None, None],
            state=state,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        )
        assert step_result.updated_state is not None
        assert step_result.activation_trace is not None
        state = step_result.updated_state
        sequential_logits.append(step_result.logits[0, -1])
        sequential_hidden.append(step_result.activation_trace.output_norm[0, -1])

    multi_result = decoder(
        suffix_token_ids,
        suffix_positions,
        state=prefix_result.updated_state,
        return_activation_trace=True,
    )
    assert multi_result.activation_trace is not None

    assert_close(
        result=multi_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="sliding-window dynamic-state multi-token logits vs sequential",
    )
    assert_close(
        result=multi_result.activation_trace.output_norm[0],
        reference=jnp.stack(sequential_hidden),
        operation_name="sliding-window dynamic-state multi-token hidden vs sequential",
    )
