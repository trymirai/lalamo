from collections.abc import Callable

import jax
import jax.numpy as jnp

from lalamo.modules import (
    Attention,
    AttentionConfig,
    Decoder,
    DecoderActivationTrace,
    DecoderConfig,
    DecoderResult,
    DenseMLPConfig,
    DynamicKVCacheLayer,
    ForwardPassMode,
    FullPrecisionLinearConfig,
    Identity,
    NormalizationConfig,
    StaticKVCacheLayer,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerActivationTrace,
    TransformerLayerConfig,
    TransformerLayerResult,
    UnscaledRoPEConfig,
    UpcastMode,
)
from tests.common import assert_close


def make_attention(*, sliding_window_size: int | None = None) -> Attention:
    config = AttentionConfig(
        qkv_projection_config=FullPrecisionLinearConfig(precision=jnp.float32),
        out_projection_config=FullPrecisionLinearConfig(precision=jnp.float32),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=4,
        is_causal=True,
        scale=None,
        sliding_window_size=sliding_window_size,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )
    return config.random_init(model_dim=8, key=jax.random.key(0))


def make_decoder(
    *,
    use_rope: bool,
    num_layers: int = 2,
    sliding_window_size: int | None = None,
) -> Decoder:
    precision = jnp.float32
    model_dim = 8
    hidden_dim = 16
    vocab_size = 32
    context_length = 32

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
        sliding_window_size=sliding_window_size,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        use_rope=use_rope,
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
        )
        if use_rope
        else None,
        local_rope_config=None,
        layer_configs=tuple(layer_config for _ in range(num_layers)),
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


def _stack_last_token(
    branch_results: list[DecoderResult],
    extractor: Callable[[DecoderResult], jnp.ndarray],
) -> jnp.ndarray:
    return jnp.stack([extractor(result) for result in branch_results])


def _decoder_activation_trace(result: DecoderResult) -> DecoderActivationTrace:
    assert result.activation_trace is not None
    return result.activation_trace


def _layer_trace(
    result: DecoderResult,
    layer_index: int,
) -> tuple[TransformerLayerResult, TransformerLayerActivationTrace]:
    activation_trace = _decoder_activation_trace(result)
    layer_result = activation_trace.layer_results[layer_index]
    assert layer_result.activation_trace is not None
    return layer_result, layer_result.activation_trace


def _build_static_cache(prefix_length: int, suffix_length: int) -> tuple[StaticKVCacheLayer, StaticKVCacheLayer]:
    empty_cache = StaticKVCacheLayer.init(
        has_sinks=False,
        capacity=prefix_length + suffix_length + 4,
        num_groups=1,
        head_dim=1,
        dtype=jnp.float32,
    )
    prefix_cache = empty_cache.extend(
        jnp.zeros((prefix_length, 1, 1), dtype=jnp.float32),
        jnp.zeros((prefix_length, 1, 1), dtype=jnp.float32),
    )
    full_cache = prefix_cache.extend(
        jnp.zeros((suffix_length, 1, 1), dtype=jnp.float32),
        jnp.zeros((suffix_length, 1, 1), dtype=jnp.float32),
    )
    return prefix_cache, full_cache


def _tree_paths(prefix_length: int, parent_indices: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for node_index, parent_index in enumerate(parent_indices):
        if parent_index < prefix_length:
            result.append([node_index])
        else:
            parent_node = parent_index - prefix_length
            result.append([*result[parent_node], node_index])
    return result


def _run_sequential_multi_tree_chain(
    *,
    decoder: Decoder,
    prefix_token_ids: jnp.ndarray,
    prefix_positions: jnp.ndarray,
    suffix_token_ids: jnp.ndarray,
    suffix_positions: jnp.ndarray,
    parent_indices: jnp.ndarray,
    initial_state: object | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, DecoderResult, DecoderResult]:
    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=initial_state,
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

    tree_result = decoder(
        suffix_token_ids,
        suffix_positions,
        state=prefix_result.updated_state,
        return_activation_trace=True,
        attention_parent_indices=parent_indices,
    )
    assert tree_result.activation_trace is not None
    return jnp.stack(sequential_logits), jnp.stack(sequential_hidden), multi_result, tree_result


def test_static_kv_cache_tree_attention_mask_matches_expected_topology() -> None:
    prefix_cache, full_cache = _build_static_cache(prefix_length=4, suffix_length=4)
    parent_indices = jnp.array([3, 4, 4, 6], dtype=jnp.int32)

    mask = full_cache.tree_attention_mask(prefix_length=prefix_cache.length, parent_indices=parent_indices)

    expected = jnp.array(
        [
            [True, True, True, True, True, False, False, False, False, False, False, False],
            [True, True, True, True, True, True, False, False, False, False, False, False],
            [True, True, True, True, True, False, True, False, False, False, False, False],
            [True, True, True, True, True, False, True, True, False, False, False, False],
        ],
        dtype=jnp.bool,
    )

    assert jnp.array_equal(mask, expected)


def test_static_kv_cache_tree_attention_mask_respects_sliding_window() -> None:
    prefix_cache, full_cache = _build_static_cache(prefix_length=4, suffix_length=4)
    parent_indices = jnp.array([3, 4, 4, 6], dtype=jnp.int32)

    mask = full_cache.tree_attention_mask(
        prefix_length=prefix_cache.length,
        parent_indices=parent_indices,
        sliding_window_size=2,
    )

    expected = jnp.array(
        [
            [False, False, False, True, True, False, False, False, False, False, False, False],
            [False, False, False, False, True, True, False, False, False, False, False, False],
            [False, False, False, False, True, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, True, True, False, False, False, False],
        ],
        dtype=jnp.bool,
    )

    assert jnp.array_equal(mask, expected)


def test_dynamic_kv_cache_causal_sliding_window_mask_matches_static_mask() -> None:
    prefix_length = 3
    suffix_length = 4
    sliding_window_size = 2

    static_cache = StaticKVCacheLayer.init(
        has_sinks=False,
        capacity=prefix_length + suffix_length,
        num_groups=1,
        head_dim=1,
        dtype=jnp.float32,
    )
    dynamic_cache_layer = DynamicKVCacheLayer.init(
        has_sinks=False,
        keys=jnp.zeros((prefix_length + suffix_length, 1, 1), dtype=jnp.float32),
        values=jnp.zeros((prefix_length + suffix_length, 1, 1), dtype=jnp.float32),
    )
    static_cache_layer = StaticKVCacheLayer(
        has_sinks=static_cache.has_sinks,
        keys=static_cache.keys,
        values=static_cache.values,
        current_length=jnp.array(prefix_length + suffix_length, dtype=jnp.int32),
    )

    dynamic_mask = dynamic_cache_layer.attention_mask(
        suffix_length=suffix_length,
        is_causal=True,
        sliding_window_size=sliding_window_size,
    )
    static_mask = static_cache_layer.attention_mask(
        suffix_length=suffix_length,
        is_causal=True,
        sliding_window_size=sliding_window_size,
    )

    assert jnp.array_equal(dynamic_mask, static_mask)


def test_tree_attention_matches_branchwise_reference() -> None:
    attention = make_attention()
    prefix_inputs = jax.random.normal(jax.random.key(2), (3, attention.model_dim), dtype=jnp.float32)
    prefix_result = attention(prefix_inputs, positional_embeddings=None, state=None, return_updated_state=True)
    assert prefix_result.state is not None

    tree_inputs = jax.random.normal(jax.random.key(3), (4, attention.model_dim), dtype=jnp.float32)
    parent_indices = jnp.array([2, 3, 3, 5], dtype=jnp.int32)

    tree_outputs = attention(
        tree_inputs,
        positional_embeddings=None,
        state=prefix_result.state,
        attention_parent_indices=parent_indices,
    ).outputs

    branch_reference_outputs = jnp.stack(
        [
            attention(
                tree_inputs[jnp.array([0])],
                positional_embeddings=None,
                state=prefix_result.state,
            ).outputs[-1],
            attention(
                tree_inputs[jnp.array([0, 1])],
                positional_embeddings=None,
                state=prefix_result.state,
            ).outputs[-1],
            attention(
                tree_inputs[jnp.array([0, 2])],
                positional_embeddings=None,
                state=prefix_result.state,
            ).outputs[-1],
            attention(
                tree_inputs[jnp.array([0, 2, 3])],
                positional_embeddings=None,
                state=prefix_result.state,
            ).outputs[-1],
        ],
    )

    assert_close(
        result=tree_outputs,
        reference=branch_reference_outputs,
        operation_name="tree attention vs branchwise reference",
    )


def _assert_decoder_tree_matches_branchwise_reference(
    *,
    decoder: Decoder,
    prefix_token_ids: jnp.ndarray,
    prefix_positions: jnp.ndarray,
    tree_token_ids: jnp.ndarray,
    tree_positions: jnp.ndarray,
    parent_indices: jnp.ndarray,
    branch_inputs: list[tuple[jnp.ndarray, jnp.ndarray]],
    lengths_without_padding: jnp.ndarray | None = None,
    use_static_prefix_state: bool = False,
) -> None:
    initial_state = None
    if use_static_prefix_state:
        batch_size, prefix_length = prefix_token_ids.shape
        suffix_length = tree_token_ids.shape[1]
        initial_state = decoder.init_static_state(
            batch_size=batch_size,
            capacity=prefix_length + suffix_length + 4,
        )

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=initial_state,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    tree_result = decoder(
        tree_token_ids,
        tree_positions,
        state=prefix_result.updated_state,
        return_activation_trace=True,
        attention_parent_indices=parent_indices,
        lengths_without_padding=lengths_without_padding,
    )
    assert tree_result.activation_trace is not None

    branch_results = [
        decoder(
            token_ids,
            token_positions,
            state=prefix_result.updated_state,
            return_activation_trace=True,
        )
        for token_ids, token_positions in branch_inputs
    ]

    if lengths_without_padding is None:
        valid_suffix_length = tree_token_ids.shape[1]
    else:
        valid_suffix_length = int(lengths_without_padding[0])
    valid_slice = slice(0, valid_suffix_length)

    assert_close(
        result=tree_result.logits[0, valid_slice],
        reference=_stack_last_token(branch_results, lambda result: result.logits[0, -1]),
        operation_name="decoder tree logits vs branchwise reference",
    )
    assert_close(
        result=tree_result.activation_trace.output_norm[0, valid_slice],
        reference=_stack_last_token(
            branch_results,
            lambda result: _decoder_activation_trace(result).output_norm[0, -1],
        ),
        operation_name="decoder tree final hidden states vs branchwise reference",
    )

    for layer_index, tree_layer_result in enumerate(tree_result.activation_trace.layer_results):
        assert tree_layer_result.activation_trace is not None
        trace = tree_layer_result.activation_trace

        assert_close(
            result=tree_layer_result.outputs[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[0].outputs[0, -1],
            ),
            operation_name=f"layer {layer_index} outputs vs branchwise reference",
        )
        assert_close(
            result=trace.inputs[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].inputs[0, -1],
            ),
            operation_name=f"layer {layer_index} inputs vs branchwise reference",
        )
        assert_close(
            result=trace.pre_mixer_norm[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].pre_mixer_norm[0, -1],
            ),
            operation_name=f"layer {layer_index} pre_mixer_norm vs branchwise reference",
        )
        assert_close(
            result=trace.mixer[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].mixer[0, -1],
            ),
            operation_name=f"layer {layer_index} mixer outputs vs branchwise reference",
        )
        assert_close(
            result=trace.mlp_inputs[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].mlp_inputs[0, -1],
            ),
            operation_name=f"layer {layer_index} mlp_inputs vs branchwise reference",
        )
        assert_close(
            result=trace.pre_mlp_norm[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].pre_mlp_norm[0, -1],
            ),
            operation_name=f"layer {layer_index} pre_mlp_norm vs branchwise reference",
        )
        assert_close(
            result=trace.mlp[0, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result, layer_index=layer_index: _layer_trace(result, layer_index)[1].mlp[0, -1],
            ),
            operation_name=f"layer {layer_index} mlp outputs vs branchwise reference",
        )


def test_decoder_tree_hidden_states_and_logits_match_branchwise_reference() -> None:
    _assert_decoder_tree_matches_branchwise_reference(
        decoder=make_decoder(use_rope=True, num_layers=2),
        prefix_token_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        prefix_positions=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        tree_token_ids=jnp.array([[4, 5, 6, 7]], dtype=jnp.int32),
        tree_positions=jnp.array([[3, 4, 4, 5]], dtype=jnp.int32),
        parent_indices=jnp.array([[2, 3, 3, 5]], dtype=jnp.int32),
        branch_inputs=[
            (jnp.array([[4]], dtype=jnp.int32), jnp.array([[3]], dtype=jnp.int32)),
            (jnp.array([[4, 5]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6, 7]], dtype=jnp.int32), jnp.array([[3, 4, 5]], dtype=jnp.int32)),
        ],
    )


def test_decoder_tree_hidden_states_with_sliding_window_match_branchwise_reference() -> None:
    _assert_decoder_tree_matches_branchwise_reference(
        decoder=make_decoder(use_rope=True, num_layers=2, sliding_window_size=2),
        prefix_token_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        prefix_positions=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        tree_token_ids=jnp.array([[4, 5, 6, 7]], dtype=jnp.int32),
        tree_positions=jnp.array([[3, 4, 4, 5]], dtype=jnp.int32),
        parent_indices=jnp.array([[2, 3, 3, 5]], dtype=jnp.int32),
        branch_inputs=[
            (jnp.array([[4]], dtype=jnp.int32), jnp.array([[3]], dtype=jnp.int32)),
            (jnp.array([[4, 5]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6, 7]], dtype=jnp.int32), jnp.array([[3, 4, 5]], dtype=jnp.int32)),
        ],
        use_static_prefix_state=True,
    )


def test_decoder_tree_hidden_states_support_batched_padding() -> None:
    decoder = make_decoder(use_rope=True, num_layers=2)
    prefix_token_ids = jnp.array([[1, 2, 3], [8, 9, 10]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2], [0, 1, 2]], dtype=jnp.int32)
    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    tree_token_ids = jnp.array([[4, 5, 6, 7], [11, 12, 0, 0]], dtype=jnp.int32)
    tree_positions = jnp.array([[3, 4, 4, 5], [3, 4, 0, 0]], dtype=jnp.int32)
    parent_indices = jnp.array([[2, 3, 3, 5], [2, 3, 0, 0]], dtype=jnp.int32)
    lengths_without_padding = jnp.array([4, 2], dtype=jnp.int32)

    tree_result = decoder(
        tree_token_ids,
        tree_positions,
        state=prefix_result.updated_state,
        return_activation_trace=True,
        attention_parent_indices=parent_indices,
        lengths_without_padding=lengths_without_padding,
    )
    assert tree_result.activation_trace is not None

    expected_paths = [
        [
            (jnp.array([[4]], dtype=jnp.int32), jnp.array([[3]], dtype=jnp.int32)),
            (jnp.array([[4, 5]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
            (jnp.array([[4, 6, 7]], dtype=jnp.int32), jnp.array([[3, 4, 5]], dtype=jnp.int32)),
        ],
        [
            (jnp.array([[11]], dtype=jnp.int32), jnp.array([[3]], dtype=jnp.int32)),
            (jnp.array([[11, 12]], dtype=jnp.int32), jnp.array([[3, 4]], dtype=jnp.int32)),
        ],
    ]

    for batch_index, branch_inputs in enumerate(expected_paths):
        branch_state = jax.tree.map(
            lambda value, batch_index=batch_index: value[batch_index : batch_index + 1],
            prefix_result.updated_state,
        )
        branch_results = [
            decoder(
                token_ids,
                token_positions,
                state=branch_state,
                return_activation_trace=True,
            )
            for token_ids, token_positions in branch_inputs
        ]

        valid_slice = slice(0, int(lengths_without_padding[batch_index]))
        assert_close(
            result=tree_result.logits[batch_index, valid_slice],
            reference=_stack_last_token(branch_results, lambda result: result.logits[0, -1]),
            operation_name=f"batched tree logits batch={batch_index}",
        )
        assert_close(
            result=tree_result.activation_trace.output_norm[batch_index, valid_slice],
            reference=_stack_last_token(
                branch_results,
                lambda result: _decoder_activation_trace(result).output_norm[0, -1],
            ),
            operation_name=f"batched tree hidden states batch={batch_index}",
        )


def test_tree_attention_matches_single_token_sequential_large_tree() -> None:
    decoder = make_decoder(use_rope=True, num_layers=2)
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    parent_indices = [2, *[3 + index for index in range(64 - 1)]]
    tree_token_ids = jnp.arange(4, 4 + 64, dtype=jnp.int32)[None, :]
    tree_positions = jnp.arange(3, 3 + 64, dtype=jnp.int32)[None, :]
    tree_result = decoder(
        tree_token_ids,
        tree_positions,
        state=prefix_result.updated_state,
        return_activation_trace=True,
        attention_parent_indices=jnp.array([parent_indices], dtype=jnp.int32),
    )
    assert tree_result.activation_trace is not None

    sequential_logits = []
    sequential_hidden = []
    state = prefix_result.updated_state
    for token_id, token_position in zip(tree_token_ids[0], tree_positions[0], strict=True):
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

    assert_close(
        result=tree_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="tree logits vs single-token sequential, large chain",
    )
    assert_close(
        result=tree_result.activation_trace.output_norm[0],
        reference=jnp.stack(sequential_hidden),
        operation_name="tree hidden states vs single-token sequential, large chain",
    )


def test_sliding_window_tree_attention_matches_single_token_sequential_large_tree() -> None:
    decoder = make_decoder(use_rope=True, num_layers=2, sliding_window_size=2)
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    initial_state = decoder.init_static_state(batch_size=1, capacity=3 + 64 + 4)
    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=initial_state,
        return_updated_state=True,
    )
    assert prefix_result.updated_state is not None

    parent_indices = [2]
    parent_indices.extend(3 + node_index // 2 for node_index in range(63))
    tree_token_values = jnp.arange(4, 4 + 64, dtype=jnp.int32)
    tree_position_values = jnp.array([parent + 1 for parent in parent_indices], dtype=jnp.int32)
    tree_result = decoder(
        tree_token_values[None, :],
        tree_position_values[None, :],
        state=prefix_result.updated_state,
        return_activation_trace=True,
        attention_parent_indices=jnp.array([parent_indices], dtype=jnp.int32),
    )
    assert tree_result.activation_trace is not None

    sequential_logits = []
    sequential_hidden = []
    for path in _tree_paths(3, parent_indices):
        state = prefix_result.updated_state
        step_result = None
        for node_index in path:
            step_result = decoder(
                tree_token_values[node_index][None, None],
                tree_position_values[node_index][None, None],
                state=state,
                return_updated_state=True,
                return_activation_trace=True,
                forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            )
            assert step_result.updated_state is not None
            state = step_result.updated_state
        assert step_result is not None
        assert step_result.activation_trace is not None
        sequential_logits.append(step_result.logits[0, -1])
        sequential_hidden.append(step_result.activation_trace.output_norm[0, -1])

    assert_close(
        result=tree_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="sliding-window tree logits vs single-token sequential, large tree",
    )
    assert_close(
        result=tree_result.activation_trace.output_norm[0],
        reference=jnp.stack(sequential_hidden),
        operation_name="sliding-window tree hidden states vs single-token sequential, large tree",
    )


def test_sliding_window_dynamic_state_multi_token_matches_single_token_sequential_large_chain() -> None:
    suffix_length = 64
    sequential_logits, sequential_hidden, multi_result, _ = _run_sequential_multi_tree_chain(
        decoder=make_decoder(use_rope=True, num_layers=2, sliding_window_size=2),
        prefix_token_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        prefix_positions=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        suffix_token_ids=jnp.arange(4, 4 + suffix_length, dtype=jnp.int32)[None, :],
        suffix_positions=jnp.arange(3, 3 + suffix_length, dtype=jnp.int32)[None, :],
        parent_indices=jnp.array([[2, *[3 + index for index in range(suffix_length - 1)]]], dtype=jnp.int32),
    )

    assert_close(
        result=multi_result.logits[0],
        reference=sequential_logits,
        operation_name="sliding-window dynamic-state multi-token logits vs single-token sequential",
    )
    assert_close(
        result=multi_result.activation_trace.output_norm[0],
        reference=sequential_hidden,
        operation_name="sliding-window dynamic-state multi-token hidden states vs single-token sequential",
    )


def test_sliding_window_dynamic_state_tree_attention_matches_single_token_sequential_large_chain() -> None:
    suffix_length = 64
    sequential_logits, sequential_hidden, _, tree_result = _run_sequential_multi_tree_chain(
        decoder=make_decoder(use_rope=True, num_layers=2, sliding_window_size=2),
        prefix_token_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        prefix_positions=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        suffix_token_ids=jnp.arange(4, 4 + suffix_length, dtype=jnp.int32)[None, :],
        suffix_positions=jnp.arange(3, 3 + suffix_length, dtype=jnp.int32)[None, :],
        parent_indices=jnp.array([[2, *[3 + index for index in range(suffix_length - 1)]]], dtype=jnp.int32),
    )

    assert_close(
        result=tree_result.logits[0],
        reference=sequential_logits,
        operation_name="sliding-window dynamic-state tree logits vs single-token sequential",
    )
    assert_close(
        result=tree_result.activation_trace.output_norm[0],
        reference=sequential_hidden,
        operation_name="sliding-window dynamic-state tree hidden states vs single-token sequential",
    )
