from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.tree_util import keystr

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillBatchMetrics,
    DistillOptimizerState,
    DistillTrainConfig,
    DistillTrainingState,
    OptimizerGroup,
    TraceDistillBatch,
    apply_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    compute_trace_distill_batch_metrics,
    compute_trace_distill_step_gradients,
    get_optimizer_group,
    initialize_distill_training_state,
    inject_lora_adapter_configs,
    inject_lora_adapters,
    is_leaf_trainable,
    lora_trainable_filter,
    make_trace_distill_batch,
    materialize_trainable_module,
    stochastically_quantize_module,
    summarize_distill_parameters,
)
from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.modules.common import (
    ParameterNorm,
    combine_parameter_leaves,
    field,
    find_field_metadata,
    iter_parameter_leaves,
    partition_parameter_leaves,
)
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import (
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    QLoRALinear,
    QLoRALinearConfig,
)
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.quantization import QuantizationMode


class AliasModule(eqx.Module):
    left: jax.Array = field()
    right: jax.Array = field()


class BufferModule(eqx.Module):
    parameter: jax.Array = field()
    buffer: jax.Array = field(trainable=False, norm=ParameterNorm.L_INF)


class FrozenModule(eqx.Module):
    left: jax.Array = field(trainable=False)
    right: jax.Array = field(trainable=False, norm=ParameterNorm.L_INF)


class NestedAliasModule(eqx.Module):
    blocks: tuple[AliasModule, ...]


class DictAliasModule(eqx.Module):
    blocks: dict[str, AliasModule]


@dataclass(frozen=True)
class LinearConfigWrapper:
    primary: GroupQuantizedLinearConfig
    secondary: tuple[GroupQuantizedLinearConfig, ...]
    tertiary: list[GroupQuantizedLinearConfig]


def _make_optimizer_state(
    training_state: DistillTrainingState,
    optimizer: optax.GradientTransformation,
) -> DistillOptimizerState:
    return DistillOptimizerState(
        training_state=training_state,
        optimizer_state=optimizer.init(training_state.master_weights),
    )


def _run_distill_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillTrainConfig,
) -> tuple[DistillOptimizerState, DistillBatchMetrics]:
    grads, metrics = compute_distill_step_gradients(
        optimizer_state.training_state,
        student,
        teacher,
        batch,
        config,
    )
    return apply_distill_gradients(optimizer_state, optimizer, grads), metrics


def _run_trace_distill_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    student: Decoder,
    batch: TraceDistillBatch,
    config: DistillTrainConfig,
) -> tuple[DistillOptimizerState, DistillBatchMetrics]:
    grads, metrics = compute_trace_distill_step_gradients(
        optimizer_state.training_state,
        student,
        batch,
        config,
    )
    return apply_distill_gradients(optimizer_state, optimizer, grads), metrics


def _leaf_by_path(tree: object) -> dict[str, object]:
    flat_with_path, _ = jtu.tree_flatten_with_path(
        tree,
        is_leaf=lambda leaf: leaf is None or isinstance(leaf, optax.contrib.MuonDimensionNumbers),
    )
    return {keystr(path).lstrip("."): leaf for path, leaf in flat_with_path}


def _selected_leaf_by_path(tree: object) -> dict[str, object]:
    return {path: leaf for path, leaf in _leaf_by_path(tree).items() if leaf is not None}


def test_iter_parameter_leaves_detects_aliases() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    leaves = iter_parameter_leaves(module)

    assert [leaf.path for leaf in leaves] == ["left", "right"]
    assert leaves[0].alias_of is None
    assert leaves[1].alias_of == "left"


def test_summarize_distill_parameters_counts_unique_aliases_once() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    leaves = iter_parameter_leaves(module)
    summary = summarize_distill_parameters(leaves, DistillTrainConfig())

    assert summary.total_parameters == 4
    assert summary.trainable_parameters == 4
    assert summary.total_master_bytes == 16
    assert summary.by_group == {OptimizerGroup.MUON: 4}


def test_initialize_distill_training_state_dedupes_alias_masters() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    state = initialize_distill_training_state(
        module,
        DistillTrainConfig(compute_dtype=jnp.bfloat16),
    )

    master_weights = _selected_leaf_by_path(state.master_weights)
    assert set(master_weights) == {"left"}
    assert master_weights["left"].dtype == jnp.float32

    materialized = materialize_trainable_module(
        module,
        state.master_weights,
        DistillTrainConfig(compute_dtype=jnp.bfloat16),
    )
    materialized_leaves = iter_parameter_leaves(materialized)

    assert materialized.left.dtype == jnp.bfloat16
    assert materialized.right.dtype == jnp.bfloat16
    assert materialized_leaves[1].alias_of == "left"


def test_find_field_metadata_tracks_nested_container_fields() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = NestedAliasModule(blocks=(AliasModule(left=shared, right=shared),))

    field_info = find_field_metadata(module, module.blocks[0].left)

    assert field_info is not None
    assert field_info.owner is module.blocks[0]
    assert field_info.field.name == "left"


def test_q_lora_policy_prefers_adapter_weights() -> None:
    config = QLoRALinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
        lora_rank=2,
        lora_scale=1.0,
    )
    layer = config.random_init(4, (4,), has_biases=True, key=jax.random.key(0))
    leaves = {leaf.field_name: leaf for leaf in iter_parameter_leaves(layer)}

    assert is_leaf_trainable(leaves["lora_down_weights"])
    assert is_leaf_trainable(leaves["lora_up_weights"])
    assert is_leaf_trainable(leaves["weights"])
    assert is_leaf_trainable(leaves["scales"])
    assert get_optimizer_group(leaves["weights"]) == OptimizerGroup.MUON
    assert get_optimizer_group(leaves["lora_down_weights"]) == OptimizerGroup.MUON


def test_q_lora_linear_config_empty_builds_q_lora_linear() -> None:
    config = QLoRALinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
        lora_rank=2,
        lora_scale=1.0,
    )

    layer = config.empty(4, (4,), has_biases=True)

    assert isinstance(layer, QLoRALinear)
    assert layer.lora_down_weights.shape == (2, 4)
    assert layer.lora_up_weights.shape == (4, 2)


def test_quant_aux_is_trainable_by_default() -> None:
    config = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    )
    layer = config.random_init(4, (4,), has_biases=True, key=jax.random.key(1))
    leaves = {leaf.field_name: leaf for leaf in iter_parameter_leaves(layer)}

    assert is_leaf_trainable(leaves["scales"])
    assert is_leaf_trainable(leaves["zero_points"])
    assert get_optimizer_group(leaves["scales"]) == OptimizerGroup.DEFAULT


def test_materialize_trainable_module_casts_only_trainable_leaves() -> None:
    module = BufferModule(
        parameter=jnp.ones((2, 2), dtype=jnp.float32),
        buffer=jnp.ones((4,), dtype=jnp.float32),
    )

    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.bfloat16)
    state = initialize_distill_training_state(module, config)
    materialized = materialize_trainable_module(module, state.master_weights, config)

    master_weights = _selected_leaf_by_path(state.master_weights)
    assert set(master_weights) == {"parameter"}
    assert master_weights["parameter"].dtype == jnp.float32
    assert materialized.parameter.dtype == jnp.bfloat16
    assert materialized.buffer.dtype == jnp.float32


def test_stochastically_quantize_module_quantizes_quantized_weights() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(14))
    layer = eqx.tree_at(
        lambda current_layer: current_layer.weights,
        layer,
        jnp.full_like(layer.weights, 1.25),
    )

    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32)
    state = initialize_distill_training_state(layer, config)
    materialized = materialize_trainable_module(layer, state.master_weights, config)
    first = stochastically_quantize_module(materialized, QuantizationMode.UINT4, jax.random.key(0))
    second = stochastically_quantize_module(materialized, QuantizationMode.UINT4, jax.random.key(0))

    assert jnp.array_equal(first.weights, second.weights)
    assert jnp.all(jnp.logical_or(first.weights == 1.0, first.weights == 2.0))


def test_initialize_distill_training_state_tracks_quant_aux_paths() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(8))

    config = DistillTrainConfig()
    state = initialize_distill_training_state(layer, config)

    assert set(_selected_leaf_by_path(state.master_weights)) == {
        "biases",
        "scales",
        "weights",
        "zero_points",
    }
    assert set(_selected_leaf_by_path(state.muon_weight_dimension_numbers)) == {"weights"}


def test_trainable_filter_narrows_to_quantized_only() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(20))

    config = DistillTrainConfig()
    full_state = initialize_distill_training_state(layer, config)
    filtered_state = initialize_distill_training_state(
        layer,
        config,
        trainable_filter=lambda info: info.quantized,
    )

    assert set(_selected_leaf_by_path(full_state.master_weights)) == {
        "biases",
        "scales",
        "weights",
        "zero_points",
    }
    assert set(_selected_leaf_by_path(filtered_state.master_weights)) == {"weights"}

    full_summary = summarize_distill_parameters(
        iter_parameter_leaves(layer),
        config,
    )
    filtered_summary = summarize_distill_parameters(
        iter_parameter_leaves(layer),
        config,
        trainable_filter=lambda info: info.quantized,
    )
    assert filtered_summary.trainable_parameters < full_summary.trainable_parameters
    assert filtered_summary.total_parameters == full_summary.total_parameters


def test_inject_lora_adapters_wraps_group_quantized_linear_without_changing_base_weights() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (3, 5), has_biases=True, key=jax.random.key(21))
    injected_layer = inject_lora_adapters(layer, 2, 0.5, jax.random.key(22))

    assert isinstance(injected_layer, QLoRALinear)
    assert injected_layer.config.lora_rank == 2
    assert injected_layer.config.lora_scale == 0.5
    assert jnp.array_equal(injected_layer.weights, layer.weights)
    assert jnp.array_equal(injected_layer.scales, layer.scales)
    assert injected_layer.lora_down_weights.shape == (2, 4)
    assert injected_layer.lora_up_weights.shape == (8, 2)
    assert jnp.all(injected_layer.lora_up_weights == 0)


def test_inject_lora_adapter_configs_wraps_group_quantized_linear_configs() -> None:
    linear_config = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    )
    config = LinearConfigWrapper(
        primary=linear_config,
        secondary=(linear_config,),
        tertiary=[linear_config],
    )

    injected_config = inject_lora_adapter_configs(config, 2, 0.5)

    assert isinstance(injected_config.primary, QLoRALinearConfig)
    assert isinstance(injected_config.secondary[0], QLoRALinearConfig)
    assert isinstance(injected_config.tertiary[0], QLoRALinearConfig)
    assert injected_config.primary.lora_rank == 2
    assert injected_config.primary.lora_scale == 0.5
    assert injected_config.primary.weight_quantization_mode == linear_config.weight_quantization_mode


def test_lora_trainable_filter_keeps_only_lora_and_auxiliary_leaves() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(23))
    injected_layer = inject_lora_adapters(layer, 2, 1.0, jax.random.key(24))

    state = initialize_distill_training_state(
        injected_layer,
        DistillTrainConfig(),
        trainable_filter=lora_trainable_filter,
    )

    assert set(_selected_leaf_by_path(state.master_weights)) == {
        "lora_down_weights",
        "lora_up_weights",
    }


def test_q_lora_import_weights_round_trips_without_transpose() -> None:
    config = QLoRALinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
        lora_rank=2,
        lora_scale=1.0,
    )
    layer = config.random_init(4, (3, 5), has_biases=True, key=jax.random.key(26))
    exported_weights = layer.export_weights()

    assert isinstance(exported_weights, dict)
    assert exported_weights["down_weights"].shape == (2, 4)
    assert exported_weights["up_weights"].shape == (8, 2)
    assert jnp.array_equal(exported_weights["down_weights"], layer.lora_down_weights)
    assert jnp.array_equal(exported_weights["up_weights"], layer.lora_up_weights)

    imported_layer = layer.import_weights(exported_weights)

    assert jnp.array_equal(imported_layer.lora_down_weights, layer.lora_down_weights)
    assert jnp.array_equal(imported_layer.lora_up_weights, layer.lora_up_weights)


def test_q_lora_linear_uses_single_fused_lora_path() -> None:
    config = QLoRALinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
        lora_rank=2,
        lora_scale=0.5,
    )
    layer = config.random_init(4, (2, 1), has_biases=True, key=jax.random.key(27))
    layer = eqx.tree_at(
        lambda current_layer: (
            current_layer.weights,
            current_layer.zero_points,
            current_layer.scales,
            current_layer.biases,
            current_layer.lora_down_weights,
            current_layer.lora_up_weights,
        ),
        layer,
        (
            jnp.zeros_like(layer.weights),
            jnp.zeros_like(layer.zero_points),
            jnp.ones_like(layer.scales),
            jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            jnp.array(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                ],
                dtype=jnp.float32,
            ),
        ),
    )

    outputs = layer(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32))

    assert len(outputs) == 2
    assert jnp.allclose(outputs[0], jnp.array([13.0, 18.5], dtype=jnp.float32))
    assert jnp.allclose(outputs[1], jnp.array([24.0], dtype=jnp.float32))


def test_muon_weight_dimension_numbers_match_optimizer_groups() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(11))

    state = initialize_distill_training_state(
        layer,
        DistillTrainConfig(),
    )

    assert set(_selected_leaf_by_path(state.master_weights)) == {
        "biases",
        "scales",
        "weights",
        "zero_points",
    }
    dimension_numbers_by_path = _leaf_by_path(state.muon_weight_dimension_numbers)
    assert dimension_numbers_by_path["weights"] == optax.contrib.MuonDimensionNumbers(
        reduction_axis=1,
        output_axis=0,
    )
    assert dimension_numbers_by_path["biases"] is None
    assert dimension_numbers_by_path["scales"] is None
    assert dimension_numbers_by_path["zero_points"] is None


def test_separable_causal_conv_weights_do_not_use_muon() -> None:
    layer = SeparableCausalConvConfig(
        precision=jnp.float32,
        has_biases=True,
    ).random_init(8, 4, key=jax.random.key(13))

    leaves = {leaf.field_name: leaf for leaf in iter_parameter_leaves(layer)}
    state = initialize_distill_training_state(layer, DistillTrainConfig())

    assert is_leaf_trainable(leaves["weights"])
    assert get_optimizer_group(leaves["weights"]) == OptimizerGroup.DEFAULT
    assert all(value is None for value in _leaf_by_path(state.muon_weight_dimension_numbers).values())


def test_iter_parameter_leaves_defaults_non_static_arrays_to_trainable_spectral() -> None:
    module = BufferModule(
        parameter=jnp.ones((2, 2), dtype=jnp.float32),
        buffer=jnp.ones((4,), dtype=jnp.float32),
    )
    leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(module)}

    assert leaves["parameter"].trainable
    assert leaves["parameter"].norm == ParameterNorm.SPECTRAL
    assert not leaves["buffer"].trainable
    assert leaves["buffer"].norm == ParameterNorm.L_INF


_TINY_LLAMA_CONFIG = HFLlamaConfig(
    torch_dtype="float32",
    architectures=["LlamaForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=32,
    initializer_range=0.02,
    intermediate_size=64,
    max_position_embeddings=64,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=4,
    num_hidden_layers=1,
    num_key_value_heads=2,
    pretraining_tp=1,
    rms_norm_eps=1e-5,
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    transformers_version="4.0.0",
    use_cache=True,
    vocab_size=32,
)


def _make_tiny_llama_decoder(*, key: jax.Array) -> Decoder:
    decoder_config = _TINY_LLAMA_CONFIG.to_decoder_config(
        context_length=64,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )
    return decoder_config.random_init(key=key)


def test_llama_decoder_marks_rope_tables_frozen() -> None:
    decoder_config = _TINY_LLAMA_CONFIG.to_decoder_config(
        context_length=64,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )
    decoder = decoder_config.empty()
    leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(decoder)}

    assert not leaves["transformer.global_rope.cosines"].trainable
    assert not leaves["transformer.global_rope.sines"].trainable
    assert leaves["transformer.global_rope.cosines"].norm == ParameterNorm.L_INF
    assert leaves["transformer.global_rope.sines"].norm == ParameterNorm.L_INF


def test_compute_distill_batch_metrics_has_zero_loss_for_matching_decoders() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(9))
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([6], dtype=jnp.int32),
    )

    metrics = compute_distill_batch_metrics(decoder, decoder, batch)

    assert metrics.valid_tokens == 5
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_compute_distill_batch_metrics_counts_all_matching_top1_tokens() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(14))
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([6], dtype=jnp.int32),
    )

    metrics = compute_distill_batch_metrics(decoder, decoder, batch)

    assert metrics.valid_tokens == 5
    assert metrics.top1_matches == 5
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_make_trace_distill_batch_pads_sequences_and_support() -> None:
    traces = (
        LalamoCompletion(
            prefix_token_ids=[1, 2],
            completion_token_ids=[3, 4],
            completion_token_logits=[{3: 0.5, 7: -1.0}, {4: 0.25}],
        ),
        LalamoCompletion(
            prefix_token_ids=[9],
            completion_token_ids=[8],
            completion_token_logits=[{8: 0.75, 6: -0.5, 5: -2.0}],
        ),
    )

    batch = make_trace_distill_batch(traces, pad_token_id=0)

    assert batch.token_ids.tolist() == [[1, 2, 3, 4], [9, 8, 0, 0]]
    assert batch.prefix_lengths.tolist() == [2, 1]
    assert batch.completion_lengths.tolist() == [2, 1]
    assert batch.support_token_ids.tolist() == [
        [[3, 7, 0], [4, 0, 0]],
        [[8, 6, 5], [0, 0, 0]],
    ]
    assert batch.support_mask.tolist() == [
        [[True, True, False], [True, False, False]],
        [[True, True, True], [False, False, False]],
    ]


def _make_trace_batch_from_decoder(decoder: Decoder) -> tuple[TraceDistillBatch, jax.Array]:
    token_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    decoder_logits = decoder(
        token_ids=token_ids,
        token_positions=jnp.arange(token_ids.shape[1], dtype=jnp.int32)[None, :],
    ).logits[:, :-1, :]
    completion_logits = jax.device_get(decoder_logits[0, 2:, :])

    def select_support(step_logits: jax.Array) -> dict[int, float]:
        support_values, support_ids = jax.lax.top_k(step_logits, 4)
        return dict(zip(support_ids.tolist(), support_values.tolist(), strict=True))

    traces = (
        LalamoCompletion(
            prefix_token_ids=[1, 2, 3],
            completion_token_ids=[4, 5],
            completion_token_logits=[select_support(step_logits) for step_logits in completion_logits],
        ),
    )
    return make_trace_distill_batch(traces, pad_token_id=0), token_ids


def test_compute_trace_distill_batch_metrics_has_zero_loss_for_matching_decoder() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(13))
    trace_batch, _ = _make_trace_batch_from_decoder(decoder)
    metrics = compute_trace_distill_batch_metrics(decoder, trace_batch)

    assert metrics.valid_tokens == 2
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_compute_trace_distill_batch_metrics_counts_all_matching_top1_tokens() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(15))
    trace_batch, _ = _make_trace_batch_from_decoder(decoder)

    metrics = compute_trace_distill_batch_metrics(decoder, trace_batch)

    assert metrics.valid_tokens == 2
    assert metrics.top1_matches == 2
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_repeated_distill_step_gradients_reduce_tiny_llama_loss() -> None:
    teacher = _make_tiny_llama_decoder(key=jax.random.key(10))
    student = eqx.tree_at(
        lambda decoder: decoder.transformer.layers[0].mixer.qkv_projection.weights,
        teacher,
        teacher.transformer.layers[0].mixer.qkv_projection.weights + 0.5,
    )
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([8], dtype=jnp.int32),
    )
    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32)
    optimizer = optax.sgd(0.05)
    training_state = initialize_distill_training_state(student, config)
    optimizer_state = _make_optimizer_state(training_state, optimizer)

    initial_metrics = compute_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        teacher,
        batch,
    )

    for _ in range(20):
        optimizer_state, _ = _run_distill_step(
            optimizer_state,
            optimizer,
            student,
            teacher,
            batch,
            config,
        )

    final_metrics = compute_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        teacher,
        batch,
    )

    assert initial_metrics.loss > 1e-4
    assert final_metrics.loss < initial_metrics.loss * 0.5


def test_applying_distill_step_gradients_matches_helper_step_update() -> None:
    teacher = _make_tiny_llama_decoder(key=jax.random.key(17))
    student = eqx.tree_at(
        lambda decoder: decoder.transformer.layers[0].mixer.qkv_projection.weights,
        teacher,
        teacher.transformer.layers[0].mixer.qkv_projection.weights + 0.5,
    )
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([8], dtype=jnp.int32),
    )
    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32)
    optimizer = optax.sgd(0.05)
    training_state = initialize_distill_training_state(student, config)
    optimizer_state = _make_optimizer_state(training_state, optimizer)

    step_optimizer_state, step_metrics = _run_distill_step(
        optimizer_state,
        optimizer,
        student,
        teacher,
        batch,
        config,
    )
    grads, grad_metrics = compute_distill_step_gradients(
        optimizer_state.training_state,
        student,
        teacher,
        batch,
        config,
    )
    applied_optimizer_state = apply_distill_gradients(optimizer_state, optimizer, grads)

    assert jnp.isclose(step_metrics.loss, grad_metrics.loss)
    assert step_metrics.valid_tokens == grad_metrics.valid_tokens

    step_weights = _selected_leaf_by_path(step_optimizer_state.training_state.master_weights)
    applied_weights = _selected_leaf_by_path(applied_optimizer_state.training_state.master_weights)
    assert set(step_weights) == set(applied_weights)
    for path in step_weights:
        step_weight = step_weights[path]
        applied_weight = applied_weights[path]
        assert jnp.allclose(step_weight, applied_weight)


def test_repeated_trace_distill_step_gradients_reduce_trace_loss() -> None:
    teacher = _make_tiny_llama_decoder(key=jax.random.key(16))
    student = eqx.tree_at(
        lambda decoder: decoder.transformer.layers[0].mixer.qkv_projection.weights,
        teacher,
        teacher.transformer.layers[0].mixer.qkv_projection.weights + 2.0,
    )
    trace_batch, _ = _make_trace_batch_from_decoder(teacher)
    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32)
    optimizer = optax.sgd(0.05)
    training_state = initialize_distill_training_state(student, config)
    optimizer_state = _make_optimizer_state(training_state, optimizer)

    initial_metrics = compute_trace_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        trace_batch,
    )

    for _ in range(20):
        optimizer_state, _ = _run_trace_distill_step(
            optimizer_state,
            optimizer,
            student,
            trace_batch,
            config,
        )

    final_metrics = compute_trace_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        trace_batch,
    )

    assert initial_metrics.loss > 1e-5
    assert final_metrics.loss < initial_metrics.loss


def test_applying_trace_distill_step_gradients_matches_helper_step_update() -> None:
    teacher = _make_tiny_llama_decoder(key=jax.random.key(18))
    student = eqx.tree_at(
        lambda decoder: decoder.transformer.layers[0].mixer.qkv_projection.weights,
        teacher,
        teacher.transformer.layers[0].mixer.qkv_projection.weights + 2.0,
    )
    trace_batch, _ = _make_trace_batch_from_decoder(teacher)
    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32)
    optimizer = optax.sgd(0.05)
    training_state = initialize_distill_training_state(student, config)
    optimizer_state = _make_optimizer_state(training_state, optimizer)

    step_optimizer_state, step_metrics = _run_trace_distill_step(
        optimizer_state,
        optimizer,
        student,
        trace_batch,
        config,
    )
    grads, grad_metrics = compute_trace_distill_step_gradients(
        optimizer_state.training_state,
        student,
        trace_batch,
        config,
    )
    applied_optimizer_state = apply_distill_gradients(optimizer_state, optimizer, grads)

    assert jnp.isclose(step_metrics.loss, grad_metrics.loss)
    assert step_metrics.valid_tokens == grad_metrics.valid_tokens

    step_weights = _selected_leaf_by_path(step_optimizer_state.training_state.master_weights)
    applied_weights = _selected_leaf_by_path(applied_optimizer_state.training_state.master_weights)
    assert set(step_weights) == set(applied_weights)
    for path in step_weights:
        step_weight = step_weights[path]
        applied_weight = applied_weights[path]
        assert jnp.allclose(step_weight, applied_weight)


def test_materialize_returns_module_unchanged_when_nothing_trainable() -> None:
    module = FrozenModule(
        left=jnp.ones((2, 2), dtype=jnp.float32),
        right=jnp.ones((4,), dtype=jnp.float32),
    )

    config = DistillTrainConfig()
    state = initialize_distill_training_state(module, config)
    materialized = materialize_trainable_module(module, state.master_weights, config)

    assert not _selected_leaf_by_path(state.master_weights)
    assert materialized is module


def test_materialize_preserves_alias_identity() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    config = DistillTrainConfig(compute_dtype=jnp.bfloat16)
    state = initialize_distill_training_state(module, config)
    materialized = materialize_trainable_module(module, state.master_weights, config)

    assert materialized.left is materialized.right


def test_partition_parameter_leaves_masks_aliases() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    partitioned = partition_parameter_leaves(
        module,
        lambda _info: True,
        lambda leaf, _info: leaf.astype(jnp.bfloat16),
    )

    partitioned_leaves = _leaf_by_path(partitioned)
    assert partitioned_leaves["left"].dtype == jnp.bfloat16
    assert partitioned_leaves["right"] is None


def test_combine_parameter_leaves_restores_alias_identity() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)
    partitioned = partition_parameter_leaves(
        module,
        lambda _info: True,
        lambda leaf, _info: leaf.astype(jnp.bfloat16),
    )

    combined = combine_parameter_leaves(module, partitioned)

    assert combined.left is combined.right
    assert combined.left.dtype == jnp.bfloat16


def test_partition_parameter_leaves_handles_dict_fields() -> None:
    first = jnp.ones((2, 2), dtype=jnp.float32)
    second = jnp.full((2, 2), 2.0, dtype=jnp.float32)
    module = DictAliasModule(
        blocks={
            "b": AliasModule(left=first, right=first),
            "a": AliasModule(left=second, right=second),
        },
    )

    partitioned = partition_parameter_leaves(
        module,
        lambda _info: True,
        lambda leaf, _info: leaf.astype(jnp.bfloat16),
    )

    partitioned_leaves = _leaf_by_path(partitioned)
    assert partitioned_leaves["blocks['a'].left"].dtype == jnp.bfloat16
    assert partitioned_leaves["blocks['a'].right"] is None
    assert partitioned_leaves["blocks['b'].left"].dtype == jnp.bfloat16
    assert partitioned_leaves["blocks['b'].right"] is None

    combined = combine_parameter_leaves(module, partitioned)
    assert combined.blocks["a"].left is combined.blocks["a"].right
    assert combined.blocks["b"].left is combined.blocks["b"].right


def test_master_weights_are_independent_copies() -> None:
    layer = FullPrecisionLinearConfig(precision=jnp.bfloat16).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=False,
        key=jax.random.key(11),
    )

    config = DistillTrainConfig(master_dtype=jnp.float32)
    state = initialize_distill_training_state(layer, config)

    master_weights = _selected_leaf_by_path(state.master_weights)
    assert master_weights["weights"].dtype == jnp.float32
    assert layer.weights.dtype == jnp.bfloat16
    assert master_weights["weights"] is not layer.weights


def test_full_pipeline_on_llama_decoder() -> None:
    decoder_config = _TINY_LLAMA_CONFIG.to_decoder_config(
        context_length=64,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )
    decoder = decoder_config.empty()

    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.bfloat16)
    state = initialize_distill_training_state(decoder, config)
    materialized = materialize_trainable_module(decoder, state.master_weights, config)

    original_leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(decoder)}
    materialized_leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(materialized)}

    selected_paths = set(_selected_leaf_by_path(state.master_weights))
    trainable_paths = {
        leaf.path
        for leaf in original_leaves.values()
        if leaf.path in selected_paths or leaf.alias_of in selected_paths
    }
    for master_weight in _selected_leaf_by_path(state.master_weights).values():
        assert master_weight.dtype == jnp.float32

    for path, leaf in materialized_leaves.items():
        if path in trainable_paths:
            assert leaf.dtype == jnp.bfloat16, f"Trainable leaf {path} should be bf16"
        else:
            assert leaf.dtype == original_leaves[path].dtype, (
                f"Frozen leaf {path} dtype changed from {original_leaves[path].dtype} to {leaf.dtype}"
            )

    assert any("norm" in path and path.endswith("scales") for path in trainable_paths)
    assert not any(path.startswith("transformer.global_rope.") for path in trainable_paths)
    assert any("embedding" in path for path in trainable_paths)
    assert any(path.endswith("weights") for path in trainable_paths)


def test_full_pipeline_round_trip_preserves_values() -> None:
    layer = FullPrecisionLinearConfig(precision=jnp.float32).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=True,
        key=jax.random.key(12),
    )

    config = DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.bfloat16)
    state = initialize_distill_training_state(layer, config)
    materialized = materialize_trainable_module(layer, state.master_weights, config)

    assert jnp.allclose(
        materialized.weights,
        layer.weights.astype(jnp.bfloat16),
        atol=0,
    )
    assert materialized.biases is not None
    assert jnp.allclose(
        materialized.biases,
        layer.biases.astype(jnp.bfloat16),
        atol=0,
    )
