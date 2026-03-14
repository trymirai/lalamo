import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillTrainConfig,
    OptimizerGroup,
    compute_distill_kl_loss,
    compute_trace_distill_kl_loss,
    distill_train_step,
    get_muon_weight_dimension_numbers,
    get_optimizer_group,
    initialize_distill_optimizer_state,
    initialize_distill_training_state,
    is_leaf_trainable,
    make_trace_distill_batch,
    materialize_trainable_module,
    summarize_distill_parameters,
)
from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.modules.common import field, iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import (
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    QLoRALinearConfig,
)
from lalamo.quantization import QuantizationMode


class AliasModule(eqx.Module):
    left: jax.Array = field()
    right: jax.Array = field()


class BufferModule(eqx.Module):
    parameter: jax.Array = field()
    buffer: jax.Array = field(trainable=False, matrix=False)


class FrozenModule(eqx.Module):
    left: jax.Array = field(trainable=False)
    right: jax.Array = field(trainable=False, matrix=False)


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

    assert len(state.trainable_parameters) == 1
    trainable_parameter = state.trainable_parameters[0]
    assert trainable_parameter.path == "left"
    assert trainable_parameter.alias_paths == ("left", "right")
    assert trainable_parameter.master_weight.dtype == jnp.float32

    materialized = materialize_trainable_module(
        module,
        state,
        DistillTrainConfig(compute_dtype=jnp.bfloat16),
    )
    materialized_leaves = iter_parameter_leaves(materialized)

    assert materialized.left.dtype == jnp.bfloat16
    assert materialized.right.dtype == jnp.bfloat16
    assert materialized_leaves[1].alias_of == "left"


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
    assert not is_leaf_trainable(leaves["weights"])
    assert is_leaf_trainable(leaves["scales"])
    assert get_optimizer_group(leaves["lora_down_weights"]) == OptimizerGroup.MUON


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
    materialized = materialize_trainable_module(module, state, config)

    assert len(state.trainable_parameters) == 1
    assert state.trainable_parameters[0].path == "parameter"
    assert state.trainable_parameters[0].master_weight.dtype == jnp.float32
    assert materialized.parameter.dtype == jnp.bfloat16
    assert materialized.buffer.dtype == jnp.float32


def test_initialize_distill_training_state_tracks_quant_aux_paths() -> None:
    layer = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (4,), has_biases=True, key=jax.random.key(8))

    config = DistillTrainConfig()
    state = initialize_distill_training_state(layer, config)

    assert {parameter.path for parameter in state.trainable_parameters} == {"biases", "scales", "zero_points"}
    assert {parameter.optimizer_group for parameter in state.trainable_parameters} == {OptimizerGroup.DEFAULT}


def test_get_muon_weight_dimension_numbers_matches_optimizer_groups() -> None:
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

    assert {parameter.path for parameter in state.trainable_parameters} == {"biases", "scales", "zero_points"}
    assert get_muon_weight_dimension_numbers(state) == (None, None, None)


def test_iter_parameter_leaves_defaults_non_static_arrays_to_trainable_matrices() -> None:
    module = BufferModule(
        parameter=jnp.ones((2, 2), dtype=jnp.float32),
        buffer=jnp.ones((4,), dtype=jnp.float32),
    )
    leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(module)}

    assert leaves["parameter"].trainable
    assert leaves["parameter"].matrix
    assert not leaves["buffer"].trainable
    assert not leaves["buffer"].matrix


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
    assert not leaves["transformer.global_rope.cosines"].matrix
    assert not leaves["transformer.global_rope.sines"].matrix


def test_compute_distill_kl_loss_is_zero_for_matching_decoders() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(9))
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([6], dtype=jnp.int32),
    )

    metrics = compute_distill_kl_loss(decoder, decoder, batch)

    assert metrics.valid_tokens == 5
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


def test_compute_trace_distill_kl_loss_is_zero_for_matching_decoder() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(13))
    token_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    decoder_logits = decoder(
        token_ids=token_ids,
        token_positions=jnp.arange(token_ids.shape[1], dtype=jnp.int32)[None, :],
    ).logits[:, :-1, :]
    completion_logits = jax.device_get(decoder_logits[0, 2:, :])
    top_k = 4

    def select_support(step_logits: jax.Array) -> dict[int, float]:
        support_values, support_ids = jax.lax.top_k(step_logits, top_k)
        return dict(zip(support_ids.tolist(), support_values.tolist(), strict=True))

    traces = (
        LalamoCompletion(
            prefix_token_ids=[1, 2, 3],
            completion_token_ids=[4, 5],
            completion_token_logits=[select_support(step_logits) for step_logits in completion_logits],
        ),
    )

    trace_batch = make_trace_distill_batch(traces, pad_token_id=0)
    metrics = compute_trace_distill_kl_loss(decoder, trace_batch)

    assert metrics.valid_tokens == 2
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_distill_train_step_reduces_tiny_llama_loss() -> None:
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
    optimizer_state = initialize_distill_optimizer_state(training_state, optimizer)

    initial_metrics = compute_distill_kl_loss(
        materialize_trainable_module(student, optimizer_state.training_state, config),
        teacher,
        batch,
    )

    for _ in range(20):
        optimizer_state, _ = distill_train_step(
            optimizer_state,
            optimizer,
            student,
            teacher,
            batch,
            config,
        )

    final_metrics = compute_distill_kl_loss(
        materialize_trainable_module(student, optimizer_state.training_state, config),
        teacher,
        batch,
    )

    assert initial_metrics.loss > 1e-4
    assert final_metrics.loss < initial_metrics.loss * 0.5


def test_materialize_returns_module_unchanged_when_nothing_trainable() -> None:
    module = FrozenModule(
        left=jnp.ones((2, 2), dtype=jnp.float32),
        right=jnp.ones((4,), dtype=jnp.float32),
    )

    config = DistillTrainConfig()
    state = initialize_distill_training_state(module, config)
    materialized = materialize_trainable_module(module, state, config)

    assert len(state.trainable_parameters) == 0
    assert materialized is module


def test_materialize_preserves_alias_identity() -> None:
    shared = jnp.ones((2, 2), dtype=jnp.float32)
    module = AliasModule(left=shared, right=shared)

    config = DistillTrainConfig(compute_dtype=jnp.bfloat16)
    state = initialize_distill_training_state(module, config)
    materialized = materialize_trainable_module(module, state, config)

    assert materialized.left is materialized.right


def test_master_weights_are_independent_copies() -> None:
    layer = FullPrecisionLinearConfig(precision=jnp.bfloat16).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=False,
        key=jax.random.key(11),
    )

    config = DistillTrainConfig(master_dtype=jnp.float32)
    state = initialize_distill_training_state(layer, config)

    assert state.trainable_parameters[0].master_weight.dtype == jnp.float32
    assert layer.weights.dtype == jnp.bfloat16
    assert state.trainable_parameters[0].master_weight is not layer.weights


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
    materialized = materialize_trainable_module(decoder, state, config)

    original_leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(decoder)}
    materialized_leaves = {leaf.path: leaf for leaf in iter_parameter_leaves(materialized)}

    trainable_paths = set()
    for parameter in state.trainable_parameters:
        assert parameter.master_weight.dtype == jnp.float32
        for path in parameter.alias_paths:
            trainable_paths.add(path)

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
    materialized = materialize_trainable_module(layer, state, config)

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
