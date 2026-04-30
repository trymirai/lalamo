import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillOptimizerState,
    DistillTrainConfig,
    DistillTrainingState,
    TraceDistillBatch,
    apply_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    compute_trace_distill_batch_metrics,
    compute_trace_distill_step_gradients,
    initialize_distill_training_state,
    inject_lora_adapter_configs,
    lora_trainable_filter,
    make_trace_distill_batch,
    materialize_trainable_module,
    stochastically_quantize_module,
)
from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.modules.activations import SiLU
from lalamo.modules.common import ParameterLeafInfo, ParameterNorm
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import GroupQuantizedLinearConfig, MLXQuantizedLinearConfig, QLoRALinear, QLoRALinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.quantization import QuantizationMode


def _make_optimizer_state(
    training_state: DistillTrainingState,
    optimizer: optax.GradientTransformation,
) -> DistillOptimizerState:
    return DistillOptimizerState(
        training_state=training_state,
        optimizer_state=optimizer.init(training_state.master_weights),
    )


def test_mlx_quantized_linear_random_init_stays_zero_centered() -> None:
    layer = MLXQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(4, (8,), has_biases=False, key=jax.random.key(32))

    prepared_weights = jax.vmap(lambda basis: jnp.concatenate(layer(basis)))(
        jnp.eye(layer.input_dim, dtype=jnp.float32),
    )

    assert float(prepared_weights.min()) < 0.0
    assert float(prepared_weights.max()) > 0.0
    assert abs(float(prepared_weights.mean())) < 0.2


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


def test_inject_lora_adapter_configs_rewrites_nested_group_quantized_linears() -> None:
    config = DenseMLPConfig(
        linear_config=GroupQuantizedLinearConfig(
            group_size=8,
            weight_quantization_mode=QuantizationMode.UINT4,
            activation_quantization_mode=None,
            activation_precision=jnp.float32,
        ),
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )

    qlora_config = inject_lora_adapter_configs(config, lora_rank=4, lora_scale=2.0)

    assert isinstance(qlora_config.linear_config, QLoRALinearConfig)
    assert qlora_config.linear_config.lora_rank == 4
    assert qlora_config.linear_config.lora_scale == 2.0


def test_lora_trainable_filter_accepts_mixture_adapter_weights() -> None:
    info = ParameterLeafInfo(
        owner_type=QLoRALinear,
        shape=(2, 4, 8),
        trainable=True,
        norm=ParameterNorm.SPECTRAL,
        quantized=False,
        quantization_mode=None,
    )

    assert lora_trainable_filter(info)


def _make_tiny_llama_decoder(*, key: jax.Array) -> Decoder:
    decoder_config = _TINY_LLAMA_CONFIG.to_decoder_config(
        context_length=64,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )
    return decoder_config.random_init(key=key)


def test_compute_distill_batch_metrics_matches_teacher_logits() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(9))
    batch = DistillBatch(
        token_ids=jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([6], dtype=jnp.int32),
    )

    metrics = compute_distill_batch_metrics(decoder, decoder, batch)

    assert metrics.valid_tokens == 5
    assert metrics.top1_matches == 5
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_compute_distill_batch_metrics_rejects_batches_without_prediction_tokens() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(34))
    batch = DistillBatch(
        token_ids=jnp.array([[1]], dtype=jnp.int32),
        lengths_without_padding=jnp.array([1], dtype=jnp.int32),
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="at least one prediction token"):
        compute_distill_batch_metrics(decoder, decoder, batch)


def test_initialize_distill_training_state_rejects_empty_trainable_filter() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(30))

    with pytest.raises(ValueError, match="at least one trainable parameter"):
        initialize_distill_training_state(
            decoder,
            DistillTrainConfig(master_dtype=jnp.float32, compute_dtype=jnp.float32),
            trainable_filter=lambda _info: False,
        )


def test_stochastic_quantization_requires_quantized_parameters() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(35))

    with pytest.raises(ValueError, match="at least one quantized parameter"):
        stochastically_quantize_module(decoder, QuantizationMode.UINT4, jax.random.key(0))


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


def test_lalamo_completion_rejects_mismatched_trace_lengths() -> None:
    with pytest.raises(ValueError, match="completion_token_ids"):
        LalamoCompletion(
            prefix_token_ids=[1],
            completion_token_ids=[2, 3],
            completion_token_logits=[{2: 0.5}],
        )


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


def test_compute_trace_distill_batch_metrics_match_teacher_support() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(13))
    trace_batch, _ = _make_trace_batch_from_decoder(decoder)
    metrics = compute_trace_distill_batch_metrics(decoder, trace_batch)

    assert metrics.valid_tokens == 2
    assert metrics.top1_matches == 2
    assert jnp.isclose(metrics.loss, 0.0, atol=1e-6)


def test_compute_trace_distill_batch_metrics_handles_padded_completion_steps() -> None:
    decoder = _make_tiny_llama_decoder(key=jax.random.key(17))

    def select_support(token_ids: list[int], prefix_length: int, completion_length: int) -> list[dict[int, float]]:
        logits = decoder(
            token_ids=jnp.asarray([token_ids], dtype=jnp.int32),
            token_positions=jnp.arange(len(token_ids), dtype=jnp.int32)[None, :],
        ).logits[0, prefix_length - 1 : prefix_length - 1 + completion_length, :]
        return [
            dict(zip(support_ids.tolist(), support_values.tolist(), strict=True))
            for support_values, support_ids in (jax.lax.top_k(step_logits, 4) for step_logits in logits)
        ]

    trace_batch = make_trace_distill_batch(
        (
            LalamoCompletion(
                prefix_token_ids=[1, 2, 3],
                completion_token_ids=[4, 5],
                completion_token_logits=select_support([1, 2, 3, 4, 5], prefix_length=3, completion_length=2),
            ),
            LalamoCompletion(
                prefix_token_ids=[6, 7],
                completion_token_ids=[8],
                completion_token_logits=select_support([6, 7, 8], prefix_length=2, completion_length=1),
            ),
        ),
        pad_token_id=0,
    )

    metrics = compute_trace_distill_batch_metrics(decoder, trace_batch)

    assert metrics.valid_tokens == 3
    assert jnp.isfinite(metrics.loss)
    assert metrics.top1_matches == 3
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
        grads, _ = compute_distill_step_gradients(
            optimizer_state.training_state,
            student,
            teacher,
            batch,
            config,
            QuantizationMode.UINT4,
            stochastic_rounding=False,
            quantization_key=jax.random.key(0),
        )
        optimizer_state = apply_distill_gradients(optimizer_state, optimizer, grads)

    final_metrics = compute_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        teacher,
        batch,
    )

    assert initial_metrics.loss > 1e-4
    assert final_metrics.loss < initial_metrics.loss * 0.5


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
        grads, _ = compute_trace_distill_step_gradients(
            optimizer_state.training_state,
            student,
            trace_batch,
            config,
            QuantizationMode.UINT4,
            stochastic_rounding=False,
            quantization_key=jax.random.key(0),
        )
        optimizer_state = apply_distill_gradients(optimizer_state, optimizer, grads)

    final_metrics = compute_trace_distill_batch_metrics(
        materialize_trainable_module(student, optimizer_state.training_state.master_weights, config),
        trace_batch,
    )

    assert initial_metrics.loss > 1e-5
    assert final_metrics.loss < initial_metrics.loss
