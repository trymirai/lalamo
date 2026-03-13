import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.distillation import (
    DistillTrainConfig,
    get_optimizer_group,
    is_leaf_trainable,
    summarize_distill_parameters,
)
from lalamo.modules.common import ParameterRole, iter_parameter_leaves, parameter_field
from lalamo.modules.embedding import (
    MLXSemiQuantizedUntiedEmbeddingConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    TiedEmbeddingConfig,
    UntiedEmbeddingConfig,
)
from lalamo.modules.linear import (
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    MLXQuantizedLinearConfig,
    QLoRALinearConfig,
)
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.quantization import QuantizationMode


class AliasModule(eqx.Module):
    left: jax.Array = parameter_field(parameter_role=ParameterRole.LINEAR_WEIGHT)
    right: jax.Array = parameter_field(parameter_role=ParameterRole.LINEAR_WEIGHT)


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
    summary = summarize_distill_parameters(leaves, DistillTrainConfig(train_base_weight=True))

    assert summary.total_parameters == 4
    assert summary.trainable_parameters == 4
    assert summary.total_master_bytes == 16
    assert summary.by_role == {"linear_weight": 4}
    assert summary.by_group == {"muon": 4}


def test_q_lora_policy_prefers_adapter_weights() -> None:
    config = QLoRALinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
        lora_rank=2,
        lora_scale=1.0,
    )
    layer = config.random_init(4, (4,), True, key=jax.random.key(0))
    leaves = {leaf.field_name: leaf for leaf in iter_parameter_leaves(layer)}

    train_config = DistillTrainConfig()

    assert is_leaf_trainable(leaves["lora_down_weights"], train_config)
    assert is_leaf_trainable(leaves["lora_up_weights"], train_config)
    assert not is_leaf_trainable(leaves["weights"], train_config)
    assert not is_leaf_trainable(leaves["scales"], train_config)
    assert get_optimizer_group(leaves["lora_down_weights"], train_config) == "muon"


def test_quant_aux_can_be_enabled_explicitly() -> None:
    config = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    )
    layer = config.random_init(4, (4,), True, key=jax.random.key(1))
    leaves = {leaf.field_name: leaf for leaf in iter_parameter_leaves(layer)}

    train_config = DistillTrainConfig(train_quant_aux=True)

    assert is_leaf_trainable(leaves["scales"], train_config)
    assert is_leaf_trainable(leaves["zero_points"], train_config)
    assert get_optimizer_group(leaves["scales"], train_config) == "quant_aux"


def test_parameter_roles_cover_llama_relevant_modules() -> None:
    full_precision = FullPrecisionLinearConfig(precision=jnp.float32).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=True,
        key=jax.random.key(2),
    )
    group_quantized = GroupQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=True,
        key=jax.random.key(3),
    )
    mlx_quantized = MLXQuantizedLinearConfig(
        group_size=2,
        weight_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).random_init(
        input_dim=4,
        output_dims=(4,),
        has_biases=True,
        key=jax.random.key(4),
    )
    tied_embedding = TiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        precision=jnp.float32,
    ).random_init(vocab_size=4, model_dim=4, key=jax.random.key(5))
    untied_embedding = UntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        precision=jnp.float32,
    ).random_init(vocab_size=4, model_dim=4, key=jax.random.key(6))
    mlx_tied_embedding = MLXQuantizedTiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=4, model_dim=4)
    mlx_untied_embedding = MLXQuantizedUntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=4, model_dim=4)
    semi_quantized_embedding = MLXSemiQuantizedUntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=4, model_dim=4)
    normalization = NormalizationConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
        use_bias=True,
    ).init(4)

    cases = [
        (full_precision, {ParameterRole.LINEAR_WEIGHT, ParameterRole.LINEAR_BIAS}),
        (
            group_quantized,
            {
                ParameterRole.LINEAR_WEIGHT,
                ParameterRole.LINEAR_BIAS,
                ParameterRole.QUANT_SCALE,
                ParameterRole.QUANT_ZERO_POINT,
            },
        ),
        (
            mlx_quantized,
            {
                ParameterRole.LINEAR_WEIGHT,
                ParameterRole.LINEAR_BIAS,
                ParameterRole.QUANT_SCALE,
                ParameterRole.QUANT_DEQ_BIAS,
            },
        ),
        (tied_embedding, {ParameterRole.INPUT_OUTPUT_EMBEDDING}),
        (untied_embedding, {ParameterRole.INPUT_EMBEDDING, ParameterRole.OUTPUT_EMBEDDING}),
        (
            mlx_tied_embedding,
            {
                ParameterRole.INPUT_OUTPUT_EMBEDDING,
                ParameterRole.QUANT_SCALE,
                ParameterRole.QUANT_DEQ_BIAS,
            },
        ),
        (
            mlx_untied_embedding,
            {
                ParameterRole.INPUT_EMBEDDING,
                ParameterRole.OUTPUT_EMBEDDING,
                ParameterRole.QUANT_SCALE,
                ParameterRole.QUANT_DEQ_BIAS,
            },
        ),
        (
            semi_quantized_embedding,
            {
                ParameterRole.INPUT_EMBEDDING,
                ParameterRole.OUTPUT_EMBEDDING,
                ParameterRole.QUANT_SCALE,
                ParameterRole.QUANT_DEQ_BIAS,
            },
        ),
        (normalization, {ParameterRole.NORM_SCALE, ParameterRole.NORM_BIAS}),
    ]

    for module, expected_roles in cases:
        roles = {leaf.parameter_role for leaf in iter_parameter_leaves(module)}
        assert roles == expected_roles
