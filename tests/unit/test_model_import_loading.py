import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.int import IntMatrixForInference, IntMatrixForTraining, IntSpec
from lalamo.compressed.microfloat import MicrofloatMatrixForInference
from lalamo.compressed.mlx import MLXMatrixForInference, MLXMatrixForTraining
from lalamo.initializer import EmptyInitializer, Initializer
from lalamo.model import Model, ModelConfig
from lalamo.model_import.loaders.dflash_loader import load_hf_dflash_draft_model
from lalamo.model_import.loaders.huggingface import (
    load_huggingface_classifier,
    load_input_embedding_matrix,
    load_linear,
    load_moe,
)
from lalamo.model_import.loaders.utils import decode_mxfp4
from lalamo.model_import.model_configs.foreign_config import ForeignConfig
from lalamo.model_import.model_configs.huggingface import HFDFlashConfig, ModernBERTConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.activations import SiLU
from lalamo.modules.classifier import Classifier
from lalamo.modules.decoder import PerLayerEmbedding, PLEModelConfig
from lalamo.modules.embedding import TiedEmbedding
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import (
    DenseMLP,
    DenseMLPConfig,
    MixtureOfExperts,
    MixtureOfExpertsConfig,
    SoftmaxRouting,
)
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import Attention
from lalamo.safetensors import safe_write
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import (
    CompressionImplementation,
    FullPrecisionMatrix,
    FullPrecisionSpec,
    GradientEstimator,
    Layout,
    MatmulConfig,
    WeightMatrix,
)
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

INPUT_DIM = 8
OUTPUT_DIM = 4
NUM_GROUPS = 2
CLASSIFIER_VOCAB_SIZE = 16
CLASSIFIER_HIDDEN_SIZE = 4
CLASSIFIER_INTERMEDIATE_SIZE = 8
CLASSIFIER_NUM_HEADS = 2
CLASSIFIER_NUM_LABELS = 2
DFLASH_MODEL_DIM = 8
DFLASH_HIDDEN_DIM = 16
DFLASH_HEAD_DIM = 4


def _pack_int32(values: Array, bits: int) -> Array:
    values_per_word = 32 // bits
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // values_per_word, values_per_word).astype(jnp.uint32)
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(bits)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint32).astype(jnp.int32)


def _linear_template(dtype: DTypeLike | None, layout: Layout = Layout.OUTPUT_INPUT) -> Linear:
    result = LinearConfig().init(
        initializer=EmptyInitializer(default_dtype=dtype, sharding_config=make_test_sharding_config()),
        input_dim=INPUT_DIM,
        output_dims=(OUTPUT_DIM,),
        has_biases=False,
    )
    if layout == Layout.OUTPUT_INPUT:
        return result

    weights = FullPrecisionSpec(layout=layout).compress(
        dummy_array((OUTPUT_DIM, INPUT_DIM), dtype, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )
    return eqx.tree_at(lambda module: module.weights, result, weights)


def _gpt_oss_moe_template() -> MixtureOfExperts:
    linear_config = LinearConfig()
    expert_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(alpha=1.702),
        has_up_biases=True,
        has_down_biases=True,
        gate_clipping=(None, 7.0),
        up_clipping=(-6.0, 8.0),
    )
    config = MixtureOfExpertsConfig(
        expert_config=expert_config,
        router_config=linear_config,
        routing_function=SoftmaxRouting(),
        num_routed_experts=2,
        num_active_routed_experts=1,
        router_has_biases=True,
        num_shared_experts=0,
        expert_hidden_dim=32,
    )
    initializer = EmptyInitializer(default_dtype=jnp.bfloat16, sharding_config=make_test_sharding_config())
    return config.init(initializer, model_dim=64, hidden_dim=64)


def _mlx_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(OUTPUT_DIM * INPUT_DIM, dtype=jnp.int32).reshape(OUTPUT_DIM, INPUT_DIM)
    return {
        path / "weight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((OUTPUT_DIM, NUM_GROUPS), dtype=jnp.bfloat16),
        path / "biases": jnp.zeros((OUTPUT_DIM, NUM_GROUPS), dtype=jnp.bfloat16),
    }


def _awq_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(INPUT_DIM * OUTPUT_DIM, dtype=jnp.int32).reshape(INPUT_DIM, OUTPUT_DIM)
    unpacked_zero_points = jnp.zeros((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.int32)
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "qzeros": _pack_int32(unpacked_zero_points, bits=8),
        path / "scales": jnp.ones((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.bfloat16),
    }


def _symmetric_awq_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(INPUT_DIM * OUTPUT_DIM, dtype=jnp.int32).reshape(INPUT_DIM, OUTPUT_DIM) + 128
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.bfloat16),
    }


def _classifier_tensor(shape: tuple[int, ...]) -> Array:
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)


def _dflash_tensor(shape: tuple[int, ...], offset: int) -> Array:
    values = jnp.arange(prod(shape), dtype=jnp.float32) + offset
    return values.reshape(shape).astype(jnp.bfloat16)


def _muse_glimmer_dflash_config() -> dict[str, object]:
    return {
        "architectures": ["MuseGlimmerAssistantModel"],
        "attention_dropout": 0,
        "block_size": 4,
        "bos_token_id": 0,
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "head_dim": DFLASH_HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": DFLASH_MODEL_DIM,
        "intermediate_size": DFLASH_HIDDEN_DIM,
        "layer_types": ["sliding_attention"],
        "mask_token_id": 31,
        "max_position_embeddings": 64,
        "model_type": "muse_glimmer_assistant",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pad_token_id": 2,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"},
        "sliding_window": 16,
        "target_layer_ids": [1, 3],
        "transformers_version": "5.15.0.dev0",
    }


def _muse_glimmer_dflash_weights() -> dict[str, Array]:
    q_dim = 2 * DFLASH_HEAD_DIM
    kv_dim = DFLASH_HEAD_DIM
    context_dim = 2 * DFLASH_MODEL_DIM
    return {
        "encoder.fc.weight": _dflash_tensor((DFLASH_MODEL_DIM, context_dim), 1),
        "encoder.output_norm_enc.weight": _dflash_tensor((DFLASH_MODEL_DIM,), 2),
        "layers.0.input_layernorm.weight": _dflash_tensor((DFLASH_MODEL_DIM,), 3),
        "layers.0.mlp.down_proj.weight": _dflash_tensor((DFLASH_MODEL_DIM, DFLASH_HIDDEN_DIM), 4),
        "layers.0.mlp.gate_proj.weight": _dflash_tensor((DFLASH_HIDDEN_DIM, DFLASH_MODEL_DIM), 5),
        "layers.0.mlp.up_proj.weight": _dflash_tensor((DFLASH_HIDDEN_DIM, DFLASH_MODEL_DIM), 6),
        "layers.0.post_attention_layernorm.weight": _dflash_tensor((DFLASH_MODEL_DIM,), 7),
        "layers.0.self_attn.k_norm.weight": _dflash_tensor((DFLASH_HEAD_DIM,), 8),
        "layers.0.self_attn.k_proj.weight": _dflash_tensor((kv_dim, DFLASH_MODEL_DIM), 9),
        "layers.0.self_attn.o_proj.weight": _dflash_tensor((DFLASH_MODEL_DIM, q_dim), 10),
        "layers.0.self_attn.q_norm.weight": _dflash_tensor((DFLASH_HEAD_DIM,), 11),
        "layers.0.self_attn.q_proj.weight": _dflash_tensor((q_dim, DFLASH_MODEL_DIM), 12),
        "layers.0.self_attn.v_proj.weight": _dflash_tensor((kv_dim, DFLASH_MODEL_DIM), 13),
        "norm.weight": _dflash_tensor((DFLASH_MODEL_DIM,), 14),
    }


def test_load_muse_glimmer_checkpoint_reshapes_to_qwen_dflash(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    with config_path.open("w") as config_file:
        json.dump(_muse_glimmer_dflash_config(), config_file)

    weights = _muse_glimmer_dflash_weights()
    with (tmp_path / "model.safetensors").open("wb") as weights_file:
        safe_write(weights_file, weights)

    reshaped_config = HFDFlashConfig.from_json(config_path)
    sharding_config = ShardingConfig.replicated()
    with jax.set_mesh(sharding_config.mesh):
        model = load_hf_dflash_draft_model(
            tmp_path,
            sharding_config=sharding_config,
            dtype=jnp.bfloat16,
        )

        assert reshaped_config.model_type == "qwen3"
        assert reshaped_config.architectures == ("DFlashDraftModel",)
        assert reshaped_config.num_target_layers == 52
        assert reshaped_config.vocab_size == 202_048
        assert reshaped_config.dflash_config.mask_token_id == 31
        assert reshaped_config.dflash_config.target_layer_ids == (1, 3)
        assert reshaped_config.sliding_window == 32
        assert not reshaped_config.sliding_attention_is_causal

        assert model.context_norm.scales is not None
        assert model.output_norm.scales is not None
        assert np.array_equal(model.context_projection.weights.decompress(), weights["encoder.fc.weight"])
        assert np.array_equal(model.context_norm.scales, weights["encoder.output_norm_enc.weight"])
        assert np.array_equal(model.output_norm.scales, weights["norm.weight"])

        (layer,) = model.layers
        assert isinstance(layer.mixer, Attention)
        assert isinstance(layer.mlp, DenseMLP)
        assert not layer.mixer.config.is_causal
        assert layer.mixer.config.sliding_window_size == 32
        expected_qkv = np.concatenate(
            tuple(weights[f"layers.0.self_attn.{projection}_proj.weight"] for projection in ("q", "k", "v")),
            axis=0,
        )
        expected_up_gate = np.concatenate(
            tuple(weights[f"layers.0.mlp.{projection}_proj.weight"] for projection in ("up", "gate")),
            axis=0,
        )
        expected_state_kv = np.concatenate(
            tuple(weights[f"layers.0.self_attn.{projection}_proj.weight"] for projection in ("k", "v")),
            axis=0,
        )
        assert np.array_equal(layer.mixer.qkvg_projection.weights.decompress(), expected_qkv)
        assert np.array_equal(layer.mlp.up_projection.weights.decompress(), expected_up_gate)
        assert np.array_equal(model.state_kv_projection.weights.decompress(), expected_state_kv)


def _classifier_template() -> Classifier:
    config = ModernBERTConfig(
        architectures=["ModernBertForSequenceClassification"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        classifier_activation="gelu",
        classifier_bias=False,
        classifier_dropout=0.0,
        classifier_pooling="mean",
        cls_token_id=1,
        decoder_bias=False,
        deterministic_flash_attn=False,
        embedding_dropout=0.0,
        eos_token_id=2,
        global_attn_every_n_layers=1,
        global_rope_theta=10000.0,
        gradient_checkpointing=False,
        hidden_activation="gelu",
        hidden_size=CLASSIFIER_HIDDEN_SIZE,
        initializer_cutoff_factor=2.0,
        initializer_range=0.02,
        intermediate_size=CLASSIFIER_INTERMEDIATE_SIZE,
        layer_norm_eps=1e-5,
        local_attention=4,
        local_rope_theta=10000.0,
        max_position_embeddings=8,
        mlp_bias=False,
        mlp_dropout=0.0,
        model_type="modernbert",
        norm_bias=False,
        norm_eps=1e-5,
        num_attention_heads=CLASSIFIER_NUM_HEADS,
        num_hidden_layers=1,
        pad_token_id=3,
        position_embedding_type="absolute",
        sep_token_id=4,
        transformers_version="test",
        vocab_size=CLASSIFIER_VOCAB_SIZE,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    return config.to_classifier_config().init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config()),
    )


def _classifier_weights(classifier: Classifier) -> Mapping[str, Array]:
    assert isinstance(classifier.embedding, TiedEmbedding)
    layer = classifier.transformer.layers[0]
    assert isinstance(layer.mixer, Attention)
    assert isinstance(layer.mlp, DenseMLP)
    assert layer.pre_mlp_norm is not None
    assert classifier.embedding_norm.scales is not None
    assert layer.pre_mlp_norm.scales is not None
    assert classifier.transformer.output_norm.scales is not None
    assert classifier.prediction_head.norm.scales is not None

    base_path = ParameterPath()
    decoder_path = base_path / "model"
    head_path = base_path / "head"
    classifier_path = base_path / "classifier"

    return {
        decoder_path / "embeddings" / "tok_embeddings" / "weight": _classifier_tensor(
            (CLASSIFIER_VOCAB_SIZE, CLASSIFIER_HIDDEN_SIZE),
        ),
        decoder_path / "embeddings" / "norm" / "weight": _classifier_tensor(classifier.embedding_norm.scales.shape),
        decoder_path / "layers" / 0 / "attn" / "Wqkv" / "weight": _classifier_tensor(
            layer.mixer.qkvg_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "attn" / "Wo" / "weight": _classifier_tensor(
            layer.mixer.out_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "mlp_norm" / "weight": _classifier_tensor(layer.pre_mlp_norm.scales.shape),
        decoder_path / "layers" / 0 / "mlp" / "Wi" / "weight": _classifier_tensor(
            layer.mlp.up_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "mlp" / "Wo" / "weight": _classifier_tensor(
            layer.mlp.down_projection.weights.shape,
        ),
        decoder_path / "final_norm" / "weight": _classifier_tensor(classifier.transformer.output_norm.scales.shape),
        head_path / "dense" / "weight": _classifier_tensor(classifier.prediction_head.dense.weights.shape),
        head_path / "norm" / "weight": _classifier_tensor(classifier.prediction_head.norm.scales.shape),
        classifier_path / "weight": _classifier_tensor(classifier.prediction_head.readout.weights.shape),
        classifier_path / "bias": _classifier_tensor((CLASSIFIER_NUM_LABELS,)),
    }


@pytest.mark.parametrize(
    ("weights_factory", "implementation", "expected_type"),
    [
        (_mlx_weights, CompressionImplementation.INFERENCE, MLXMatrixForInference),
        (_mlx_weights, CompressionImplementation.TRAINING, MLXMatrixForTraining),
        (_awq_weights, CompressionImplementation.INFERENCE, IntMatrixForInference),
        (_awq_weights, CompressionImplementation.TRAINING, IntMatrixForTraining),
    ],
)
@pytest.mark.parametrize("template_layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_load_linear_quantized_checkpoint_uses_requested_dtype_and_implementation(
    weights_factory: Callable[[ParameterPath], Mapping[str, Array]],
    implementation: CompressionImplementation,
    expected_type: type[MLXMatrixForInference | MLXMatrixForTraining | IntMatrixForInference | IntMatrixForTraining],
    template_layout: Layout,
) -> None:
    path = ParameterPath("layer")

    loaded = load_linear(
        _linear_template(jnp.bfloat16, layout=template_layout),
        weights_factory(path),
        path,
        implementation=implementation,
    )

    assert isinstance(loaded.weights, expected_type)
    assert loaded.weights.spec.layout == Layout.OUTPUT_INPUT
    assert loaded.weights.dtype == jnp.bfloat16


def test_mlx_quantized_per_layer_embedding_forwards_training_config() -> None:
    initializer = EmptyInitializer(default_dtype=jnp.bfloat16, sharding_config=make_test_sharding_config())
    token_path = ParameterPath("embed_tokens_per_layer")
    token_embedding = load_input_embedding_matrix(
        initializer.embedding_matrix(OUTPUT_DIM, INPUT_DIM),
        _mlx_weights(token_path),
        token_path,
        implementation=CompressionImplementation.TRAINING,
    )
    assert isinstance(token_embedding, MLXMatrixForTraining)
    assert token_embedding.spec.bits == 8

    config = PLEModelConfig(
        ple_dim=OUTPUT_DIM,
        num_layers=NUM_GROUPS,
        ple_vocab_size=OUTPUT_DIM,
        ple_embed_scale=1.0,
        model_projection_scale=1.0,
        input_scale=1.0,
        linear_config=LinearConfig(),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        ),
    )
    per_layer_embedding = PerLayerEmbedding(
        config=config,
        sharding_config=initializer.sharding_config,
        token_embedding=token_embedding,
        model_projection=LinearConfig().init(
            initializer,
            input_dim=OUTPUT_DIM,
            output_dims=(INPUT_DIM,),
            has_biases=False,
        ),
        projection_norm=config.norm_config.init(initializer, config.ple_dim),
    )
    forward_pass_config = MatmulConfig.for_training(GradientEstimator.LOCAL_ADDITIVE_NOISE)

    with pytest.raises(ValueError, match="Local additive noise is not implemented"):
        per_layer_embedding(
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.ones((1, 1, OUTPUT_DIM), dtype=jnp.bfloat16),
            forward_pass_config=forward_pass_config,
            keychain=Keychain.init(0, sharding_config=initializer.sharding_config),
        )


def test_load_linear_symmetric_awq_without_qzeros_uses_symmetric_spec() -> None:
    path = ParameterPath("layer")

    loaded = load_linear(
        _linear_template(jnp.bfloat16),
        _symmetric_awq_weights(path),
        path,
        implementation=CompressionImplementation.INFERENCE,
    )

    assert isinstance(loaded.weights, IntMatrixForInference)
    assert loaded.weights.spec.is_symmetric
    assert loaded.weights.packed_zero_points is None


def test_load_linear_full_precision_weak_template_preserves_checkpoint_dtype() -> None:
    path = ParameterPath("layer")
    weights = jnp.arange(OUTPUT_DIM * INPUT_DIM, dtype=jnp.float32).reshape(OUTPUT_DIM, INPUT_DIM).astype(jnp.bfloat16)

    loaded = load_linear(
        _linear_template(None),
        {path / "weight": weights},
        path,
    )

    assert isinstance(loaded.weights, FullPrecisionMatrix)
    assert loaded.weights.dtype == jnp.bfloat16


def test_load_linear_full_precision_strong_template_forces_template_dtype() -> None:
    path = ParameterPath("layer")
    weights = jnp.arange(OUTPUT_DIM * INPUT_DIM, dtype=jnp.float32).reshape(OUTPUT_DIM, INPUT_DIM).astype(jnp.bfloat16)

    loaded = load_linear(
        _linear_template(jnp.float32),
        {path / "weight": weights},
        path,
    )

    assert isinstance(loaded.weights, FullPrecisionMatrix)
    assert loaded.weights.dtype == jnp.float32


def test_load_gpt_oss_moe_preserves_native_mxfp4_payload_and_canonical_rows() -> None:
    module = _gpt_oss_moe_template()
    path = ParameterPath()
    experts_path = path / "experts"
    gate_up_blocks = jnp.arange(2 * 64 * 2 * 16, dtype=jnp.int32).astype(jnp.uint8).reshape(2, 64, 2, 16)
    gate_up_scale_bytes = ((jnp.arange(2 * 64 * 2, dtype=jnp.int32) % 5) + 125).astype(jnp.uint8)
    gate_up_scale_bytes = gate_up_scale_bytes.reshape(2, 64, 2)
    gate_up_scales = jax.lax.bitcast_convert_type(gate_up_scale_bytes, jnp.float8_e8m0fnu)
    down_blocks = jnp.arange(2 * 64 * 16, dtype=jnp.int32).astype(jnp.uint8).reshape(2, 64, 1, 16)
    down_scale_bytes = ((jnp.arange(2 * 64, dtype=jnp.int32) % 3) + 126).astype(jnp.uint8)
    down_scale_bytes = down_scale_bytes.reshape(2, 64, 1)
    down_scales = jax.lax.bitcast_convert_type(down_scale_bytes, jnp.float8_e8m0fnu)
    gate_up_bias = jnp.arange(64, dtype=jnp.bfloat16)
    down_bias = jnp.arange(64, dtype=jnp.bfloat16)
    weights: Mapping[str, Array] = {
        path / "router" / "weight": jnp.zeros((2, 64), dtype=jnp.bfloat16),
        path / "router" / "bias": jnp.zeros((2,), dtype=jnp.bfloat16),
        experts_path / "gate_up_proj_blocks": gate_up_blocks,
        experts_path / "gate_up_proj_scales": gate_up_scales,
        experts_path / "gate_up_proj_bias": gate_up_bias,
        experts_path / "down_proj_blocks": down_blocks,
        experts_path / "down_proj_scales": down_scales,
        experts_path / "down_proj_bias": down_bias,
    }

    loaded = load_moe(module, weights, path)

    up_weights = loaded.experts.up_projection.weights
    down_weights = loaded.experts.down_projection.weights
    assert isinstance(up_weights, MicrofloatMatrixForInference)
    assert isinstance(down_weights, MicrofloatMatrixForInference)
    expected_up_blocks = jnp.concatenate((gate_up_blocks[:, 1::2], gate_up_blocks[:, 0::2]), axis=1)
    expected_up_blocks = expected_up_blocks.reshape(2, 64, 2, 2, 8).reshape(2, 64, 32)
    expected_up_scales = jnp.concatenate((gate_up_scale_bytes[:, 1::2], gate_up_scale_bytes[:, 0::2]), axis=1)
    expected_up_scales = jnp.repeat(expected_up_scales, 2, axis=-1)
    assert up_weights.spec.group_size == 16
    assert jnp.array_equal(up_weights.packed_weights, expected_up_blocks)
    assert jnp.array_equal(up_weights.packed_scales, expected_up_scales)
    assert down_weights.spec.group_size == 32
    assert jnp.array_equal(down_weights.packed_weights, down_blocks.reshape(2, 64, 16))
    assert jnp.array_equal(down_weights.packed_scales, down_scale_bytes)

    expected_up_weights = decode_mxfp4(
        gate_up_blocks,
        gate_up_scale_bytes,
        dtype=up_weights.dtype,
        flatten=False,
    )
    expected_up_weights = jnp.concatenate((expected_up_weights[:, 1::2], expected_up_weights[:, 0::2]), axis=1)
    expected_up_weights = expected_up_weights.reshape(2, 64, 64)
    expected_down_weights = decode_mxfp4(
        down_blocks,
        down_scale_bytes,
        dtype=down_weights.dtype,
        flatten=False,
    ).reshape(2, 64, 32)
    np.testing.assert_array_equal(up_weights.decompress(), expected_up_weights)
    np.testing.assert_array_equal(down_weights.decompress(), expected_down_weights)

    assert loaded.experts.up_projection.biases is not None
    assert loaded.experts.down_projection.biases is not None
    expected_up_bias = jnp.broadcast_to(gate_up_bias[1::2] + 1, (2, 32))
    expected_gate_bias = jnp.broadcast_to(gate_up_bias[0::2], (2, 32))
    assert jnp.array_equal(loaded.experts.up_projection.biases[..., :32], expected_up_bias)
    assert jnp.array_equal(loaded.experts.up_projection.biases[..., 32:], expected_gate_bias)
    assert jnp.array_equal(loaded.experts.down_projection.biases, jnp.broadcast_to(down_bias, (2, 64)))


def test_load_huggingface_classifier_uses_hf_embedding_layout() -> None:
    classifier = _classifier_template()
    weights = _classifier_weights(classifier)

    loaded = load_huggingface_classifier(classifier, weights)

    assert isinstance(loaded.embedding, TiedEmbedding)
    assert isinstance(classifier.embedding, TiedEmbedding)
    assert loaded.embedding.embedding.shape == classifier.embedding.embedding.shape
    token_embedding = loaded.embedding.embedding.lookup_embedding(
        0,
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )
    expected_embedding = weights["model.embeddings.tok_embeddings.weight"][0].astype(token_embedding.dtype)
    np.testing.assert_array_equal(token_embedding, expected_embedding)


@dataclass(frozen=True)
class TinyConfig(LalamoConfig):
    def init(self, initializer: Initializer) -> "TinyModule":
        return TinyModule(
            config=self,
            sharding_config=make_test_sharding_config(),
            matrix=initializer.weight_matrix(output_dim=4, input_dim=4),
            fp8_values=initializer.zeros((4,)),
            fp16_values=initializer.zeros((4,)),
        )


class TinyModule(LalamoModule[TinyConfig]):
    matrix: WeightMatrix
    fp8_values: Array
    fp16_values: Array


@dataclass(frozen=True)
class TinyModelConfig(ModelConfig[ChatCodecConfig]):
    module_config: TinyConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "TinyModel":
        return TinyModel(
            config=self,
            sharding_config=make_test_sharding_config(),
            token_codec=self.token_codec_config.init(tokenizer),
            module=self.module_config.init(initializer),
        )


class TinyModel(Model[ChatCodecConfig, TinyModelConfig, ChatCodec]):
    token_codec: ChatCodec
    module: TinyModule


@dataclass(frozen=True)
class TinyForeignConfig(ForeignConfig[TinyModelConfig]):
    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, TinyModel)
        return TinyModel(
            config=model.config,
            sharding_config=make_test_sharding_config(),
            token_codec=model.token_codec,
            module=TinyModule(
                config=model.module.config,
                sharding_config=make_test_sharding_config(),
                matrix=IntSpec(bits=4, group_size=2).compress(
                    weights_dict["matrix"].astype(model.module.matrix.dtype),
                    implementation=implementation,
                    sharding_config=make_test_sharding_config(),
                ),
                fp8_values=model.module.fp8_values,
                fp16_values=model.module.fp16_values,
            ),
        )


def test_foreign_config_load_initializes_model_with_requested_dtype_and_implementation() -> None:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))

    loaded = TinyForeignConfig().load(
        config=TinyModelConfig(
            token_codec_config=ChatCodecConfig(
                prompt_template="",
                output_parser_regex=None,
                system_role_name="system",
                user_role_name="user",
                assistant_role_name="assistant",
                eos_token=None,
                bos_token=None,
            ),
            module_config=TinyConfig(),
        ),
        tokenizer=tokenizer,
        dtype=jnp.bfloat16,
        weights_dict={"matrix": jnp.arange(16, dtype=jnp.float32).reshape(4, 4)},
        implementation=CompressionImplementation.TRAINING,
        sharding_config=make_test_sharding_config(),
    )

    assert isinstance(loaded, TinyModel)
    assert loaded.token_codec.tokenizer is tokenizer
    assert isinstance(loaded.module.matrix, IntMatrixForTraining)
    assert loaded.module.matrix.dtype == jnp.bfloat16


def _tiny_model_config() -> TinyModelConfig:
    return TinyModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        module_config=TinyConfig(),
    )


def _tiny_model_with_dtypes(
    tokenizer: Tokenizer,
    *,
    matrix_dtype: DTypeLike = jnp.float32,
    fp8_values_dtype: DTypeLike = jnp.float8_e4m3fn,
    fp16_values_dtype: DTypeLike = jnp.float16,
) -> TinyModel:
    config = _tiny_model_config()
    return TinyModel(
        config=config,
        sharding_config=make_test_sharding_config(),
        token_codec=config.token_codec_config.init(tokenizer),
        module=TinyModule(
            config=TinyConfig(),
            sharding_config=make_test_sharding_config(),
            matrix=IntSpec(bits=8, group_size=2).compress(
                jnp.arange(16, dtype=matrix_dtype).reshape(4, 4),
                sharding_config=make_test_sharding_config(),
            ),
            fp8_values=jnp.arange(4, dtype=fp8_values_dtype),
            fp16_values=jnp.arange(4, dtype=fp16_values_dtype),
        ),
    )


def test_model_export_load_with_weak_initializer_preserves_saved_float_dtypes(tmp_path: Path) -> None:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    original = _tiny_model_with_dtypes(tokenizer)

    original.save(tmp_path)
    restored = TinyModel.load(tmp_path, sharding_config=make_test_sharding_config())

    assert isinstance(restored.module.matrix, IntMatrixForInference)
    assert restored.module.matrix.spec == original.module.matrix.spec
    assert restored.module.matrix.dtype == original.module.matrix.dtype
    assert restored.module.fp8_values.dtype == jnp.float8_e4m3fn
    assert restored.module.fp16_values.dtype == jnp.float16


def test_model_export_load_with_strong_initializer_forces_saved_float_dtypes(tmp_path: Path) -> None:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    original = _tiny_model_with_dtypes(tokenizer)
    original.save(tmp_path)

    restored = TinyModel.load(tmp_path, dtype=jnp.bfloat16, sharding_config=make_test_sharding_config())

    assert isinstance(restored.module.matrix, IntMatrixForInference)
    assert restored.module.matrix.dtype == jnp.bfloat16
    assert restored.module.fp8_values.dtype == jnp.bfloat16
    assert restored.module.fp16_values.dtype == jnp.bfloat16
