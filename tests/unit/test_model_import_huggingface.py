import jax.numpy as jnp
import pytest

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_linear, load_mlx_quantized_tied_embedding
from lalamo.model_import.model_configs.huggingface.bonsai import BonsaiYarnRopeScalingConfig, HFBonsaiConfig
from lalamo.model_import.model_configs.huggingface.common import MLXQuantizationConfig
from lalamo.model_import.model_configs.huggingface.lfm2 import HFLFM2Config, QuantizationConfig
from lalamo.modules import config_converter
from lalamo.modules.embedding import MLXQuantizedTiedEmbeddingConfig
from lalamo.modules.linear import GroupQuantizedLinearConfig, LinearConfig, MLXQuantizedLinearConfig, QLoRALinearConfig
from lalamo.quantization import QuantizationMode


def test_hf_lfm2_uses_legacy_quantization_field() -> None:
    config = HFLFM2Config(
        architectures=["Lfm2ForCausalLM"],
        block_auto_adjust_ff_dim=False,
        block_dim=8,
        block_ff_dim=16,
        block_ffn_dim_multiplier=1.0,
        block_mlp_init_scale=1.0,
        block_multiple_of=8,
        block_norm_eps=1e-5,
        block_out_init_scale=1.0,
        block_use_swiglu=True,
        block_use_xavier_init=True,
        bos_token_id=0,
        conv_L_cache=4,
        conv_bias=False,
        conv_dim=8,
        conv_use_xavier_init=True,
        eos_token_id=1,
        hidden_size=8,
        initializer_range=0.02,
        max_position_embeddings=32,
        model_type="lfm2",
        norm_eps=1e-5,
        num_attention_heads=2,
        num_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        pad_token_id=0,
        rope_theta=10000.0,
        transformers_version="0",
        use_cache=True,
        use_pos_enc=True,
        vocab_size=16,
        dtype="float32",
        full_attn_idxs=[0],
        quantization=QuantizationConfig(group_size=4, bits=8),
        quantization_config=None,
    )

    decoder_config = config.to_decoder_config(
        context_length=16,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )

    assert decoder_config.embedding_config.embedding_quantization_mode == QuantizationMode.UINT8


def test_load_mlx_quantized_tied_embedding_updates_detected_quantization_mode() -> None:
    module = MLXQuantizedTiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=4,
        embedding_quantization_mode=QuantizationMode.UINT8,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=8)

    loaded = load_mlx_quantized_tied_embedding(
        module,
        {
            ParameterPath("embedding") / "weight": jnp.zeros((2, 1), dtype=jnp.int32),
            ParameterPath("embedding") / "scales": jnp.ones((2, 2), dtype=jnp.float32),
            ParameterPath("embedding") / "biases": jnp.zeros((2, 2), dtype=jnp.float32),
        },
        ParameterPath("embedding"),
    )

    assert loaded.config.embedding_quantization_mode == QuantizationMode.UINT4


def test_load_mlx_quantized_linear_updates_detected_quantization_mode() -> None:
    module = MLXQuantizedLinearConfig(
        group_size=4,
        weight_quantization_mode=QuantizationMode.UINT8,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(8, (8,), has_biases=False)

    loaded = load_linear(
        module,
        {
            ParameterPath("linear") / "weight": jnp.zeros((8, 1), dtype=jnp.int32),
            ParameterPath("linear") / "scales": jnp.ones((8, 2), dtype=jnp.float32),
            ParameterPath("linear") / "biases": jnp.zeros((8, 2), dtype=jnp.float32),
        },
        ParameterPath("linear"),
    )

    assert loaded.config.weight_quantization_mode == QuantizationMode.UINT4


def test_load_mlx_quantized_linear_detects_uint1() -> None:
    module = MLXQuantizedLinearConfig(
        group_size=32,
        weight_quantization_mode=QuantizationMode.UINT8,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(32, (8,), has_biases=False)

    loaded = load_linear(
        module,
        {
            ParameterPath("linear") / "weight": jnp.zeros((8, 1), dtype=jnp.int32),
            ParameterPath("linear") / "scales": jnp.ones((8, 1), dtype=jnp.float32),
            ParameterPath("linear") / "biases": jnp.zeros((8, 1), dtype=jnp.float32),
        },
        ParameterPath("linear"),
    )

    assert loaded.config.weight_quantization_mode == QuantizationMode.UINT1


def test_hf_bonsai_config_uses_uint1_quantization() -> None:
    config = HFBonsaiConfig(
        architectures=["Qwen3ForCausalLM"],
        eos_token_id=151645,
        torch_dtype="bfloat16",
        attention_bias=False,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        max_window_layers=1,
        model_type="qwen3",
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        sliding_window=64,
        tie_word_embeddings=True,
        use_sliding_window=True,
        vocab_size=32,
        head_dim=8,
        quantization=MLXQuantizationConfig(group_size=32, bits=1),
        rope_scaling=BonsaiYarnRopeScalingConfig(
            rope_type="yarn",
            factor=4.0,
            original_max_position_embeddings=32,
        ),
    )

    decoder_config = config.to_decoder_config(
        context_length=64,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )

    assert decoder_config.embedding_config.embedding_quantization_mode == QuantizationMode.UINT1
    qkv_config = decoder_config.transformer_config.layer_configs[0].mixer_config.qkv_projection_config
    assert qkv_config.weight_quantization_mode == QuantizationMode.UINT1


@pytest.mark.parametrize(
    ("inner_config", "expected_type"),
    [
        (
            {
                "type": "GroupQuantizedLinearConfig",
                "group_size": 32,
                "weight_quantization_mode": "uint4",
                "activation_quantization_mode": None,
                "activation_precision": "bfloat16",
            },
            GroupQuantizedLinearConfig,
        ),
        (
            {
                "type": "QLoRALinearConfig",
                "group_size": 32,
                "weight_quantization_mode": "uint4",
                "activation_quantization_mode": None,
                "activation_precision": "bfloat16",
                "lora_rank": 16,
                "lora_scale": 1.0,
            },
            QLoRALinearConfig,
        ),
    ],
)
def test_linear_config_structure_accepts_rht_wrapper(inner_config: dict[str, object], expected_type: type) -> None:
    config = config_converter.structure(
        {
            "type": "RHTLinearWrapperConfig",
            "block_size": 32,
            "inner_config": inner_config,
        },
        LinearConfig,
    )

    assert isinstance(config, expected_type)
