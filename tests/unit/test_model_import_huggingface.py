import json
from pathlib import Path

import jax.numpy as jnp

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_linear, load_mlx_quantized_tied_embedding
from lalamo.model_import.model_configs.huggingface.lfm2 import HFLFM2Config
from lalamo.modules.embedding import MLXQuantizedTiedEmbeddingConfig
from lalamo.modules.linear import MLXQuantizedLinearConfig
from lalamo.quantization import QuantizationMode


def test_hf_lfm2_from_json_accepts_legacy_quantization_field(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Lfm2ForCausalLM"],
                "block_auto_adjust_ff_dim": False,
                "block_dim": 8,
                "block_ff_dim": 16,
                "block_ffn_dim_multiplier": 1.0,
                "block_mlp_init_scale": 1.0,
                "block_multiple_of": 8,
                "block_norm_eps": 1e-5,
                "block_out_init_scale": 1.0,
                "block_use_swiglu": True,
                "block_use_xavier_init": True,
                "bos_token_id": 0,
                "conv_L_cache": 4,
                "conv_bias": False,
                "conv_dim": 8,
                "conv_use_xavier_init": True,
                "dtype": "float32",
                "eos_token_id": 1,
                "full_attn_idxs": [0],
                "hidden_size": 8,
                "initializer_range": 0.02,
                "max_position_embeddings": 32,
                "model_type": "lfm2",
                "norm_eps": 1e-5,
                "num_attention_heads": 2,
                "num_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "pad_token_id": 0,
                "quantization": {"group_size": 4, "bits": 8},
                "rope_theta": 10000.0,
                "transformers_version": "0",
                "use_cache": True,
                "use_pos_enc": True,
                "vocab_size": 16,
            }
        )
    )
    config = HFLFM2Config.from_json(config_path)

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
