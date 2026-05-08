import json
from pathlib import Path

import jax.numpy as jnp

from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.modules.rope import LinearScalingRoPEConfig


def test_neutts_nano_config_uses_dtype_and_linear_rope(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 128000,
                "dtype": "float32",
                "eos_token_id": 128261,
                "head_dim": 64,
                "hidden_act": "silu",
                "hidden_size": 576,
                "initializer_range": 0.02,
                "intermediate_size": 2304,
                "max_position_embeddings": 2048,
                "mlp_bias": False,
                "model_type": "llama",
                "num_attention_heads": 9,
                "num_hidden_layers": 24,
                "num_key_value_heads": 3,
                "pad_token_id": 128001,
                "pretraining_tp": 1,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "factor": 32.0,
                    "rope_type": "linear",
                    "type": "linear",
                },
                "rope_theta": 500000,
                "tie_word_embeddings": True,
                "transformers_version": "4.57.6",
                "use_cache": True,
                "vocab_size": 194256,
            },
        ),
    )

    config = HFLlamaConfig.from_json(config_path)
    decoder_config = config.to_decoder_config(
        context_length=None,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )

    rope_config = decoder_config.transformer_config.layer_configs[0].rope_config
    assert config.default_precision == jnp.dtype("float32")
    assert isinstance(rope_config, LinearScalingRoPEConfig)
    assert rope_config.scaling_factor == 32.0
