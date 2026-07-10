import json

from frozendict import frozendict

from lalamo.model_import.model_configs.huggingface.gemma4 import (
    Gemma4RopeParameters,
    HFGemma4TextConfig,
    RopeParameters,
)
from lalamo.modules import (
    AttentionConfig,
    AttentionProjectionMode,
    DecoderConfig,
    DenseMLPConfig,
    MixtureOfExpertsConfig,
)


def _config(
    *,
    hidden_size_per_layer_input: int = 2,
    vocab_size_per_layer_input: int = 16,
    num_kv_shared_layers: int = 2,
    use_double_wide_mlp: bool = True,
    enable_moe_block: bool = False,
) -> HFGemma4TextConfig:
    return HFGemma4TextConfig(
        hidden_size=8,
        intermediate_size=16,
        model_type="gemma4_text",
        num_hidden_layers=4,
        sliding_window=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=None,
        head_dim=4,
        global_head_dim=4,
        max_position_embeddings=128,
        rope_parameters=Gemma4RopeParameters(
            full_attention=RopeParameters(
                rope_theta=1_000_000.0,
                rope_type="proportional",
                partial_rotary_factor=0.25,
            ),
            sliding_attention=RopeParameters(rope_theta=10_000.0, rope_type="default"),
        ),
        final_logit_softcapping=None,
        vocab_size=32,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        vocab_size_per_layer_input=vocab_size_per_layer_input,
        num_kv_shared_layers=num_kv_shared_layers,
        use_double_wide_mlp=use_double_wide_mlp,
        enable_moe_block=enable_moe_block,
        num_experts=4 if enable_moe_block else None,
        top_k_experts=2 if enable_moe_block else None,
        moe_intermediate_size=6 if enable_moe_block else None,
    )


def test_gemma4_config_variants() -> None:
    decoder_config = _config().to_decoder_config(context_length=None, metadata_dict={})
    borrowed_layer = decoder_config.transformer_config.layer_configs[2]
    assert decoder_config.transformer_config.kv_reuse_map == frozendict({2: 0, 3: 1})
    assert isinstance(borrowed_layer.mixer_config, AttentionConfig)
    assert borrowed_layer.mixer_config.projection_mode is AttentionProjectionMode.QKV

    restored_config = DecoderConfig.from_json(json.loads(json.dumps(decoder_config.to_json())))
    assert restored_config.transformer_config.kv_reuse_map == frozendict({2: 0, 3: 1})
    assert isinstance(restored_config.transformer_config.kv_reuse_map, frozendict)

    moe = _config(enable_moe_block=True).to_decoder_config(context_length=None, metadata_dict={})
    parallel_layer_config = moe.transformer_config.layer_configs[0]
    assert isinstance(parallel_layer_config.mlp_config, DenseMLPConfig)
    assert isinstance(parallel_layer_config.parallel_mlp_config, MixtureOfExpertsConfig)
    assert parallel_layer_config.mlp_output_norm_config is not None
    assert parallel_layer_config.parallel_mlp_output_norm_config is not None
