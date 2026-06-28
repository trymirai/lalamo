from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders import huggingface as hf_loaders
from lalamo.model_import.model_configs.huggingface.gemma4 import (
    Gemma4RopeParameters,
    HFGemma4Config,
    HFGemma4TextConfig,
    RopeParameters,
)
from lalamo.model_import.model_spec import FileSpec
from lalamo.model_import.model_specs.gemma import GEMMA_MODELS
from lalamo.modules import (
    AttentionConfig,
    AttentionProjectionMode,
    DenseMLP,
    DenseMLPConfig,
    Keychain,
    MixtureOfExperts,
    MixtureOfExpertsConfig,
    ParallelMLP,
    ParallelMLPConfig,
    ProportionalRoPEConfig,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import LogicalAxis
from lalamo.weight_matrix import CompressionImplementation
from tests.helpers import make_test_sharding_config

if TYPE_CHECKING:
    from jaxtyping import Array


def _rope_parameters() -> Gemma4RopeParameters:
    return Gemma4RopeParameters(
        full_attention=RopeParameters(rope_theta=1_000_000.0, rope_type="proportional", partial_rotary_factor=0.25),
        sliding_attention=RopeParameters(rope_theta=10_000.0, rope_type="default"),
    )


def _gemma4_text_config(
    *,
    hidden_size_per_layer_input: int = 2,
    vocab_size_per_layer_input: int = 16,
    num_kv_shared_layers: int = 2,
    use_double_wide_mlp: bool = True,
    attention_k_eq_v: bool = False,
    enable_moe_block: bool = False,
    num_global_key_value_heads: int | None = None,
    num_experts: int | None = None,
    top_k_experts: int | None = None,
    moe_intermediate_size: int | None = None,
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
        num_global_key_value_heads=num_global_key_value_heads,
        head_dim=4,
        global_head_dim=4,
        max_position_embeddings=128,
        rope_parameters=_rope_parameters(),
        final_logit_softcapping=None,
        vocab_size=32,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        vocab_size_per_layer_input=vocab_size_per_layer_input,
        num_kv_shared_layers=num_kv_shared_layers,
        use_double_wide_mlp=use_double_wide_mlp,
        attention_k_eq_v=attention_k_eq_v,
        enable_moe_block=enable_moe_block,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        moe_intermediate_size=moe_intermediate_size,
    )


def test_gemma4_shared_kv_uses_compact_state_and_ple() -> None:
    decoder_config = _gemma4_text_config().to_decoder_config(context_length=None, metadata_dict={})
    transformer_config = decoder_config.transformer_config

    assert decoder_config.ple_model_config is not None
    assert transformer_config.kv_source_per_layer == (0, 1, 0, 1)

    source_layer = transformer_config.layer_configs[0]
    borrowed_layer = transformer_config.layer_configs[2]
    assert isinstance(source_layer.mixer_config, AttentionConfig)
    assert isinstance(borrowed_layer.mixer_config, AttentionConfig)
    assert source_layer.mixer_config.projection_mode is AttentionProjectionMode.QKV
    assert borrowed_layer.mixer_config.projection_mode is AttentionProjectionMode.BORROWED_KV
    assert borrowed_layer.mixer_config.key_norm_config is None
    assert borrowed_layer.hidden_dim == 32
    assert source_layer.has_post_layer_scalar
    assert isinstance(transformer_config.layer_configs[1].rope_config, ProportionalRoPEConfig)


def test_gemma4_dense_big_models_disable_ple_share_qk_values_and_keep_layer_scalar() -> None:
    decoder_config = _gemma4_text_config(
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262_144,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
    ).to_decoder_config(context_length=None, metadata_dict={})
    layer_configs = decoder_config.transformer_config.layer_configs
    sliding_attention = layer_configs[0].mixer_config
    full_attention = layer_configs[1].mixer_config

    assert decoder_config.ple_model_config is None
    assert decoder_config.transformer_config.kv_source_per_layer == (0, 1, 2, 3)
    assert isinstance(sliding_attention, AttentionConfig)
    assert isinstance(full_attention, AttentionConfig)
    assert sliding_attention.projection_mode is AttentionProjectionMode.QKV
    assert full_attention.projection_mode is AttentionProjectionMode.KEY_SAME_AS_VALUE
    assert full_attention.qkv_output_dims == (8, 4)
    assert layer_configs[0].has_post_layer_scalar


def test_gemma4_embedding_scale_matches_hf_bfloat16_rounding() -> None:
    config = replace(
        _gemma4_text_config(
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
        ),
        hidden_size=1536,
    )
    decoder_config = config.to_decoder_config(context_length=None, metadata_dict={})

    assert decoder_config.embedding_config.input_scale == jnp.array(1536**0.5, dtype=jnp.bfloat16).item()


def test_gemma4_instruct_root_eos_list_parses(tmp_path: Path) -> None:
    raw_config = {
        "text_config": asdict(_gemma4_text_config()),
        "dtype": "bfloat16",
        "model_type": "gemma4",
        "eos_token_id": [1, 106],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config))

    config = HFGemma4Config.from_json(config_path)

    assert config.eos_token_ids == [1, 106]


def test_gemma4_base_eos_uses_text_config_when_root_missing(tmp_path: Path) -> None:
    text_config = asdict(_gemma4_text_config())
    text_config["eos_token_id"] = 1
    raw_config = {
        "text_config": text_config,
        "dtype": "bfloat16",
        "model_type": "gemma4",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config))

    config = HFGemma4Config.from_json(config_path)

    assert config.eos_token_ids == [1]


def test_gemma4_moe_config_parses_hf_fields(tmp_path: Path) -> None:
    text_config = asdict(
        _gemma4_text_config(
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
            num_global_key_value_heads=1,
        ),
    )
    text_config.update(
        eos_token_id=1,
        attention_k_eq_v=True,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=6,
    )
    raw_config = {
        "text_config": text_config,
        "dtype": "bfloat16",
        "model_type": "gemma4",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config))

    config = HFGemma4Config.from_json(config_path)
    decoder_config = config.to_decoder_config(context_length=None, metadata_dict={})
    layer_config = decoder_config.transformer_config.layer_configs[0]
    parallel_mlp_config = layer_config.mlp_config
    assert isinstance(parallel_mlp_config, ParallelMLPConfig)
    moe_config = parallel_mlp_config.parallel_mlp_config

    assert isinstance(parallel_mlp_config.primary_mlp_config, DenseMLPConfig)
    assert isinstance(moe_config, MixtureOfExpertsConfig)
    assert moe_config.num_routed_experts == 4
    assert moe_config.num_active_routed_experts == 2
    assert moe_config.num_shared_experts == 0
    assert moe_config.expert_hidden_dim == 6


def test_gemma4_specs_have_chat_templates() -> None:
    gemma4_specs = [spec for spec in GEMMA_MODELS if spec.family == "Gemma-4"]
    assert gemma4_specs
    for spec in gemma4_specs:
        if spec.name.endswith("-it"):
            assert spec.configs.chat_template == FileSpec("chat_template.jinja")
        else:
            assert isinstance(spec.configs.chat_template, str)


def test_gemma4_ple_loader_casts_checkpoint_tensors_to_model_dtype() -> None:
    hf_config = HFGemma4Config(
        text_config=_gemma4_text_config(),
        dtype="bfloat16",
        model_type="gemma4",
        eos_token_id=[1],
    )
    decoder = hf_config.to_decoder_config(context_length=None, metadata_dict={}).init(
        EmptyInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
        ),
    )
    assert decoder.per_layer_embedding is not None

    checkpoint_prefix = ParameterPath("checkpoint")
    weights: dict[str, Array] = {
        checkpoint_prefix / "embed_tokens_per_layer" / "weight": jnp.ones((16, 8), dtype=jnp.bfloat16),
        checkpoint_prefix / "per_layer_model_projection" / "weight": jnp.ones((8, 8), dtype=jnp.bfloat16),
        checkpoint_prefix / "per_layer_projection_norm" / "weight": jnp.ones((2,), dtype=jnp.bfloat16),
    }
    for i in range(4):
        layer_path = checkpoint_prefix / "layers" / i
        weights.update(
            {
                layer_path / "per_layer_input_gate" / "weight": jnp.ones((2, 8), dtype=jnp.bfloat16),
                layer_path / "per_layer_projection" / "weight": jnp.ones((8, 2), dtype=jnp.bfloat16),
                layer_path / "post_per_layer_input_norm" / "weight": jnp.ones((8,), dtype=jnp.bfloat16),
                layer_path / "layer_scalar": jnp.ones((1,), dtype=jnp.bfloat16),
            },
        )

    loaded = hf_config._load_gemma4_weights(  # noqa: SLF001
        decoder,
        weights,
        implementation=CompressionImplementation.INFERENCE,
    )
    assert loaded.per_layer_embedding is not None
    assert loaded.per_layer_embedding.token_embedding.dtype == jnp.float32
    assert loaded.transformer.layers[0].post_layer_scalar is not None
    assert loaded.transformer.layers[0].post_layer_scalar.dtype == jnp.float32

    sharding_config = make_test_sharding_config()
    bf16_token_ple = replace(
        loaded.per_layer_embedding,
        token_embedding=loaded.per_layer_embedding.token_embedding.astype(jnp.bfloat16),
    )
    per_layer_inputs = bf16_token_ple(
        jax.device_put(
            jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
            sharding_config.resolve_sharding((LogicalAxis.BATCH, None)),
        ),
        jax.device_put(
            jnp.ones((2, 2, 8), dtype=jnp.float32),
            sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None)),
        ),
        keychain=Keychain.init(43, sharding_config=sharding_config),
    )
    first_input, *_ = per_layer_inputs
    assert first_input.dtype == jnp.float32


def test_gemma4_no_ple_loader_still_loads_layer_scalar() -> None:
    hf_config = HFGemma4Config(
        text_config=_gemma4_text_config(
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
            num_global_key_value_heads=1,
        ),
        dtype="bfloat16",
        model_type="gemma4",
        eos_token_id=[1],
    )
    decoder = hf_config.to_decoder_config(context_length=None, metadata_dict={}).init(
        EmptyInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
        ),
    )
    assert decoder.per_layer_embedding is None
    assert decoder.transformer.layers[0].post_layer_scalar is not None

    checkpoint_prefix = ParameterPath("checkpoint")
    weights: dict[str, Array] = {
        checkpoint_prefix / "layers" / i / "layer_scalar": jnp.array([i + 1], dtype=jnp.bfloat16)
        for i in range(len(decoder.transformer.layers))
    }

    loaded = hf_config._load_gemma4_weights(  # noqa: SLF001
        decoder,
        weights,
        implementation=CompressionImplementation.INFERENCE,
    )

    assert loaded.transformer.layers[0].post_layer_scalar is not None
    assert loaded.transformer.layers[0].post_layer_scalar.item() == 1.0
    assert loaded.transformer.layers[1].post_layer_scalar is not None
    assert loaded.transformer.layers[1].post_layer_scalar.item() == 2.0


def test_gemma4_parallel_mlp_loader_bakes_static_scales() -> None:
    hf_config = HFGemma4Config(
        text_config=_gemma4_text_config(
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_global_key_value_heads=1,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=6,
        ),
        dtype="bfloat16",
        model_type="gemma4",
        eos_token_id=[1],
    )
    decoder = hf_config.to_decoder_config(context_length=None, metadata_dict={}).init(
        EmptyInitializer(
            default_dtype=jnp.float32,
            sharding_config=make_test_sharding_config(),
        ),
    )
    layer = decoder.transformer.layers[0]
    assert isinstance(layer.mlp, ParallelMLP)

    checkpoint_prefix = ParameterPath("checkpoint")
    layer_path = checkpoint_prefix / "layers" / 0
    mlp_path = layer_path / "mlp"
    weights: dict[str, Array] = {
        layer_path / "input_layernorm" / "weight": jnp.ones((8,), dtype=jnp.float32),
        layer_path / "self_attn" / "q_proj" / "weight": jnp.zeros((8, 8), dtype=jnp.float32),
        layer_path / "self_attn" / "k_proj" / "weight": jnp.zeros((4, 8), dtype=jnp.float32),
        layer_path / "self_attn" / "v_proj" / "weight": jnp.zeros((4, 8), dtype=jnp.float32),
        layer_path / "self_attn" / "o_proj" / "weight": jnp.zeros((8, 8), dtype=jnp.float32),
        layer_path / "self_attn" / "q_norm" / "weight": jnp.ones((4,), dtype=jnp.float32),
        layer_path / "self_attn" / "k_norm" / "weight": jnp.ones((4,), dtype=jnp.float32),
        layer_path / "post_attention_layernorm" / "weight": jnp.ones((8,), dtype=jnp.float32),
        layer_path / "pre_feedforward_layernorm" / "weight": jnp.arange(1, 9, dtype=jnp.float32),
        layer_path / "pre_feedforward_layernorm_2" / "weight": jnp.arange(11, 19, dtype=jnp.float32),
        layer_path / "post_feedforward_layernorm_1" / "weight": jnp.arange(21, 29, dtype=jnp.float32),
        layer_path / "post_feedforward_layernorm_2" / "weight": jnp.arange(31, 39, dtype=jnp.float32),
        layer_path / "post_feedforward_layernorm" / "weight": jnp.ones((8,), dtype=jnp.float32),
        layer_path / "router" / "proj" / "weight": jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
        layer_path / "router" / "scale": jnp.arange(41, 49, dtype=jnp.float32),
        layer_path / "router" / "per_expert_scale": jnp.arange(51, 55, dtype=jnp.float32),
        layer_path / "experts" / "gate_up_proj": jnp.arange(4 * 12 * 8, dtype=jnp.float32).reshape(4, 12, 8),
        layer_path / "experts" / "down_proj": jnp.arange(4 * 8 * 6, dtype=jnp.float32).reshape(4, 8, 6),
        mlp_path / "up_proj" / "weight": jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8),
        mlp_path / "gate_proj" / "weight": jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) + 1000,
        mlp_path / "down_proj" / "weight": jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16),
    }

    loaded_layer = hf_loaders.load_transformer_layer(
        layer,
        weights,
        layer_path,
        layer_path,
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "up_proj",
        "gate_proj",
        "down_proj",
        permute_conv=False,
        implementation=CompressionImplementation.INFERENCE,
    )
    loaded_mlp = loaded_layer.mlp
    assert isinstance(loaded_mlp, ParallelMLP)
    assert isinstance(loaded_mlp.primary_mlp, DenseMLP)
    assert isinstance(loaded_mlp.parallel_mlp, MixtureOfExperts)

    pre_dense_scale = weights[layer_path / "pre_feedforward_layernorm" / "weight"]
    pre_moe_scale = weights[layer_path / "pre_feedforward_layernorm_2" / "weight"]
    post_dense_scale = weights[layer_path / "post_feedforward_layernorm_1" / "weight"]
    post_moe_scale = weights[layer_path / "post_feedforward_layernorm_2" / "weight"]
    per_expert_scale = weights[layer_path / "router" / "per_expert_scale"]

    router_multiplier = weights[layer_path / "router" / "scale"] * jnp.asarray(
        8**-0.5,
        dtype=jnp.float32,
    )
    expected_router = weights[layer_path / "router" / "proj" / "weight"] * router_multiplier[None, :]
    assert jnp.array_equal(loaded_layer.pre_mlp_norm.scales, jnp.ones((8,), dtype=jnp.float32))
    assert jnp.array_equal(loaded_mlp.primary_output_norm.scales, post_dense_scale)
    assert jnp.array_equal(loaded_mlp.parallel_output_norm.scales, post_moe_scale)
    assert jnp.array_equal(loaded_mlp.parallel_mlp.router.weights.decompress(), expected_router)

    routed_gate_up = weights[layer_path / "experts" / "gate_up_proj"] * pre_moe_scale[None, None, :]
    expected_expert_up_gate = jnp.concatenate([routed_gate_up[:, 6:, :], routed_gate_up[:, :6, :]], axis=1)
    assert jnp.array_equal(loaded_mlp.parallel_mlp.experts.up_projection.weights.decompress(), expected_expert_up_gate)

    expected_routed_down = weights[layer_path / "experts" / "down_proj"] * per_expert_scale[:, None, None]
    assert jnp.array_equal(loaded_mlp.parallel_mlp.experts.down_projection.weights.decompress(), expected_routed_down)

    expected_dense_up = weights[mlp_path / "up_proj" / "weight"] * pre_dense_scale[None, :]
    expected_dense_gate = weights[mlp_path / "gate_proj" / "weight"] * pre_dense_scale[None, :]
    expected_dense_up_gate = jnp.concatenate([expected_dense_up, expected_dense_gate], axis=0)
    assert jnp.array_equal(loaded_mlp.primary_mlp.up_projection.weights.decompress(), expected_dense_up_gate)
    assert jnp.array_equal(
        loaded_mlp.primary_mlp.down_projection.weights.decompress(),
        weights[mlp_path / "down_proj" / "weight"],
    )
