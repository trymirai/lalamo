from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_gemma4_moe_block
from lalamo.model_import.model_configs.huggingface.gemma4 import (
    Gemma4RopeParameters,
    HFGemma4Config,
    HFGemma4TextConfig,
    RopeParameters,
)
from lalamo.model_import.model_spec import FileSpec
from lalamo.model_import.model_specs.gemma import GEMMA_MODELS
from lalamo.modules import (
    GELU,
    AttentionConfig,
    AttentionProjectionMode,
    DenseMLPConfig,
    Gemma4MoEBlock,
    Gemma4MoEBlockConfig,
    Keychain,
    LinearConfig,
    NormalizationConfig,
    ProportionalRoPEConfig,
    UpcastMode,
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
        num_global_key_value_heads=1,
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
        tie_word_embeddings=True,
        attention_k_eq_v=attention_k_eq_v,
        enable_moe_block=enable_moe_block,
        num_experts=4 if enable_moe_block else None,
        top_k_experts=2 if enable_moe_block else None,
        moe_intermediate_size=6 if enable_moe_block else None,
    )


def _gemma4_moe_block(
    *,
    num_experts: int = 4,
    num_active_experts: int = 2,
    expert_hidden_dim: int = 5,
    model_dim: int = 4,
) -> Gemma4MoEBlock:
    linear_config = LinearConfig()
    norm_config = NormalizationConfig(
        epsilon=1e-6,
        scale_offset=None,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )
    expert_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=GELU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    initializer = EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config())
    return Gemma4MoEBlockConfig(
        expert_config=expert_config,
        router_config=linear_config,
        norm_config=norm_config,
        num_experts=num_experts,
        num_active_experts=num_active_experts,
        expert_hidden_dim=expert_hidden_dim,
        router_norm_epsilon=1e-6,
    ).init(initializer, model_dim=model_dim)


def test_gemma4_shared_kv_uses_compact_state_and_ple() -> None:
    decoder_config = _gemma4_text_config().to_decoder_config(context_length=None, metadata_dict={})
    transformer_config = decoder_config.transformer_config

    assert decoder_config.ple_model_config is not None
    assert transformer_config.kv_source_per_layer == (0, 1, 0, 1)
    assert transformer_config.kv_cache_source_layers == (0, 1)
    assert transformer_config.kv_cache_width == 2

    source_layer = transformer_config.layer_configs[0]
    borrowed_layer = transformer_config.layer_configs[2]
    assert isinstance(source_layer.mixer_config, AttentionConfig)
    assert isinstance(borrowed_layer.mixer_config, AttentionConfig)
    assert source_layer.mixer_config.projection_mode is AttentionProjectionMode.QKV
    assert borrowed_layer.mixer_config.projection_mode is AttentionProjectionMode.BORROWED_Q
    assert borrowed_layer.mixer_config.key_norm_config is None
    assert borrowed_layer.hidden_dim == 32
    assert isinstance(transformer_config.layer_configs[1].rope_config, ProportionalRoPEConfig)


def test_gemma4_dense_big_models_disable_ple_and_share_qk_values() -> None:
    decoder_config = _gemma4_text_config(
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262_144,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        attention_k_eq_v=True,
    ).to_decoder_config(context_length=None, metadata_dict={})
    layer_configs = decoder_config.transformer_config.layer_configs
    sliding_attention = layer_configs[0].mixer_config
    full_attention = layer_configs[1].mixer_config

    assert decoder_config.ple_model_config is None
    assert decoder_config.transformer_config.kv_source_per_layer == (0, 1, 2, 3)
    assert isinstance(sliding_attention, AttentionConfig)
    assert isinstance(full_attention, AttentionConfig)
    assert sliding_attention.projection_mode is AttentionProjectionMode.QKV
    assert full_attention.projection_mode is AttentionProjectionMode.QK_SHARED_VALUE
    assert full_attention.qkv_output_dims == (8, 4)


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
        ),
    )
    text_config.update(
        eos_token_id=1,
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
    moe_config = decoder_config.transformer_config.layer_configs[0].gemma4_moe_config

    assert moe_config is not None
    assert moe_config.num_experts == 4
    assert moe_config.num_active_experts == 2
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

    base = ParameterPath("model")
    weights: dict[str, Array] = {
        base / "embed_tokens_per_layer" / "weight": jnp.ones((16, 8), dtype=jnp.bfloat16),
        base / "per_layer_model_projection" / "weight": jnp.ones((8, 8), dtype=jnp.bfloat16),
        base / "per_layer_projection_norm" / "weight": jnp.ones((2,), dtype=jnp.bfloat16),
    }
    for i in range(4):
        layer_path = base / "layers" / i
        weights.update(
            {
                layer_path / "per_layer_input_gate" / "weight": jnp.ones((2, 8), dtype=jnp.bfloat16),
                layer_path / "per_layer_projection" / "weight": jnp.ones((8, 2), dtype=jnp.bfloat16),
                layer_path / "post_per_layer_input_norm" / "weight": jnp.ones((8,), dtype=jnp.bfloat16),
                layer_path / "layer_scalar": jnp.ones((1,), dtype=jnp.bfloat16),
            },
        )

    loaded = hf_config._load_ple_weights(  # noqa: SLF001
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


def test_load_gemma4_moe_block_uses_hf_weight_paths() -> None:
    block = _gemma4_moe_block()
    path = ParameterPath("model.layers.0")
    gate_up = jnp.arange(4 * 10 * 4, dtype=jnp.float32).reshape(4, 10, 4)
    weights: dict[str, Array] = {
        path / "router" / "proj" / "weight": jnp.ones((4, 4), dtype=jnp.float32),
        path / "router" / "scale": jnp.arange(4, dtype=jnp.float32),
        path / "router" / "per_expert_scale": jnp.arange(4, dtype=jnp.float32),
        path / "experts" / "gate_up_proj": gate_up,
        path / "experts" / "down_proj": jnp.ones((4, 4, 5), dtype=jnp.float32),
        path / "pre_feedforward_layernorm_2" / "weight": jnp.ones((4,), dtype=jnp.float32),
        path / "post_feedforward_layernorm_1" / "weight": jnp.ones((4,), dtype=jnp.float32) * 2,
        path / "post_feedforward_layernorm_2" / "weight": jnp.ones((4,), dtype=jnp.float32) * 3,
    }

    loaded = load_gemma4_moe_block(block, weights, path)

    expected_up_gate = jnp.concatenate([gate_up[:, 5:, :], gate_up[:, :5, :]], axis=1)
    assert jnp.array_equal(loaded.router.weights.decompress(), weights[path / "router" / "proj" / "weight"])
    assert jnp.array_equal(loaded.router_scale, weights[path / "router" / "scale"])
    assert jnp.array_equal(loaded.per_expert_scale, weights[path / "router" / "per_expert_scale"])
    assert jnp.array_equal(loaded.experts.up_projection.weights.decompress(), expected_up_gate)
    assert jnp.array_equal(
        loaded.experts.down_projection.weights.decompress(),
        weights[path / "experts" / "down_proj"],
    )
    assert jnp.array_equal(loaded.pre_moe_norm.scales, weights[path / "pre_feedforward_layernorm_2" / "weight"])
    assert jnp.array_equal(loaded.post_dense_norm.scales, weights[path / "post_feedforward_layernorm_1" / "weight"])
    assert jnp.array_equal(loaded.post_moe_norm.scales, weights[path / "post_feedforward_layernorm_2" / "weight"])


def test_gemma4_moe_router_keeps_expert_axis_unsharded() -> None:
    block = _gemma4_moe_block()
    initializer = EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config())
    sharded_router = LinearConfig().init(initializer, 4, (4,), has_biases=False)

    assert not block.router.weights.is_sharded
    with pytest.raises(ValueError, match="expert axis"):
        replace(block, router=sharded_router)
