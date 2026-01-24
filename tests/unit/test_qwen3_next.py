import jax
import jax.numpy as jnp

from lalamo.model_import.decoder_configs.huggingface.qwen3_next import HFQwen3NextConfig
from lalamo.common import ParameterPath
from lalamo.modules import (
    AttentionConfig,
    DeltaNetAttentionConfig,
    DenseMLP,
    DenseMLPConfig,
    Decoder,
    SparseMoEConfig,
    TiedEmbeddingConfig,
)
from lalamo.modules.token_mixers import Attention, DeltaNetAttention


def _base_config(**overrides: object) -> HFQwen3NextConfig:
    base = dict(
        torch_dtype="float16",
        architectures=["Qwen3NextForCausalLM"],
        attention_dropout=0.0,
        bos_token_id=1,
        decoder_sparse_step=2,
        eos_token_id=2,
        full_attention_interval=2,
        head_dim=4,
        hidden_act="silu",
        hidden_size=8,
        initializer_range=0.02,
        intermediate_size=16,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        max_position_embeddings=64,
        mlp_only_layers=[1],
        model_type="qwen3_next",
        moe_intermediate_size=16,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=4,
        num_key_value_heads=1,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        shared_expert_intermediate_size=8,
        tie_word_embeddings=True,
        transformers_version="4.48.0",
        use_cache=True,
        use_sliding_window=False,
        vocab_size=128,
    )
    base.update(overrides)
    return HFQwen3NextConfig(**base)


def _loadable_config() -> HFQwen3NextConfig:
    return _base_config(
        num_hidden_layers=2,
        num_experts=0,
        num_experts_per_tok=0,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        full_attention_interval=2,
        tie_word_embeddings=True,
    )


def _build_hf_weights_for_qwen3_next(decoder: Decoder) -> dict[ParameterPath, jnp.ndarray]:
    weights: dict[ParameterPath, jnp.ndarray] = {}
    decoder_path = ParameterPath("model")

    weights[decoder_path / "embed_tokens" / "weight"] = decoder.embedding.weights

    for layer_idx, layer in enumerate(decoder.transformer.layers):
        layer_path = decoder_path / "layers" / layer_idx

        weights[layer_path / "input_layernorm" / "weight"] = layer.pre_mixer_norm.scales
        weights[layer_path / "post_attention_layernorm" / "weight"] = layer.pre_mlp_norm.scales

        if isinstance(layer.mixer, DeltaNetAttention):
            mixer_path = layer_path / "linear_attn"
            weights[mixer_path / "in_proj_qkvz" / "weight"] = layer.mixer.in_proj_qkvz.weights
            weights[mixer_path / "in_proj_ba" / "weight"] = layer.mixer.in_proj_ba.weights
            weights[mixer_path / "conv.weight"] = layer.mixer.conv.weights
            weights[mixer_path / "out_proj" / "weight"] = layer.mixer.out_proj.weights
            weights[mixer_path / "norm" / "weight"] = layer.mixer.norm.scales
            weights[mixer_path / "dt_bias"] = layer.mixer.dt_bias
            weights[mixer_path / "A_log"] = layer.mixer.a_log
        elif isinstance(layer.mixer, Attention):
            mixer_path = layer_path / "self_attn"
            weights[mixer_path / "q_proj" / "weight"] = layer.mixer.q_proj.weights
            weights[mixer_path / "k_proj" / "weight"] = layer.mixer.k_proj.weights
            weights[mixer_path / "v_proj" / "weight"] = layer.mixer.v_proj.weights
            weights[mixer_path / "o_proj" / "weight"] = layer.mixer.out_projection.weights
            if layer.mixer.query_norm is not None:
                weights[mixer_path / "q_norm" / "weight"] = layer.mixer.query_norm.scales
            if layer.mixer.key_norm is not None:
                weights[mixer_path / "k_norm" / "weight"] = layer.mixer.key_norm.scales
        else:
            raise TypeError(f"Unsupported mixer type: {type(layer.mixer)}")

        if not isinstance(layer.mlp, DenseMLP):
            raise TypeError(f"Unsupported MLP type: {type(layer.mlp)}")

        mlp_path = layer_path / "mlp"
        hidden_dim = layer.mlp.hidden_dim
        up_weights = layer.mlp.up_projection.weights
        weights[mlp_path / "up_proj" / "weight"] = up_weights[:hidden_dim]
        weights[mlp_path / "gate_proj" / "weight"] = up_weights[hidden_dim:]
        weights[mlp_path / "down_proj" / "weight"] = layer.mlp.down_projection.weights

    weights[decoder_path / "norm" / "weight"] = decoder.transformer.output_norm.scales
    return weights


def test_qwen3_next_decoder_config_mixer_and_mlp_types() -> None:
    config = _base_config()
    decoder_config = config.to_decoder_config(
        context_length=32,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )

    assert isinstance(decoder_config.embedding_config, TiedEmbeddingConfig)

    layer_configs = decoder_config.transformer_config.layer_configs
    assert len(layer_configs) == config.num_hidden_layers

    # full_attention_interval=2 => layers 2 and 4 are full attention
    assert isinstance(layer_configs[0].mixer_config, DeltaNetAttentionConfig)
    assert isinstance(layer_configs[1].mixer_config, AttentionConfig)
    assert isinstance(layer_configs[2].mixer_config, DeltaNetAttentionConfig)
    assert isinstance(layer_configs[3].mixer_config, AttentionConfig)

    # decoder_sparse_step=2 + num_experts>0 => layer 4 uses SparseMoE, layer 2 is in mlp_only_layers
    assert isinstance(layer_configs[0].mlp_config, DenseMLPConfig)
    assert isinstance(layer_configs[1].mlp_config, DenseMLPConfig)
    assert isinstance(layer_configs[2].mlp_config, DenseMLPConfig)
    assert isinstance(layer_configs[3].mlp_config, SparseMoEConfig)


def test_qwen3_next_weights_load() -> None:
    config = _loadable_config()
    decoder_config = config.to_decoder_config(
        context_length=32,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        metadata_dict={},
    )
    source_decoder = decoder_config.random_init(key=jax.random.PRNGKey(0))
    weights_dict = _build_hf_weights_for_qwen3_next(source_decoder)

    target_decoder = decoder_config.random_init(key=jax.random.PRNGKey(1))
    loaded_decoder = config._load_weights(target_decoder, weights_dict)  # noqa: SLF001

    assert jnp.array_equal(
        loaded_decoder.embedding.weights,
        source_decoder.embedding.weights,
    )

    token_ids = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    token_positions = jnp.broadcast_to(jnp.arange(token_ids.shape[1]), token_ids.shape)
    result = loaded_decoder(token_ids=token_ids, token_positions=token_positions)
    assert result.logits.shape == (token_ids.shape[0], token_ids.shape[1], config.vocab_size)
