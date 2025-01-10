import jax.numpy as jnp
import transformers
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.modules.rope import RoPE, RoPEParams

from .common import assert_close, checkify_forward, from_torch, to_torch

ROPE_ATOL = 4e-3


def test_unscaled_rope() -> None:
    rope_theta = 10000.0
    hidden_size = 4096
    max_position_embeddings = 8192
    num_attention_heads = 4
    head_size = hidden_size // num_attention_heads
    llama_config = LlamaConfig(
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=dict(
            rope_type="default",
        ),
    )
    hf_layer = LlamaRotaryEmbedding(
        config=llama_config,
    )
    fs_layer = RoPE(
        head_dim=head_size,
        max_sequence_length=max_position_embeddings,
        params=RoPEParams(
            theta=rope_theta,
            use_scaling=False,
            scaling_factor=1.0,
            original_context_length=max_position_embeddings,
            low_frequency_factor=1.0,
            high_frequency_factor=1.0,
        ),
        precision=jnp.float32,
    )
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192, 1)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(hf_cosines, fs_positional_embeddings.cosines, atol=ROPE_ATOL)
    assert_close(hf_sines, fs_positional_embeddings.sines, atol=ROPE_ATOL)


def test_scaled_rope() -> None:
    rope_theta = 10000.0
    hidden_size = 4096
    max_position_embeddings = 8192 * 64
    original_context_length = 8192
    scaling_factor = 64.0
    num_attention_heads = 4
    head_size = hidden_size // num_attention_heads
    low_frequency_factor = 1.0
    high_frequency_factor = 4.0
    llama_config = LlamaConfig(
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=dict(
            rope_type="llama3",
            factor=scaling_factor,
            original_max_position_embeddings=original_context_length,
            low_freq_factor=low_frequency_factor,
            high_freq_factor=high_frequency_factor,
        ),
    )
    hf_layer = LlamaRotaryEmbedding(
        config=llama_config,
    )
    fs_layer = RoPE(
        head_dim=head_size,
        max_sequence_length=max_position_embeddings,
        params=RoPEParams(
            theta=rope_theta,
            use_scaling=True,
            scaling_factor=scaling_factor,
            original_context_length=original_context_length,
            low_frequency_factor=low_frequency_factor,
            high_frequency_factor=high_frequency_factor,
        ),
        precision=jnp.float32,
    )
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192 * 32, 17)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(hf_cosines, fs_positional_embeddings.cosines, atol=ROPE_ATOL)
    assert_close(hf_sines, fs_positional_embeddings.sines, atol=ROPE_ATOL)


def test_rope(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
) -> None:
    hf_layer = huggingface_llama.model.rotary_emb
    fs_layer = fartsovka_llama.rope
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192 * 16, 17)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(hf_cosines, fs_positional_embeddings.cosines, atol=ROPE_ATOL)
    assert_close(hf_sines, fs_positional_embeddings.sines, atol=ROPE_ATOL)
