import jax
import jax.numpy as jnp
import transformers
from einops import rearrange
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

from fartsovka.modules import Decoder, LlamaRoPEConfig, UnscaledRoPEConfig
from tests.executorch_llama.rope import apply_rotary_emb

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch

ROPE_ATOL = 0.02


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
    fs_layer_config = UnscaledRoPEConfig(
        precision=jnp.float32,
        max_sequence_length=max_position_embeddings,
        base=rope_theta,
    )
    fs_layer = fs_layer_config.init(head_size, max_position_embeddings)
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192, 1)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name="rope_cosines",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name="rope_sines",
    )


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
    fs_layer_config = LlamaRoPEConfig(
        precision=jnp.float32,
        base=rope_theta,
        max_sequence_length=max_position_embeddings,
        scaling_factor=scaling_factor,
        original_context_length=original_context_length,
        low_frequency_factor=low_frequency_factor,
        high_frequency_factor=high_frequency_factor,
    )
    fs_layer = fs_layer_config.init(head_size, max_position_embeddings)
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192 * 32, 17)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name="rope_cosines",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name="rope_sines",
    )


def test_rope(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: Decoder,
) -> None:
    hf_layer = huggingface_llama.model.rotary_emb
    fs_layer = fartsovka_llama.rope
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192, 17)
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_cosines, hf_sines = hf_layer(sample_input_torch.float(), sample_input_torch)
    hf_cosines = from_torch(hf_cosines.squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0))
    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name="rope_cosines",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name="rope_sines",
    )


def test_executorch_apply_rotary_emb(
    fartsovka_llama: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    head_dim = fartsovka_llama.rope.head_dim
    sequence_length = 10
    indices = jnp.arange(sequence_length)

    positional_embeddings = fartsovka_llama.rope(indices)
    cosines, sines = (
        positional_embeddings.cosines[:, : head_dim // 2],
        positional_embeddings.sines[:, : head_dim // 2],
    )

    cosines_torch = to_torch(cosines)
    sines_torch = to_torch(sines)

    sample_input = jax.random.normal(rng_key, (sequence_length, head_dim))

    sample_input_reordered = rearrange(
        sample_input,
        "tokens (reim rotors) -> tokens (rotors reim)",
        rotors=head_dim // 2,
    )
    sample_input_torch = to_torch(sample_input_reordered).unsqueeze(1).unsqueeze(0)

    fs_output = positional_embeddings.apply(sample_input)
    et_output, _ = apply_rotary_emb(sample_input_torch, sample_input_torch, cosines_torch, sines_torch)
    et_output = from_torch(et_output.squeeze(0).squeeze(1))

    rearranged_et_output = rearrange(
        et_output,
        "tokens (rotors reim) -> tokens (reim rotors)",
        tokens=sequence_length,
        rotors=head_dim // 2,
        reim=2,
    )

    assert_close(
        result=fs_output,
        reference=rearranged_et_output,
        rtol=QUANTIZED_RTOL,
        operation_name="executorch_apply_rotary_emb",
    )
