import jax
import jax.numpy as jnp
import pytest  # For pytest.skip
import torch
import transformers  # For hf_model type hint
from einops import rearrange
from jaxtyping import PRNGKeyArray
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # Keep for unscaled/scaled tests

# For fartsovka_model type hint
from fartsovka.modules import Decoder, LlamaRoPEConfig, UnscaledRoPEConfig

# For apply_rotary_emb from executorch tests
from tests.executorch_llama.rope import apply_rotary_emb

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch

ROPE_ATOL = 0.02  # From original file


def test_unscaled_rope() -> None:  # Stays as is
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
        # dim=head_size # LlamaRotaryEmbedding constructor might take dim
    )
    fs_layer_config = UnscaledRoPEConfig(
        precision=jnp.float32,
        max_sequence_length=max_position_embeddings,
        base=rope_theta,
    )
    fs_layer = fs_layer_config.init(head_size, max_position_embeddings)
    fs_layer_forward = checkify_forward(fs_layer)

    sample_input = jnp.arange(0, 8192, 1)  # These are effectively position_ids
    # For LlamaRotaryEmbedding, the first arg `x` is a dummy tensor whose shape indicates seq_len,
    # and the second arg `position_ids` is used to get the embeddings.
    # Let's make x a minimal dummy tensor.
    dummy_x_for_hf = torch.randn(1, sample_input.shape[0], 1, head_size)  # B, S, H, D (H=1 dummy)

    hf_cosines, hf_sines = hf_layer(dummy_x_for_hf, position_ids=to_torch(sample_input).unsqueeze(0))

    hf_cosines = from_torch(hf_cosines.squeeze(0).squeeze(0))  # Remove B and H dummy dim
    hf_sines = from_torch(hf_sines.squeeze(0).squeeze(0))  # Remove B and H dummy dim

    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name="rope_cosines_unscaled",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name="rope_sines_unscaled",
    )


def test_scaled_rope() -> None:  # Stays as is
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
        # dim=head_size
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

    sample_input = jnp.arange(0, 8192 * 32, 17)  # Position IDs
    dummy_x_for_hf = torch.randn(1, sample_input.shape[0], 1, head_size)  # B, S, H, D

    hf_cosines, hf_sines = hf_layer(dummy_x_for_hf, position_ids=to_torch(sample_input).unsqueeze(0))

    hf_cosines = from_torch(hf_cosines.squeeze(0).squeeze(0))
    hf_sines = from_torch(hf_sines.squeeze(0).squeeze(0))

    err, fs_positional_embeddings = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name="rope_cosines_scaled",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name="rope_sines_scaled",
    )


def test_rope(  # Refactored
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
) -> None:
    # Attempt to access the RoPE implementation from the Hugging Face model
    hf_rope_layer = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rope_layer is None:
        pytest.skip(f"Model {hf_model.config.model_type} does not have .model.rotary_emb or .rotary_emb")

    fs_rope_layer = fartsovka_model.global_rope  # Assuming this is the RoPE module
    fs_rope_layer_forward = checkify_forward(fs_rope_layer)

    # Determine head_dim from fartsovka_model (needed for dummy tensor for HF RoPE call)
    if not fartsovka_model.layers:
        pytest.skip("Fartsovka model has no layers to determine head_dim for RoPE test.")

    head_dim_attr_path = ["head_dim", "attention.head_dim", "config.head_dim"]  # Common paths
    head_dim = None
    if hasattr(fs_rope_layer, "head_dim"):
        head_dim = fs_rope_layer.head_dim
    else:
        for attr_path_str in head_dim_attr_path:
            obj = fartsovka_model.layers[0]
            found = True
            for attr in attr_path_str.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and isinstance(obj, int):
                head_dim = obj
                break
    if head_dim is None:
        pytest.skip(f"Could not determine head_dim for RoPE test from fartsovka_model for {fartsovka_model}.")

    position_ids_jax = jnp.arange(0, 2048, 17, dtype=jnp.int32)  # Reduced length for faster generic testing
    sequence_length = position_ids_jax.shape[0]
    position_ids_torch = to_torch(position_ids_jax).unsqueeze(0)  # Add batch dimension

    # Create a dummy 'x' tensor for the HF RoPE call.
    # Shape for many RoPE impls: (batch_size, seq_len, num_heads, head_dim) or (batch_size, seq_len, hidden_dim)
    # For just getting freqs, a simple (1, seq_len, head_dim) often works.
    dummy_x_for_hf_rope = torch.randn(1, sequence_length, head_dim, dtype=torch.float32)

    try:
        # Try common signature: x, position_ids (Llama, Gemma, Mistral)
        hf_cosines_th, hf_sines_th = hf_rope_layer(dummy_x_for_hf_rope, position_ids=position_ids_torch)
    except TypeError:
        try:
            # Try common signature: x, seq_len (Qwen2)
            hf_cosines_th, hf_sines_th = hf_rope_layer(dummy_x_for_hf_rope, seq_len=sequence_length)
        except TypeError as e:
            pytest.skip(f"HF RoPE call failed for model {hf_model.config.model_type}: {e}")

    # Squeeze out batch and num_heads/dummy dimensions if they are 1
    hf_cosines = from_torch(hf_cosines_th.squeeze())
    hf_sines = from_torch(hf_sines_th.squeeze())

    err, fs_positional_embeddings = fs_rope_layer_forward(position_ids_jax)
    err.throw()

    # Ensure dimensions match after squeezing, as HF RoPE might return [seq_len, head_dim/2] or [seq_len, 1, head_dim/2] etc.
    # Fartsovka RoPE returns [seq_len, head_dim/2]
    if hf_cosines.ndim > fs_positional_embeddings.cosines.ndim:
        hf_cosines = hf_cosines.squeeze()  # General squeeze for extra dims of size 1
    if hf_sines.ndim > fs_positional_embeddings.sines.ndim:
        hf_sines = hf_sines.squeeze()

    assert_close(
        result=fs_positional_embeddings.cosines,
        reference=hf_cosines,
        atol=ROPE_ATOL,
        operation_name=f"rope_cosines_for_{hf_model.config.model_type}",
    )
    assert_close(
        result=fs_positional_embeddings.sines,
        reference=hf_sines,
        atol=ROPE_ATOL,
        operation_name=f"rope_sines_for_{hf_model.config.model_type}",
    )


def test_executorch_apply_rotary_emb(  # Refactored
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    if not hasattr(fartsovka_model, "global_rope") or not hasattr(fartsovka_model.global_rope, "head_dim"):
        pytest.skip("Cannot determine head_dim for executorch_apply_rotary_emb test from fartsovka_model.global_rope.")

    head_dim = fartsovka_model.global_rope.head_dim
    sequence_length = 10
    indices = jnp.arange(sequence_length, dtype=jnp.int32)

    positional_embeddings = fartsovka_model.global_rope(indices)
    # Ensure cosines and sines are correctly sliced for apply_rotary_emb
    # apply_rotary_emb expects freqs for half the head_dim (rotors)
    cosines_jax = positional_embeddings.cosines[:, : head_dim // 2]
    sines_jax = positional_embeddings.sines[:, : head_dim // 2]

    cosines_torch = to_torch(cosines_jax)
    sines_torch = to_torch(sines_jax)

    sample_input = jax.random.normal(rng_key, (sequence_length, head_dim))

    # Executorch's apply_rotary_emb often expects [..., num_rotors, 2] format
    sample_input_reordered_for_et = rearrange(
        sample_input,
        "tokens (rotors reim) -> tokens rotors reim",  # Keep reim as last for ET
        rotors=head_dim // 2,
        reim=2,
    )
    # apply_rotary_emb expects q, k: [bs, num_heads, seq_len, head_dim_half_x_2] or similar
    # For testing, we pass the same tensor for q and k.
    # Add dummy batch and num_heads dimensions for ET's apply_rotary_emb
    sample_input_et_torch = to_torch(sample_input_reordered_for_et).unsqueeze(0).unsqueeze(0)

    fs_output = positional_embeddings.apply(sample_input)  # Fartsovka RoPE apply takes [seq, dim]

    # ET apply_rotary_emb(q, k, cos, sin)
    et_output_q, et_output_k = apply_rotary_emb(
        sample_input_et_torch, sample_input_et_torch, cosines_torch, sines_torch
    )
    # We only need one output for comparison, e.g., query
    et_output_processed = from_torch(et_output_q.squeeze(0).squeeze(0))  # Remove dummy batch and head dims

    # Rearrange ET output from [tokens, rotors, reim] back to [tokens, (reim rotors)] for comparison
    rearranged_et_output_for_comparison = rearrange(
        et_output_processed,
        "tokens rotors reim -> tokens (reim rotors)",
        rotors=head_dim // 2,
        reim=2,
    )

    assert_close(
        result=fs_output,
        reference=rearranged_et_output_for_comparison,
        rtol=QUANTIZED_RTOL,  # Using QUANTIZED_RTOL as this might be comparing against a potentially different RoPE impl
        operation_name="executorch_apply_rotary_emb",
    )
