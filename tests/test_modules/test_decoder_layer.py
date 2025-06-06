import jax
import pytest
import torch
import transformers  # For hf_model type hint
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Decoder and ETTransformer are needed for specific QLoRA test fixtures and parent model type hints
from fartsovka.modules import Decoder
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import (
    MAX_TOKEN_INDEX,  # Still used for random position_ids
    QUANTIZED_ATOL,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)

# Import LayerPair for type hinting the new fixture
from .conftest import LayerPair


def test_decoder_layer(
    layers: LayerPair,  # Fixture iterated_layers_data yields LayerPair
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    # layers.reference_layer is the HF/reference model's full DecoderLayer
    # layers.fartsovka_layer is the Fartsovka DecoderLayer
    hf_layer_ref = layers.reference_layer  # This is the HF model's DecoderLayer object
    if hf_model.config.model_type == "llama":
        assert isinstance(hf_layer_ref, LlamaDecoderLayer)

    fs_layer_uut = layers.fartsovka_layer  # This is your Fartsovka DecoderLayer object
    fs_layer_forward = checkify_forward(fs_layer_uut)

    if not (
        hasattr(fs_layer_uut, "mlp")
        and hasattr(fs_layer_uut.mlp, "up_projection")
        and hasattr(fs_layer_uut.mlp.up_projection, "input_dim")
    ):
        pytest.skip(f"Could not determine input_dim for Fartsovka layer: {fs_layer_uut}")
    input_dim = fs_layer_uut.mlp.up_projection.input_dim
    sequence_length = 64

    # Split rng_key for distinct random operations
    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)

    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Attempt to get RoPE from hf_model.model.rotary_emb
    # Most HF CausalLM models have rotary_emb on their base model (e.g., hf_model.model)
    hf_rotary_emb_module = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rotary_emb_module is None:
        pytest.skip(f"Could not find rotary_emb for HF model {hf_model.config.name_or_path}")

    # The HF rotary_emb often expects the input tensor `x` to derive seq_len, and `position_ids`.
    # The actual application of RoPE happens inside the attention mechanism usually.
    # For testing the DecoderLayer, we need the (cos, sin) to pass to the HF layer if it expects them.
    # However, many HF DecoderLayer implementations expect position_ids and calculate RoPE internally.
    cos_sin_for_hf_layer = None
    # LlamaDecoderLayer's forward pass takes `position_ids`, not precomputed `position_embeddings`.
    # It then calls its self_attn, which uses its own rotary_emb with these position_ids.
    # So we don't need to pass (cos,sin) to the hf_layer_ref directly for Llama.
    # Other models might differ. The key is to match how the HF layer expects to receive positional info.

    positional_embeddings_fs = fartsovka_model.global_rope(position_ids)

    # Create causal mask
    torch_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=sample_input_torch.dtype)
        * torch.finfo(sample_input_torch.dtype).min,
        diagonal=1,
    )
    # HF models usually expect attention_mask shape [batch_size, 1, seq_len, target_seq_len] or [batch_size, seq_len, target_seq_len]
    # For causal mask, it's often [1, 1, seq_len, seq_len] for self-attention
    final_torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)

    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    # The hf_layer_ref is a full DecoderLayer, so it takes position_ids and attention_mask
    hf_output_tuple = hf_layer_ref(  # type: ignore
        sample_input_torch,
        attention_mask=final_torch_mask,
        position_ids=position_ids_torch,  # Pass position_ids
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    hf_output_tensor = hf_output_tuple[0] if isinstance(hf_output_tuple, tuple) else hf_output_tuple
    hf_output_final = from_torch(hf_output_tensor.squeeze(0))

    err, fs_output_struct = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings_fs,
        local_positional_embeddings=positional_embeddings_fs,  # Assuming same for this generic test
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output_struct.output,  # Assuming DecoderLayerOutput has an 'output' field
        reference=hf_output_final,
        operation_name=f"decoder_layer_for_{hf_model.config.model_type}",
    )


# Commented out test_gemma2_decoder_layer remains as is for now


def test_qlora_decoder_layer(
    qlora_layers: LayerPair,  # Fixture qlora_iterated_layers_data yields LayerPair
    executorch_llama: ETTransformer,  # QLoRA test uses specific ET model
    fartsovka_qlora_llama: Decoder,  # QLoRA test uses specific Fartsovka QLoRA model
    rng_key: PRNGKeyArray,
) -> None:
    # qlora_layers.fartsovka_layer is a DecoderLayer from fartsovka_qlora_llama
    # qlora_layers.reference_layer is an ETDecoderLayer (TransformerBlock)
    fs_layer_uut = qlora_layers.fartsovka_layer
    fs_layer_forward = checkify_forward(fs_layer_uut)
    et_layer_ref = qlora_layers.reference_layer  # This is an ETDecoderLayer (TransformerBlock)

    if not (
        hasattr(fs_layer_uut, "mlp")
        and hasattr(fs_layer_uut.mlp, "up_projection")
        and hasattr(fs_layer_uut.mlp.up_projection, "input_dim")
    ):
        pytest.skip(f"Could not determine input_dim for Fartsovka QLoRA layer: {fs_layer_uut}")
    input_dim = fs_layer_uut.mlp.up_projection.input_dim
    sequence_length = 64

    # Split rng_key for distinct random operations
    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)

    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)

    # ETTransformer.rope.get_freqs is used here for the Executorch model side
    freqs_cos, freqs_sin = executorch_llama.rope.get_freqs(None, sequence_length)
    # Fartsovka QLoRA model uses its global_rope for its positional_embeddings
    positional_embeddings_fs = fartsovka_qlora_llama.global_rope(position_ids)

    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # ETDecoderLayer (TransformerBlock) forward pass takes x, freqs_cos, freqs_sin
    # It does not take a full attention_mask. Causality is handled by its internal scaled_dot_product_attention.
    et_output_tensor = et_layer_ref(sample_input_torch, freqs_cos, freqs_sin)  # type: ignore
    et_output_final = from_torch(et_output_tensor.squeeze(0))

    err, fs_output_struct = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings_fs,
        local_positional_embeddings=positional_embeddings_fs,  # Assuming same for this QLoRA test
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output_struct.output,  # Assuming DecoderLayerOutput has an 'output' field
        reference=et_output_final,
        atol=QUANTIZED_ATOL,
        rtol=QUANTIZED_RTOL,
        operation_name="qlora_decoder_layer",
    )
