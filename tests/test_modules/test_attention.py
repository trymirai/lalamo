import jax
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import LlamaAttention

from fartsovka.modules.llama import BaselineLlama

from .common import assert_close, checkify_forward, from_torch, to_torch


def test_attention_no_mask_no_cache(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_llama.model.layers[0].self_attn
    assert isinstance(hf_layer, LlamaAttention)
    fs_layer = fartsovka_llama.layers[0].attention
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim
    sequence_length = 64

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)
    # We create a zero mask to ensure that the attention is not affected by the mask
    torch_zero_mask = torch.zeros((1, 1, sequence_length, sequence_length))
    positional_embeddings = fartsovka_llama.rope(position_ids)

    # Run forward passes
    hf_output = from_torch(
        hf_layer(sample_input_torch, position_embeddings=(cos, sin), attention_mask=torch_zero_mask)[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings)
    err.throw()
    assert_close(hf_output, fs_output.attention_output)
