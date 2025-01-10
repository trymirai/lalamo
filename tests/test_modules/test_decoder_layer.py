import jax
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from fartsovka.models.baseline_llama import BaselineLlama

from .common import assert_close, checkify_forward, from_torch, to_torch


def test_decoder_layer(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_llama.model.layers[0]
    assert isinstance(hf_layer, LlamaDecoderLayer)
    fs_layer = fartsovka_llama.layers[0]
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim
    sequence_length = 64

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)
    positional_embeddings = fartsovka_llama.rope(position_ids)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    hf_output = from_torch(
        hf_layer(sample_input_torch, attention_mask=torch_mask, position_embeddings=(cos, sin))[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings, mask=jax_mask)
    err.throw()
    assert_close(hf_output, fs_output.output)
