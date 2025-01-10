import jax
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama

from .common import assert_close, checkify_forward, from_torch, to_torch


def test_decoder(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    fs_decoder = fartsovka_llama
    fs_decoder_forward = checkify_forward(fs_decoder)

    sequence_length = 512
    vocab_size = fs_decoder.vocab_dim

    # Generate random token IDs
    token_key, _ = jax.random.split(rng_key)
    token_ids = jax.random.randint(token_key, (sequence_length,), 0, vocab_size)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    torch_pre_softmax = huggingface_llama(
        token_ids_torch,
        attention_mask=torch_mask,
        position_ids=position_ids_torch,
    )[0]
    # Run forward passes
    hf_output = from_torch(torch_pre_softmax.squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(hf_output, fs_output.output, atol=1e-3)
