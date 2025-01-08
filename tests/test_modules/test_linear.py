import jax
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from torch import nn

from .common import from_torch, to_torch


def test_full_precision_linear(huggingface_llama: transformers.LlamaModel, rng_key: PRNGKeyArray):
    hf_layer: nn.Linear = huggingface_llama.model.layers[0].mlp.up_proj
    input_dim = hf_layer.in_features
    output_dim = hf_layer.out_features

    sample_input = jax.random.normal(rng_key, (input_dim,))

    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))

    sample_input_jax = from_torch(sample_input_torch)
    jax_output = hf_layer(sample_input_jax)
