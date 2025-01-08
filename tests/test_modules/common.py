import pytest
import torch
import transformers
from jax import numpy as jnp


@torch.no_grad()
def from_torch(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.cpu().numpy())


@torch.no_grad()
def to_torch(array: jnp.ndarray) -> torch.Tensor:
    return torch.tensor(array)


@pytest.fixture(scope="module")
def huggingface_llama() -> transformers.LlamaModel:
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model.eval()
    return model
