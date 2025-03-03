import jax
import pytest
import torch
import equinox as eqx
import jax.numpy as jnp
from safetensors import safe_open
from jaxtyping import PRNGKeyArray
from huggingface_hub import hf_hub_download

from fartsovka.modules.activations import Activation
from fartsovka.modules.resblock import ResBlockConfig
from fartsovka.modules.medusa import Medusa, create_medusa_model
from fartsovka.modules.linear import FullPrecisionLinearConfig

from .common import (
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)

def load_medusa_weights(repo_id: str, filename: str) -> dict[str, torch.Tensor]:

    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with safe_open(file_path, framework="pt") as f:
        return {key: f.get_tensor(key).to(torch.float32) for key in f.keys()}

def load_medusa_into_fartsovka(
    model: Medusa, 
    weights_dict: dict[str, torch.Tensor]
) -> Medusa:
    
    new_projections = []
    for i in range(model.num_heads):
        old_projection = model.medusa_projections[i]
        
        weight_key = f"{i}.0.linear.weight"
        bias_key = f"{i}.0.linear.bias"
        
        weights = weights_dict[weight_key]
        bias = weights_dict[bias_key]
        
        weights_jax = from_torch(weights.T)
        bias_jax = from_torch(bias)
        
        new_projection = eqx.tree_at(
            lambda m: (m.weights, m.biases),
            old_projection,
            (weights_jax, bias_jax)
        )
        
        new_projections.append(new_projection)
    
    return eqx.tree_at(
        lambda m: m.medusa_projections,
        model,
        new_projections
    )

def test_medusa_weights_loading(rng_key: PRNGKeyArray) -> None:

    hidden_size = 2048
    vocab_size = 128256
    num_heads = 3
    num_layers = 1
    
    resblock_config = ResBlockConfig(
        linear_config=FullPrecisionLinearConfig(precision=jnp.float32),
        activation=Activation.SILU,
    )
    linear_config = FullPrecisionLinearConfig(precision=jnp.float32)
    
    model = create_medusa_model(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
        key=rng_key,
        resblock_config=resblock_config,
        linear_config=linear_config,
    )
    
    weights_dict = load_medusa_weights(
        "getmirai/llama-3.2-1b-medusa-3-1",
        "medusa_lm_head.safetensors"
    )
    
    loaded_model = load_medusa_into_fartsovka(model, weights_dict)
    
    assert loaded_model.num_heads == num_heads
    
    for i in range(num_heads):
        original_weights = model.medusa_projections[i].export_weights()["weights"]
        loaded_weights = loaded_model.medusa_projections[i].export_weights()["weights"]
        
        assert not jnp.allclose(original_weights, loaded_weights)
        
        weight_key = f"{i}.0.linear.weight"
        source_weights = from_torch(weights_dict[weight_key])
        assert jnp.allclose(loaded_weights, source_weights)

def test_medusa_head_linear_forward(
    rng_key: PRNGKeyArray,
) -> None:
    
    hidden_size = 2048
    num_heads = 3
    num_layers = 1
    
    resblock_config = ResBlockConfig(
        linear_config=FullPrecisionLinearConfig(precision=jnp.float32),
        activation=Activation.SILU,
    )
    linear_config = FullPrecisionLinearConfig(precision=jnp.float32)
    
    model = create_medusa_model(
        hidden_size=hidden_size,
        vocab_size=128256,
        num_heads=num_heads,
        num_layers=num_layers,
        key=rng_key,
        resblock_config=resblock_config,
        linear_config=linear_config,
    )
    
    weights_dict = load_medusa_weights(
        "getmirai/llama-3.2-1b-medusa-3-1",
        "medusa_lm_head.safetensors"
    )
    
    model = load_medusa_into_fartsovka(model, weights_dict)
    
    sample_input = jax.random.normal(rng_key, (hidden_size, hidden_size))
    sample_input_torch = to_torch(sample_input)
    
    for i in range(model.num_heads):

        projection = model.medusa_projections[i]
        projection_forward = checkify_forward(projection)
        
        torch_weights = weights_dict[f"{i}.0.linear.weight"]
        torch_bias = weights_dict[f"{i}.0.linear.bias"]
        
        with torch.no_grad():
            torch_output = torch.nn.functional.linear(
                sample_input_torch,
                torch_weights,
                torch_bias
            )
        
        err, fs_output = projection_forward(sample_input)
        err.throw()
        
        torch_output_jax = from_torch(torch_output)
        fs_output  = jnp.array(fs_output).squeeze()

        print(f"torch_output_jax shape: {torch_output_jax.shape if hasattr(torch_output_jax, 'shape') else None}")
        print(f"fs_output shape: {fs_output.shape if hasattr(fs_output, 'shape') else None}")
        
        assert_close(
            result=fs_output,
            reference=torch_output_jax,
            operation_name=f"medusa_head_{i}_linear"
        )

