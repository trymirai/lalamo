import jax
import math
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
    new_heads = []
    for i in range(model.num_heads):
        head_blocks = []
        old_block = model.medusa_heads[i][0]
        
        weight_key = f"{i}.0.linear.weight"
        bias_key = f"{i}.0.linear.bias"
        
        weights = weights_dict[weight_key]
        bias = weights_dict[bias_key]
        
        weights_jax = from_torch(weights) 
        bias_jax = from_torch(bias)
        
        new_block = eqx.tree_at(
            lambda b: (b.linear.weights, b.linear.biases),
            old_block,
            (weights_jax, bias_jax)
        )
        head_blocks.append(new_block)
        
        for j in range(1, model.num_layers):
            head_blocks.append(model.medusa_heads[i][j])
            
        new_heads.append(head_blocks)

    return eqx.tree_at(
        lambda m: m.medusa_heads,
        model,
        new_heads
    )


def test_medusa_weights_loading(rng_key: PRNGKeyArray) -> None:
    hidden_size = 2048
    num_heads = 3
    num_layers = 1
    
    resblock_config = ResBlockConfig(
        linear_config=FullPrecisionLinearConfig(precision=jnp.float32),
        activation=Activation.SILU,
    )
    
    model = create_medusa_model(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        key=rng_key,
        resblock_config=resblock_config,
    )
    
    weights_dict = load_medusa_weights(
        "getmirai/llama-3.2-1b-medusa-3-1",
        "medusa_lm_head.safetensors"
    )
    
    loaded_model = load_medusa_into_fartsovka(model, weights_dict)
    
    assert loaded_model.num_heads == num_heads
    
    for i in range(num_heads):
        original_linear = model.medusa_heads[i][0].linear
        loaded_linear = loaded_model.medusa_heads[i][0].linear
        original_weights = original_linear.export_weights()["weights"]
        loaded_weights = loaded_linear.export_weights()["weights"]
        
        assert not jnp.allclose(original_weights, loaded_weights)
        
        weight_key = f"{i}.0.linear.weight"
        source_weights = from_torch(weights_dict[weight_key]).T
        assert jnp.allclose(loaded_weights, source_weights)


class TorchTransposedLinear(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) / math.sqrt(hidden_size))
        self.biases = torch.nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights @ x + self.biases


class TorchResBlock(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = TorchTransposedLinear(hidden_size)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class TorchMedusa(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.medusa_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                *[TorchResBlock(hidden_size) for _ in range(num_layers)]
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        medusa_states = []
        for i in range(self.num_heads):
            mhidden_states = self.medusa_heads[i](hidden_states)
            medusa_states.append(mhidden_states)
        return torch.stack(medusa_states, dim=0)


def test_medusa_pytorch_comparison(rng_key: PRNGKeyArray) -> None:
    hidden_size = 2048
    num_heads = 3
    num_layers = 1

    resblock_config = ResBlockConfig(
        linear_config=FullPrecisionLinearConfig(precision=jnp.float32),
        activation=Activation.SILU,
    )
    
    jax_model = create_medusa_model(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        key=rng_key,
        resblock_config=resblock_config,
    )

    torch_model = TorchMedusa(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

    weights_dict = load_medusa_weights(
        "getmirai/llama-3.2-1b-medusa-3-1",
        "medusa_lm_head.safetensors"
    )
    
    loaded_jax_model = load_medusa_into_fartsovka(jax_model, weights_dict)
    
    for i in range(num_heads):
        weight_key = f"{i}.0.linear.weight"
        bias_key = f"{i}.0.linear.bias"
        head = torch_model.medusa_heads[i]
        assert isinstance(head, torch.nn.Sequential)
        block = head[0]
        assert isinstance(block, TorchResBlock)
        block.linear.weights.data = weights_dict[weight_key]
        block.linear.biases.data = weights_dict[bias_key]
    
    sample_input = jax.random.normal(rng_key, (hidden_size, hidden_size), dtype=jnp.float32)
    sample_input_torch = to_torch(sample_input)

    torch_out = torch_model(sample_input_torch)
    print(f"\nTorch: {torch_out[:1]}")
    jax_out = loaded_jax_model(sample_input)
    print(f"\nJax: {jax_out[:1]}")
    jax_from_torch = from_torch(torch_out)

    assert_close(
        result=jax_from_torch,
        reference=jax_out,
    )