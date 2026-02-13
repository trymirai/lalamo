from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array

from lalamo.common import ParameterPath


def _fuse_weight_norm_conv1d(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> tuple[Array, Array | None]:
    import torch
    from torch import nn
    from torch.nn.utils import remove_weight_norm, weight_norm

    from lalamo.modules.torch_interop import jax_to_torch

    """Fuse weight normalization for a Conv1d layer using PyTorch's remove_weight_norm.

    Creates a temporary PyTorch Conv1d module, applies weight_norm, loads the weight_g
    and weight_v parameters, then calls remove_weight_norm to get the fused weight.

    Args:
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Path to the weight-normalized layer (e.g., "quantizers/0/in_proj").

    Returns:
        Tuple of (fused_weight, bias) as JAX arrays.
    """
    weight_g = weights_dict[path / "weight_g"]
    weight_v = weights_dict[path / "weight_v"]
    bias = weights_dict[path / "bias"]

    # weight_g shape: (out_channels, 1, 1) for Conv1d
    # weight_v shape: (out_channels, in_channels, kernel_size)
    out_channels, in_channels, kernel_size = weight_v.shape

    # Create a temporary Conv1d and apply weight_norm
    temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
    temp_conv = weight_norm(temp_conv, name="weight", dim=0)

    # Load the weight_g and weight_v parameters
    with torch.no_grad():
        temp_conv.weight_g = torch.nn.Parameter(jax_to_torch(weight_g), requires_grad=False)
        temp_conv.weight_v = torch.nn.Parameter(jax_to_torch(weight_v), requires_grad=False)
        if bias is not None:
            temp_conv.bias = torch.nn.Parameter(jax_to_torch(bias), requires_grad=False)

    # Fuse with remove_weight_norm
    temp_conv = remove_weight_norm(temp_conv, name="weight")

    # Extract fused weight and convert back to JAX array
    fused_weight = jnp.array(temp_conv.weight.detach().numpy())
    fused_bias = jnp.array(temp_conv.bias.detach().numpy()) if temp_conv.bias is not None else None

    return fused_weight, fused_bias


def _fuse_parametrized_weight_norm_conv1d(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    is_transposed: bool = False,
) -> tuple[Array, Array | None]:
    import torch
    from torch import nn
    from torch.nn.utils.parametrizations import weight_norm as param_weight_norm
    from torch.nn.utils.parametrize import remove_parametrizations

    from lalamo.modules.torch_interop import jax_to_torch

    """Fuse weight normalization for a Conv1d layer using PyTorch's remove_parametrizations.

    This handles the newer parametrization format where weights are stored as:
        - path/parametrizations/weight/original0 (weight_g)
        - path/parametrizations/weight/original1 (weight_v)
        - path/bias

    Args:
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Path to the weight-normalized layer.
        is_transposed: If True, creates ConvTranspose1d instead of Conv1d.

    Returns:
        Tuple of (fused_weight, bias) as JAX arrays.
    """
    weight_g = weights_dict[path / "parametrizations" / "weight" / "original0"]
    weight_v = weights_dict[path / "parametrizations" / "weight" / "original1"]
    bias = weights_dict[path / "bias"]

    if is_transposed:
        # ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
        in_channels, out_channels, kernel_size = weight_v.shape
        temp_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
    else:
        # Conv1d weight shape: (out_channels, in_channels, kernel_size)
        out_channels, in_channels, kernel_size = weight_v.shape
        temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    # Apply weight_norm to match the parametrization
    temp_conv = param_weight_norm(temp_conv, name="weight", dim=0)

    # Load the weight_g and weight_v parameters
    with torch.no_grad():
        temp_conv.parametrizations.weight.original0 = torch.nn.Parameter(jax_to_torch(weight_g), requires_grad=False)
        temp_conv.parametrizations.weight.original1 = torch.nn.Parameter(jax_to_torch(weight_v), requires_grad=False)
        if bias is not None:
            temp_conv.bias = torch.nn.Parameter(jax_to_torch(bias), requires_grad=False)

    # Fuse with remove_parametrizations
    remove_parametrizations(temp_conv, "weight")

    # Extract fused weight and convert back to JAX array
    fused_weight = jnp.array(temp_conv.weight.detach().numpy())
    fused_bias = jnp.array(temp_conv.bias.detach().numpy()) if temp_conv.bias is not None else None

    return fused_weight, fused_bias
