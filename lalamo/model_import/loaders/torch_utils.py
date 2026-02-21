from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array

from lalamo.common import ParameterPath


def fuse_weight_norm_conv1d_as_linear(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> tuple[Array, Array | None]:
    """Fuse weight normalization for a Conv1d layer (mimic PyTorch's remove_weight_norm).

    Here we expect conv-1d with kernel_size==1, so basically it is used as linear projection

    Args:
        weights_dict: Dictionary mapping parameter paths to weight arrays.
        path: Path to the weight-normalized layer

    Returns:
        Tuple of (fused_weight, bias) as JAX arrays.
    """
    weight_g = weights_dict[path / "weight_g"]
    weight_v = weights_dict[path / "weight_v"]
    bias = weights_dict[path / "bias"]

    weight = weight_g * (weight_v / jnp.linalg.norm(weight_v, axis=1, keepdims=True))
    return (weight, bias)


def fuse_parametrized_weight_norm_conv1d(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    is_transposed: bool = False,
) -> tuple[Array, Array | None]:
    """Fuse weight normalization for a Conv1d layer (mimic PyTorch's remove_parametrizations).

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
        reshape_dim, _, _ = weight_v.shape
    else:
        # Conv1d weight shape: (out_channels, in_channels, kernel_size)
        reshape_dim, _, _ = weight_v.shape

    norms = jnp.linalg.norm(weight_v.reshape(reshape_dim, -1), axis=1, keepdims=True)
    norms = norms.reshape(reshape_dim, 1, 1)

    weight = weight_g * (weight_v / norms)
    return (weight, bias)
