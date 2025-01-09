import numpy as np
import torch
from jax import numpy as jnp
from jax.experimental.checkify import checkify, div_checks, index_checks, nan_checks, user_checks

__all__ = ["assert_close", "from_torch", "to_torch"]

ATOL = 1e-5


def assert_close(a: jnp.ndarray, b: jnp.ndarray, atol: float = ATOL) -> None:
    absvalues = jnp.abs(a - b)
    absmax, absmax_idx = jnp.max(absvalues), jnp.argmax(absvalues)
    absmax_idx = tuple(i.item() for i in jnp.unravel_index(absmax_idx, absvalues.shape))
    assert jnp.allclose(a, b, atol=atol), f"Absmax: {absmax} at index {absmax_idx}, shape: {a.shape}"


@torch.no_grad()
def from_torch(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.cpu().numpy())


@torch.no_grad()
def to_torch(array: jnp.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array))


def checkify_forward(module):  # noqa: ANN001, ANN202
    return checkify(
        module.__call__,
        errors=index_checks | nan_checks | div_checks | user_checks,
    )
