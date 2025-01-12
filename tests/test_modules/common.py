import numpy as np
import torch
from jax import numpy as jnp
from jax.experimental.checkify import checkify, div_checks, index_checks, nan_checks, user_checks

__all__ = ["assert_close", "from_torch", "to_torch"]

ATOL = 1e-5
QUANTIZED_ATOL = 1e-5


LAYERS_TO_TEST = list(range(16))


def assert_close(a: jnp.ndarray, b: jnp.ndarray, atol: float = ATOL, operation_name: str | None = None) -> None:
    absdiff = jnp.abs(a - b)
    num_violations = jnp.sum(absdiff > atol)
    errmax, errmax_idx = jnp.max(absdiff), jnp.argmax(absdiff)
    rms_diff = jnp.sqrt(jnp.mean(jnp.square(absdiff)))
    rms_a = jnp.sqrt(jnp.mean(jnp.square(a)))
    rms_b = jnp.sqrt(jnp.mean(jnp.square(b)))
    rel_rms_b = rms_diff / (rms_b + 1e-10)
    rel_rms_a = rms_diff / (rms_a + 1e-10)
    errmax_idx = tuple(i.item() for i in jnp.unravel_index(errmax_idx, absdiff.shape))
    if operation_name is not None:
        operation_description = f" during {operation_name}"
    else:
        operation_description = ""
    message = (
        f"{num_violations} violations > {atol}{operation_description}."
        f" Max error: {errmax:.5f} at index {errmax_idx}. Error RMS: {rms_diff:.5f}."
        f" RMS of a: {rms_a:.5f}, RMS of b: {rms_b:.5f}."
        f" Relative error RMS: {rel_rms_a:.0%} of RMS of a, {rel_rms_b:.0%} of RMS of b."
        f" Shape: {a.shape}"
    )
    assert jnp.allclose(a, b, atol=atol), message


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
