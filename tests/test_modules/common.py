import numpy as np
import torch
from jax import numpy as jnp
from jax.experimental.checkify import checkify, div_checks, index_checks, nan_checks, user_checks

__all__ = ["assert_close", "from_torch", "to_torch"]

ATOL = 1e-3
RTOL = 0.01
QUANTIZED_ATOL = 0.03
QUANTIZED_RTOL = 0.1


LAYERS_TO_TEST = list(range(16))


def assert_close(
    *,
    result: jnp.ndarray,
    reference: jnp.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
    operation_name: str | None = None,
) -> None:
    absdiff = jnp.abs(result - reference)

    allowed_diff = atol + rtol * jnp.abs(reference)
    err = jnp.maximum(absdiff - allowed_diff, 0)
    err_rel = err / (jnp.abs(reference) + 1e-10)
    max_err = jnp.max(err)
    max_err_idx = tuple(i.item() for i in jnp.unravel_index(jnp.argmax(err), err.shape))
    max_err_rel = err_rel[max_err_idx]
    max_err_reference_value = reference[max_err_idx]

    num_violations = jnp.sum(err > 0)

    rms_diff = jnp.sqrt(jnp.mean(jnp.square(absdiff)))
    rms_result = jnp.sqrt(jnp.mean(jnp.square(result)))
    rms_reference = jnp.sqrt(jnp.mean(jnp.square(reference)))
    rel_rms_reference = rms_diff / (rms_reference + 1e-10)

    if operation_name is not None:
        operation_description = f" during {operation_name}"
    else:
        operation_description = ""

    # Convert JAX scalars to Python floats for safe formatting
    max_err_float = float(max_err.item())
    max_err_rel_float = float(max_err_rel.item())
    max_err_reference_value_float = float(max_err_reference_value.item())
    rms_diff_float = float(rms_diff.item())
    rms_result_float_orig_msg = float(rms_result.item()) # renamed to avoid clash with later one
    rms_reference_float_orig_msg = float(rms_reference.item()) # renamed
    rel_rms_reference_float = float(rel_rms_reference.item())
    num_violations_int = int(num_violations.item())

    message = (
        f"{num_violations_int} violations > {atol:.1e} + {rtol:.2%}{operation_description}."
        f" Max error: {max_err_float:.3g} ({max_err_rel_float:.2%}) at index {max_err_idx}"
        f" (reference value: {max_err_reference_value_float:.3g})."
        f" Error RMS: {rms_diff_float:.3g}."
        f" RMS of result: {rms_result_float_orig_msg:.3g}, RMS of reference: {rms_reference_float_orig_msg:.3g}."
        f" Relative error RMS: {rel_rms_reference_float:.2%} of RMS of reference."
        f" Shape: {result.shape}"
    )
    if not jnp.allclose(result, reference, atol=atol, rtol=rtol):
        # Constructing a detailed failure message
        header = f"{operation_name}: result and reference are not close enough."
        atol_info = f"Absolute tolerance (atol): {atol}"
        rtol_info = f"Relative tolerance (rtol): {rtol}"
        
        # Calculate and format differences
        diff = jnp.abs(result - reference)
        max_abs_diff = jnp.max(diff)
        mean_abs_diff = jnp.mean(diff)
        
        # Ensure RMS values are Python floats for formatting
        rms_result_float = float(rms_result.item())
        rms_reference_float = float(rms_reference.item())

        # Format error statistics
        error_stats = (
            f" Max absolute difference: {max_abs_diff:.3g}. Mean absolute difference: {mean_abs_diff:.3g}."
            f" RMS of result: {rms_result_float:.3g}, RMS of reference: {rms_reference_float:.3g}."
        )
        
        # Combine all parts into the final message
        full_message = f"\n{header}\n{atol_info}\n{rtol_info}\n{error_stats}"
        
        raise AssertionError(full_message)


@torch.no_grad()
def from_torch(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.cpu().numpy())


@torch.no_grad()
def to_torchold(array: jnp.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array))

def to_torch(array: jnp.ndarray) -> torch.Tensor:
    if array.dtype == jnp.bfloat16:
        return torch.from_numpy(np.array(array.astype(jnp.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(array))


def checkify_forward(module):  # noqa: ANN001, ANN202
    return checkify(
        module.__call__,
        errors=index_checks | nan_checks | div_checks | user_checks,
    )
