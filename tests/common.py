import math

from jax import numpy as jnp
from jax.experimental.checkify import checkify, div_checks, nan_checks, user_checks

__all__ = ["assert_close", "checkify_forward"]

ATOL = 1e-3
RTOL = 0.03


def checkify_forward(module):  # noqa: ANN001, ANN201
    return checkify(
        module.__call__,
        errors=nan_checks | div_checks | user_checks,
    )


def assert_close(
    *,
    result: jnp.ndarray,
    reference: jnp.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
    fraction_of_allowed_violations: float = 0.0,
    operation_name: str | None = None,
) -> None:
    assert result.shape == reference.shape, (
        f"{operation_name} shapes do not match: {result.shape} != {reference.shape}"
    )
    assert result.dtype == reference.dtype, (
        f"{operation_name} data types do not match: {result.dtype} != {reference.dtype}"
    )

    result = result.astype(jnp.float32)
    reference = reference.astype(jnp.float32)

    absdiff = jnp.abs(result - reference)

    allowed_diff = atol + rtol * jnp.abs(reference)
    violations = jnp.maximum(absdiff - allowed_diff, 0)
    num_violations = jnp.count_nonzero(violations).item()
    max_allowed_violations = math.ceil(reference.size * fraction_of_allowed_violations)

    err_rel = absdiff / (jnp.abs(reference) + 1e-10)
    max_err_idx = tuple(i.item() for i in jnp.unravel_index(jnp.argmax(violations), absdiff.shape))
    max_err = absdiff[max_err_idx].item()
    max_err_rel = err_rel[max_err_idx].item()
    max_err_reference_value = reference[max_err_idx].item()

    rms_diff = jnp.sqrt(jnp.mean(jnp.square(absdiff))).item()
    rms_result = jnp.sqrt(jnp.mean(jnp.square(result))).item()
    rms_reference = jnp.sqrt(jnp.mean(jnp.square(reference))).item()
    rel_rms_reference = rms_diff / (rms_reference + 1e-10)

    if operation_name is not None:
        operation_description = f" during {operation_name}"
    else:
        operation_description = ""

    if max_allowed_violations > 0:
        allowed_violation_explainer = f" (Max {max_allowed_violations} allowed)"
    else:
        allowed_violation_explainer = ""

    message = (
        f"{num_violations} violations > {atol:.1e} + {rtol:.2%} out of total {reference.size} elements"
        f"{allowed_violation_explainer}{operation_description}."
        f" Worst violation: {max_err:.3g} ({max_err_rel:.2%}) at index {max_err_idx}"
        f" (reference value: {max_err_reference_value:.3g})."
        f" Error RMS: {rms_diff:.3g}."
        f" RMS of result: {rms_result:.3g}, RMS of reference: {rms_reference:.3g}."
        f" Relative error RMS: {rel_rms_reference:.2%} of RMS of reference."
        f" Shape: {result.shape}"
    )
    assert num_violations <= max_allowed_violations, message
