from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array

from lalamo.safetensors import safe_read

ATOL = 1e-2
RTOL = 0.03
BF16_ATOL = 0.04
BF16_RTOL = 0.06


@dataclass(frozen=True)
class TraceArrayComparison:
    path: str
    passed: bool
    message: str


@dataclass(frozen=True)
class TraceComparison:
    results: tuple[TraceArrayComparison, ...]
    reference_only_paths: tuple[str, ...]
    result_only_paths: tuple[str, ...]

    @property
    def failed_results(self) -> tuple[TraceArrayComparison, ...]:
        return tuple(result for result in self.results if not result.passed)

    @property
    def passed(self) -> bool:
        return bool(self.results) and not self.failed_results and not self.reference_only_paths


def compare_trace_files(reference_path: Path, result_path: Path) -> TraceComparison:
    with jax.debug_nans(new_val=False), reference_path.open("rb") as reference_fd, result_path.open("rb") as result_fd:
        _, reference_arrays = safe_read(reference_fd)
        _, result_arrays = safe_read(result_fd)

        reference_paths = set(reference_arrays)
        result_paths = set(result_arrays)
        common_paths = tuple(sorted(reference_paths & result_paths))
        return TraceComparison(
            results=tuple(
                _compare_array(
                    path,
                    reference_arrays[path],
                    result_arrays[path],
                )
                for path in common_paths
            ),
            reference_only_paths=tuple(sorted(reference_paths - result_paths)),
            result_only_paths=tuple(sorted(result_paths - reference_paths)),
        )


def _compare_array(path: str, reference: Array, result: Array) -> TraceArrayComparison:
    reference = jnp.asarray(jax.device_get(reference))
    result = jnp.asarray(jax.device_get(result))
    dtype_message = f"dtype {reference.dtype} vs {result.dtype}"

    if reference.shape != result.shape:
        return TraceArrayComparison(
            path=path,
            passed=False,
            message=f"shape mismatch: {reference.shape} != {result.shape}; {dtype_message}",
        )

    if reference.dtype != result.dtype:
        return TraceArrayComparison(
            path=path,
            passed=False,
            message=f"dtype mismatch: {dtype_message}",
        )

    if jnp.issubdtype(reference.dtype, jnp.floating):
        return _compare_floating_array(
            path,
            reference.astype(jnp.float32),
            result.astype(jnp.float32),
            dtype_message,
            _floating_tolerance(reference.dtype, result.dtype),
        )

    violations = reference != result
    num_violations = int(jnp.count_nonzero(violations).item())
    return TraceArrayComparison(
        path=path,
        passed=num_violations == 0,
        message=f"{num_violations} exact mismatches out of {reference.size} elements; {dtype_message}",
    )


def _floating_tolerance(reference_dtype: jnp.dtype, result_dtype: jnp.dtype) -> tuple[float, float]:
    if jnp.bfloat16 in (reference_dtype, result_dtype):
        return (BF16_ATOL, BF16_RTOL)
    return (ATOL, RTOL)


def _compare_floating_array(
    path: str,
    reference: Array,
    result: Array,
    dtype_message: str,
    tolerance: tuple[float, float],
) -> TraceArrayComparison:
    atol, rtol = tolerance
    if reference.size == 0:
        return TraceArrayComparison(
            path=path,
            passed=True,
            message=f"0 violations out of 0 elements; {dtype_message}",
        )

    absdiff = jnp.abs(reference - result)
    allowed_diff = atol + rtol * jnp.abs(reference)
    violations = absdiff > allowed_diff
    num_violations = int(jnp.count_nonzero(violations).item())
    max_error_index = tuple(i.item() for i in jnp.unravel_index(jnp.argmax(absdiff), absdiff.shape))
    max_error = float(absdiff[max_error_index].item())
    rms_error = float(jnp.sqrt(jnp.mean(jnp.square(absdiff))).item())
    reference_nonfinite = bool(jnp.any(~jnp.isfinite(reference)).item())
    result_nonfinite = bool(jnp.any(~jnp.isfinite(result)).item())

    return TraceArrayComparison(
        path=path,
        passed=num_violations == 0 and not reference_nonfinite and not result_nonfinite,
        message=(
            f"{num_violations} violations out of {reference.size} elements; "
            f"atol={atol:.3g}; rtol={rtol:.3g}; "
            f"worst {max_error:.3g} at {max_error_index}; "
            f"rms {rms_error:.3g}; reference_nonfinite={reference_nonfinite}; "
            f"result_nonfinite={result_nonfinite}; {dtype_message}"
        ),
    )
