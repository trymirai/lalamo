from pathlib import Path

import jax
import jax.numpy as jnp
from typer.testing import CliRunner

from lalamo.commands import compare_traces
from lalamo.main import app
from lalamo.safetensors import safe_write
from lalamo.trace_comparator import compare_trace_files
from tests.conftest import RunLalamo


def _write_trace(path: Path, arrays: dict[str, jnp.ndarray]) -> None:
    with path.open("wb") as fd:
        safe_write(fd, arrays)


def test_compare_trace_files_compares_shared_arrays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(
        reference_path,
        {
            "activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "activation_trace.output_norm": jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
        },
    )
    _write_trace(
        result_path,
        {
            "activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "activation_trace.output_norm": jnp.asarray([[1.0, 2.001]], dtype=jnp.float32),
            "result_only": jnp.asarray([1], dtype=jnp.int32),
        },
    )

    comparison = compare_trace_files(reference_path, result_path)

    assert comparison.passed
    assert comparison.reference_only_paths == ()
    assert comparison.result_only_paths == ("result_only",)


def test_compare_trace_files_fails_missing_result_arrays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(
        reference_path,
        {
            "activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "activation_trace.output_norm": jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
        },
    )
    _write_trace(result_path, {"activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert comparison.reference_only_paths == ("activation_trace.output_norm",)


def test_compare_trace_files_reports_failures(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32)})
    _write_trace(result_path, {"activation_trace.token_ids": jnp.asarray([[1, 3]], dtype=jnp.int32)})

    comparison = compare_traces(reference_path, result_path)

    (failed_result,) = comparison.failed_results
    assert failed_result.path == "activation_trace.token_ids"
    assert not comparison.passed


def test_compare_trace_files_uses_bfloat16_tolerance(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    reference = jnp.zeros((100,), dtype=jnp.bfloat16)
    result = reference.at[:2].set(jnp.asarray(0.03, dtype=jnp.bfloat16))
    _write_trace(reference_path, {"activation_trace.output_norm": reference})
    _write_trace(result_path, {"activation_trace.output_norm": result})

    comparison = compare_trace_files(reference_path, result_path)

    assert comparison.passed
    assert "atol=0.04" in comparison.results[0].message


def test_compare_trace_files_reports_float_dtype_mismatches(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.output_norm": jnp.asarray([1.0], dtype=jnp.float32)})
    _write_trace(result_path, {"activation_trace.output_norm": jnp.asarray([1.0], dtype=jnp.bfloat16)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert "dtype mismatch" in comparison.failed_results[0].message


def test_compare_trace_files_rejects_single_bfloat16_outlier(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    reference = jnp.zeros((34,), dtype=jnp.bfloat16)
    result = reference.at[0].set(jnp.asarray(0.5, dtype=jnp.bfloat16))
    _write_trace(reference_path, {"activation_trace.output_norm": reference})
    _write_trace(result_path, {"activation_trace.output_norm": result})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert "1 violations" in comparison.failed_results[0].message


def test_compare_trace_files_keeps_float32_tolerance_strict(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    reference = jnp.zeros((100,), dtype=jnp.float32)
    result = reference.at[:2].set(0.03)
    _write_trace(reference_path, {"activation_trace.output_norm": reference})
    _write_trace(result_path, {"activation_trace.output_norm": result})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert "atol=0.01" in comparison.failed_results[0].message


def test_compare_trace_files_rejects_tiny_float_mismatches(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.output_norm": jnp.asarray([0.0], dtype=jnp.float32)})
    _write_trace(result_path, {"activation_trace.output_norm": jnp.asarray([0.03], dtype=jnp.float32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert "1 violations" in comparison.failed_results[0].message


def test_compare_trace_files_handles_empty_float_arrays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.output_norm": jnp.asarray([], dtype=jnp.float32)})
    _write_trace(result_path, {"activation_trace.output_norm": jnp.asarray([], dtype=jnp.float32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert comparison.passed
    assert "0 elements" in comparison.results[0].message


def test_compare_trace_files_rejects_nonfinite_values(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    with jax.debug_nans(new_val=False):
        _write_trace(reference_path, {"activation_trace.output_norm": jnp.asarray([jnp.nan], dtype=jnp.float32)})
    _write_trace(result_path, {"activation_trace.output_norm": jnp.asarray([0.0], dtype=jnp.float32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert "reference_nonfinite=True" in comparison.failed_results[0].message


def test_compare_trace_files_reports_dtype_mismatches(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32)})
    _write_trace(result_path, {"activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.uint32)})

    comparison = compare_trace_files(reference_path, result_path)

    (failed_result,) = comparison.failed_results
    assert "dtype mismatch" in failed_result.message


def test_compare_trace_files_reports_shape_mismatches(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.output_norm": jnp.zeros((1, 2), dtype=jnp.float32)})
    _write_trace(result_path, {"activation_trace.output_norm": jnp.zeros((2, 1), dtype=jnp.float32)})

    comparison = compare_trace_files(reference_path, result_path)

    (failed_result,) = comparison.failed_results
    assert "shape mismatch" in failed_result.message


def test_compare_trace_files_requires_shared_arrays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"reference": jnp.asarray([1], dtype=jnp.int32)})
    _write_trace(result_path, {"result": jnp.asarray([1], dtype=jnp.int32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert comparison.reference_only_paths == ("reference",)


def test_compare_trace_files_rejects_empty_reference(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {})
    _write_trace(result_path, {"result": jnp.asarray([1], dtype=jnp.int32)})

    comparison = compare_trace_files(reference_path, result_path)

    assert not comparison.passed
    assert comparison.results == ()
    assert comparison.result_only_paths == ("result",)


def test_compare_traces_cli(tmp_path: Path, run_lalamo: RunLalamo) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(
        reference_path,
        {
            "activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
        },
    )
    _write_trace(
        result_path,
        {
            "activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "result_only": jnp.asarray([1], dtype=jnp.int32),
        },
    )

    output = run_lalamo("compare-traces", str(reference_path), str(result_path))

    assert "1 shared arrays" in output
    assert "Overcaptured result arrays" in output
    assert "result_only" in output


def test_compare_traces_cli_fails_on_mismatched_arrays(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.safetensors"
    result_path = tmp_path / "result.safetensors"
    _write_trace(reference_path, {"activation_trace.token_ids": jnp.asarray([[1, 2]], dtype=jnp.int32)})
    _write_trace(result_path, {"activation_trace.token_ids": jnp.asarray([[1, 3]], dtype=jnp.int32)})

    result = CliRunner().invoke(app, ["compare-traces", str(reference_path), str(result_path)], terminal_width=240)

    assert result.exit_code == 1
    assert "1 failed" in result.output
    assert "activation_trace.token_ids" in result.output
