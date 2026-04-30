import json
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from lalamo.distill_runner import ComputeDTypeName, DistillResult, EvaluationMetrics, OptimizerName
from lalamo.main import DistillRecipe, app
from lalamo.quantization import QuantizationMode


def _distill_result(output_dir: Path) -> DistillResult:
    metrics = EvaluationMetrics(kl_divergence=0.0, top1_agreement=1.0, valid_tokens=1)
    return DistillResult(
        completed_steps=1,
        initial_eval=metrics,
        final_eval=metrics,
        compilation_seconds=0.0,
        elapsed_seconds=0.0,
        seconds_per_step=0.0,
        output_model_path=output_dir / "student-distilled",
    )


@patch("lalamo.main._distill")
@patch("lalamo.main._infer_student_quantization_mode", return_value=QuantizationMode.UINT8)
def test_distill_cli_invokes_recipe(
    mock_infer_quantization_mode: Mock,
    mock_distill: Mock,
    tmp_path: Path,
) -> None:
    teacher_path = tmp_path / "teacher"
    student_path = tmp_path / "student"
    dataset_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "distilled"
    teacher_path.mkdir()
    student_path.mkdir()
    dataset_path.write_bytes(b"parquet")
    mock_distill.return_value = _distill_result(output_dir)

    result = CliRunner().invoke(
        app,
        [
            "distill",
            str(teacher_path),
            str(student_path),
            str(dataset_path),
            "--recipe",
            DistillRecipe.QLORA.value,
            "--steps",
            "123",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    config = mock_distill.call_args.args[0]
    assert callable(mock_distill.call_args.kwargs["progress_callback"])
    assert mock_infer_quantization_mode.call_args.args == (student_path,)
    assert config.teacher_path == teacher_path
    assert config.student_path == student_path
    assert config.dataset_path == dataset_path
    assert config.output_dir == output_dir
    assert config.num_steps == 123
    assert config.quantization_mode == QuantizationMode.UINT8
    assert config.lora_rank == 32
    assert config.optimizer_name == OptimizerName.ADAMW


@patch("lalamo.main._distill")
def test_distill_config_cli_invokes_runner(mock_distill: Mock, tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher"
    student_path = tmp_path / "student"
    dataset_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "distilled"
    config_path = tmp_path / "distill.json"
    teacher_path.mkdir()
    student_path.mkdir()
    dataset_path.write_bytes(b"parquet")
    mock_distill.return_value = _distill_result(output_dir)
    config_path.write_text(
        json.dumps(
            {
                "teacher_path": str(teacher_path),
                "student_path": str(student_path),
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "num_steps": 10,
                "learning_rate": 0.0001,
                "gradient_clip_norm": 0.5,
                "optimizer_name": "adamw",
                "quantization_mode": "uint8",
                "lora_rank": 4,
                "compute_dtype_name": "float32",
                "stochastic_rounding": False,
            },
        ),
    )

    result = CliRunner().invoke(app, ["distill-config", str(config_path)])

    assert result.exit_code == 0
    config = mock_distill.call_args.args[0]
    assert config.teacher_path == teacher_path
    assert config.student_path == student_path
    assert config.dataset_path == dataset_path
    assert config.output_dir == output_dir
    assert config.num_steps == 10
    assert config.learning_rate == 0.0001
    assert config.gradient_clip_norm == 0.5
    assert config.optimizer_name == OptimizerName.ADAMW
    assert config.quantization_mode == QuantizationMode.UINT8
    assert config.lora_rank == 4
    assert config.compute_dtype_name == ComputeDTypeName.FLOAT32
    assert not config.stochastic_rounding


def test_distill_config_example_prints_valid_json() -> None:
    result = CliRunner().invoke(app, ["distill-config-example"])

    assert result.exit_code == 0
    config = json.loads(result.output)
    assert config["teacher_path"] == "models/teacher"
    assert config["optimizer_name"] == OptimizerName.MUON.value
    assert config["quantization_mode"] == QuantizationMode.UINT4.value


@patch("lalamo.main._distill")
def test_distill_config_cli_rejects_wrong_json_types(mock_distill: Mock, tmp_path: Path) -> None:
    config_path = tmp_path / "distill.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_path": "teacher",
                "student_path": "student",
                "dataset_path": "dataset.parquet",
                "output_dir": "distilled",
                "stochastic_rounding": "false",
            },
        ),
    )

    result = CliRunner().invoke(app, ["distill-config", str(config_path)])

    assert result.exit_code != 0
    assert "stochastic_rounding" in result.output
    mock_distill.assert_not_called()
