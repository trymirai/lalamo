from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from lalamo.main import CliDistillCallbacks, DistillRecipe, app
from lalamo.quantization import QuantizationMode


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
            "--devices",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert mock_distill.call_count == 1
    config = mock_distill.call_args.args[0]
    assert mock_distill.call_args.kwargs["callbacks_type"] is CliDistillCallbacks
    assert mock_infer_quantization_mode.call_args.args == (student_path,)
    assert config.teacher_path == teacher_path
    assert config.student_path == student_path
    assert config.dataset_path == dataset_path
    assert config.output_dir == output_dir
    assert config.num_devices == 2
    assert config.num_steps == 123
    assert config.quantization_mode == QuantizationMode.UINT8
    assert config.lora_rank == 32
    assert config.optimizer_name.value == "adamw"


@patch("lalamo.main._distill")
def test_distill_advanced_cli_invokes_runner(mock_distill: Mock, tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher"
    student_path = tmp_path / "student"
    dataset_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "distilled"
    teacher_path.mkdir()
    student_path.mkdir()
    dataset_path.write_bytes(b"parquet")

    result = CliRunner().invoke(
        app,
        [
            "distill-advanced",
            str(teacher_path),
            str(student_path),
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--num-steps",
            "1",
            "--warmup-steps",
            "8",
            "--gradient-clip-norm",
            "0.01",
            "--gradient-accumulation-steps",
            "4",
            "--lora-rank",
            "16",
            "--quantization-mode",
            QuantizationMode.UINT8.value,
        ],
    )

    assert result.exit_code == 0
    assert mock_distill.call_count == 1
    assert mock_distill.call_args.kwargs["callbacks_type"] is CliDistillCallbacks

    config = mock_distill.call_args.args[0]
    assert config.teacher_path == teacher_path
    assert config.student_path == student_path
    assert config.dataset_path == dataset_path
    assert config.output_dir == output_dir
    assert config.num_steps == 1
    assert config.warmup_steps == 8
    assert config.gradient_clip_norm == 0.01
    assert config.gradient_accumulation_steps == 4
    assert config.lora_rank == 16
    assert config.quantization_mode == QuantizationMode.UINT8


def test_distill_advanced_cli_requires_output_dir(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher"
    student_path = tmp_path / "student"
    dataset_path = tmp_path / "dataset.parquet"
    teacher_path.mkdir()
    student_path.mkdir()
    dataset_path.write_bytes(b"parquet")

    result = CliRunner().invoke(
        app,
        [
            "distill-advanced",
            str(teacher_path),
            str(student_path),
            str(dataset_path),
        ],
    )

    assert result.exit_code != 0
    assert "Missing option '--output-dir'" in result.output
