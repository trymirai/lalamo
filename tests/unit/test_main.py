from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from lalamo.main import CliDistillCallbacks, app
from lalamo.quantization import QuantizationMode


@patch("lalamo.main._distill")
def test_distill_cli_invokes_runner(mock_distill: Mock, tmp_path: Path) -> None:
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
            "--output-dir",
            str(output_dir),
            "--num-steps",
            "1",
            "--lora-rank",
            "16",
            "--bits",
            "8",
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
    assert config.lora_rank == 16
    assert config.quantization_mode == QuantizationMode.UINT8
