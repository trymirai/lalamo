from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evals.types import InferenceOutput

from lalamo.evals.benchmark.command import benchmark_command_handler


def create_predictions_parquet(
    path: Path,
    records: list[dict],
    metadata: dict[str, str] | None = None,
) -> None:
    df = pl.DataFrame(records)
    table = pa.Table.from_pandas(df.to_pandas())

    if metadata:
        schema_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
        new_schema = table.schema.with_metadata(schema_metadata)
        table = table.cast(new_schema)

    pq.write_table(table, path)


class TestBenchmarkMetadataValidation:
    def test_extracts_metadata_correctly(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"
        records = [
            {
                "id": "test_1",
                "question": "What is 2+2?",
                "model_output": "4",
                "chain_of_thought": None,
                "answer": "4",
                "metadata": None,
            }
        ]
        metadata = {
            "model_name": "llama-3.2-1b",
            "inference_engine": "local",
        }
        create_predictions_parquet(predictions_path, records, metadata)

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.prepare_for_benchmark.return_value = tmp_path / "prepared.parquet"
        mock_adapter.run_benchmark.return_value = {"accuracy": 0.95}

        with patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

            mock_adapter.run_benchmark.assert_called_once()
            call_kwargs = mock_adapter.run_benchmark.call_args.kwargs
            assert call_kwargs["model_name"] == "llama-3.2-1b"
            assert call_kwargs["inference_engine"] == "local"

    def test_fails_on_missing_model_name_metadata(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"
        records = [
            {
                "id": "test_1",
                "question": "What is 2+2?",
                "model_output": "4",
                "chain_of_thought": None,
                "answer": "4",
                "metadata": None,
            }
        ]
        metadata = {"inference_engine": "local"}
        create_predictions_parquet(predictions_path, records, metadata)

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            pytest.raises(ValueError, match="missing required metadata: model_name"),
        ):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

    def test_fails_on_missing_inference_engine_metadata(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"
        records = [
            {
                "id": "test_1",
                "question": "What is 2+2?",
                "model_output": "4",
                "chain_of_thought": None,
                "answer": "4",
                "metadata": None,
            }
        ]
        metadata = {"model_name": "llama-3.2-1b"}
        create_predictions_parquet(predictions_path, records, metadata)

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            pytest.raises(ValueError, match="missing required metadata: inference_engine"),
        ):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )


class TestBenchmarkErrorHandling:
    def test_fails_on_missing_file(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "nonexistent.parquet"

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            pytest.raises(FileNotFoundError, match="Predictions file not found"),
        ):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

    def test_fails_on_empty_predictions(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"
        create_predictions_parquet(
            predictions_path,
            [],
            metadata={"model_name": "test", "inference_engine": "local"},
        )

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            pytest.raises(ValueError, match="Predictions file is empty"),
        ):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

    def test_fails_on_missing_required_columns(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"
        records = [{"id": "test_1", "model_output": "4"}]
        create_predictions_parquet(
            predictions_path,
            records,
            metadata={"model_name": "test", "inference_engine": "local"},
        )

        mock_adapter = Mock()
        mock_adapter.get_benchmark_split.return_value = "test"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            pytest.raises(ValueError, match="Missing required columns"),
        ):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

    def test_fails_on_unknown_eval_name(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.parquet"

        with (
            patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {}),
            pytest.raises(ValueError, match="Unknown eval.*Available evals"),
        ):
            benchmark_command_handler(
                eval_name="unknown/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )
