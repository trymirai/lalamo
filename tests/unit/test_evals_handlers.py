from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("evals")
pytest.importorskip("openai")

from evals.types import DatasetLoadConfig, EvalPrompt, InferenceConfig, InferenceOutput, InternalEvalRecord, MessageRole, PromptMessage

from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.inference.command import infer_command_handler
from lalamo.evals.inference.engines import LalamoEngineConfig


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
                eval_repo="test/repo",
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
                eval_repo="test/repo",
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
                eval_repo="test/repo",
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
            pytest.raises(ValueError, match="Predictions file not found"),
        ):
            benchmark_command_handler(
                eval_repo="test/repo",
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
                eval_repo="test/repo",
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
                eval_repo="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )



class TestInferCommandHandlerMetadata:
    def test_writes_metadata_to_predictions_parquet(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        output_dir = tmp_path / "output"

        test_records = [
            InternalEvalRecord(
                id="test_1",
                question="What is 2+2?",
                answer="4",
                metadata=None,
            )
        ]
        df = pl.DataFrame([{"id": r.id, "question": r.question, "answer": r.answer, "metadata": r.metadata} for r in test_records])
        df.write_parquet(dataset_dir / "test.parquet")

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(
            temperature=0.0,
            max_output_length=2048,
        )
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = [
            EvalPrompt(
                id="test_1",
                messages=[PromptMessage(role=MessageRole.USER, content="What is 2+2?")],
            )
        ]

        def mock_run_inference(input_path: Path, output_path: Path, callbacks) -> None:
            # Create a mock output parquet with expected columns
            output_df = pl.DataFrame({
                "response": ["4"],
                "chain_of_thought": [None],
            })
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_parquet(output_path)

        mock_engine = Mock()
        mock_engine.get_conversation_column_name.return_value = "messages"
        mock_engine.run_inference.side_effect = mock_run_inference
        mock_engine.parse_output.return_value = [
            InferenceOutput(
                id="test_1",
                response="4",
                question="What is 2+2?",
                answer="4",
            )
        ]

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("test-model"),
                batch_size=1,
                vram_gb=None,
            )

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                engine_config=engine_config,
                inference_overrides={},
            )

            predictions_path = output_dir / "predictions.parquet"
            assert predictions_path.exists()

            table = pq.read_table(predictions_path)
            schema_metadata = table.schema.metadata or {}

            assert b"model_name" in schema_metadata
            assert b"inference_engine" in schema_metadata
            assert schema_metadata[b"model_name"] == b"test-model"
            assert schema_metadata[b"inference_engine"] == b"local"

    def test_enriches_predictions_with_ground_truth(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        output_dir = tmp_path / "output"

        test_records = [
            InternalEvalRecord(
                id="test_1",
                question="What is 2+2?",
                answer="4",
                metadata={"category": "math"},
            ),
            InternalEvalRecord(
                id="test_2",
                question="What is 3+3?",
                answer="6",
                metadata={"category": "math"},
            ),
        ]
        df = pl.DataFrame([{"id": r.id, "question": r.question, "answer": r.answer, "metadata": r.metadata} for r in test_records])
        df.write_parquet(dataset_dir / "test.parquet")

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(max_output_length=2048)
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = [
            EvalPrompt(
                id="test_1",
                messages=[PromptMessage(role=MessageRole.USER, content="What is 2+2?")],
            ),
            EvalPrompt(
                id="test_2",
                messages=[PromptMessage(role=MessageRole.USER, content="What is 3+3?")],
            ),
        ]

        def mock_run_inference(input_path: Path, output_path: Path, callbacks) -> None:
            # Create a mock output parquet with expected columns
            output_df = pl.DataFrame({
                "response": ["4", "6"],
                "chain_of_thought": [None, None],
            })
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_parquet(output_path)

        mock_engine = Mock()
        mock_engine.get_conversation_column_name.return_value = "messages"
        mock_engine.run_inference.side_effect = mock_run_inference
        mock_engine.parse_output.return_value = [
            InferenceOutput(
                id="test_1",
                response="4",
                question="What is 2+2?",
                answer="4",
                metadata={"category": "math"},
            ),
            InferenceOutput(
                id="test_2",
                response="6",
                question="What is 3+3?",
                answer="6",
                metadata={"category": "math"},
            ),
        ]

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("test-model"),
                batch_size=1,
                vram_gb=None,
            )

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                engine_config=engine_config,
                inference_overrides={},
            )

            predictions_path = output_dir / "predictions.parquet"
            df = pl.read_parquet(predictions_path)

            assert "answer" in df.columns
            assert "question" in df.columns
            assert df["answer"].to_list() == ["4", "6"]
            assert df["question"].to_list() == ["What is 2+2?", "What is 3+3?"]


class TestInferCommandHandlerDatasetLoading:
    def test_loads_dataset_with_limit(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        output_dir = tmp_path / "output"

        test_records = [
            InternalEvalRecord(id=f"test_{i}", question=f"Q{i}?", answer=f"A{i}")
            for i in range(100)
        ]
        df = pl.DataFrame([{"id": r.id, "question": r.question, "answer": r.answer, "metadata": r.metadata} for r in test_records])
        df.write_parquet(dataset_dir / "test.parquet")

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(max_output_length=2048)
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=10)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = [
            EvalPrompt(
                id=f"test_{i}",
                messages=[PromptMessage(role=MessageRole.USER, content=f"Q{i}?")],
            )
            for i in range(10)
        ]

        def mock_run_inference(input_path: Path, output_path: Path, callbacks) -> None:
            # Create a mock output parquet with expected columns (10 records for limit test)
            output_df = pl.DataFrame({
                "response": [f"A{i}" for i in range(10)],
                "chain_of_thought": [None] * 10,
            })
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_parquet(output_path)

        mock_engine = Mock()
        mock_engine.get_conversation_column_name.return_value = "messages"
        mock_engine.run_inference.side_effect = mock_run_inference
        mock_engine.parse_output.return_value = []

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("test-model"),
                batch_size=1,
                vram_gb=None,
            )

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                engine_config=engine_config,
                inference_overrides={},
                limit=10,
            )

            call_args = mock_adapter.format_prompts.call_args[0][0]
            loaded_records = call_args["test"]
            assert len(loaded_records) == 10
