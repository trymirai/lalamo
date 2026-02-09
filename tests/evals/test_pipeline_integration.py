from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pyarrow.parquet as pq
from evals.types import DatasetLoadConfig, InferenceConfig, InferenceOutput, InternalEvalRecord

from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.inference.command import infer_command_handler
from lalamo.evals.inference.engines import LalamoEngineConfig


class TestFullPipelineIntegration:
    def test_convert_infer_benchmark_pipeline(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        infer_output_dir = tmp_path / "infer_output"

        test_records = [
            InternalEvalRecord(
                id="q1",
                question="What is the capital of France?",
                answer="Paris",
                metadata={"category": "geography"},
            ),
            InternalEvalRecord(
                id="q2",
                question="What is 5+3?",
                answer="8",
                metadata={"category": "math"},
            ),
        ]

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(
            temperature=0.0,
            max_output_length=2048,
        )
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = []
        mock_adapter.prepare_for_benchmark.return_value = tmp_path / "prepared.parquet"
        mock_adapter.run_benchmark.return_value = {"accuracy": 1.0}

        mock_handler_class = Mock()
        mock_handler_class.download_split = Mock(return_value=test_records)

        eval_spec = Mock()
        eval_spec.name = "test_eval"
        eval_spec.repo = "test/repo"
        eval_spec.splits = ["test"]
        eval_spec.handler_type = mock_handler_class

        with patch("lalamo.evals.datasets.command.REPO_TO_EVAL", {"test/repo": eval_spec}):
            convert_dataset_handler(
                eval_repo="test/repo",
                output_dir=dataset_dir,
                callbacks=Mock(),
            )

        assert (dataset_dir / "test.parquet").exists()
        assert (dataset_dir / "config.json").exists()

        converted_df = pl.read_parquet(dataset_dir / "test.parquet")
        assert len(converted_df) == 2
        assert set(converted_df.columns) == {"id", "question", "answer", "metadata"}
        assert converted_df["id"].to_list() == ["q1", "q2"]

        mock_engine = Mock()
        mock_engine.run_inference.return_value = tmp_path / "raw_output.parquet"
        mock_engine.parse_output.return_value = [
            InferenceOutput(
                id="q1",
                response="Paris",
                question="What is the capital of France?",
                answer="Paris",
                metadata={"category": "geography"},
            ),
            InferenceOutput(
                id="q2",
                response="8",
                question="What is 5+3?",
                answer="8",
                metadata={"category": "math"},
            ),
        ]

        eval_spec.handler_type = lambda: mock_adapter

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": eval_spec}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("test-model"),
                batch_size=1,
                vram_gb=None,
            )

            predictions_path = infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=infer_output_dir,
                engine_config=engine_config,
                inference_overrides=InferenceConfig(),
            )

        assert predictions_path.exists()

        predictions_df = pl.read_parquet(predictions_path)
        assert len(predictions_df) == 2
        assert set(predictions_df.columns) == {
            "id",
            "question",
            "model_output",
            "chain_of_thought",
            "answer",
            "metadata",
        }
        assert predictions_df["model_output"].to_list() == ["Paris", "8"]
        assert predictions_df["answer"].to_list() == ["Paris", "8"]

        table = pq.read_table(predictions_path)
        schema_metadata = table.schema.metadata
        assert b"model_name" in schema_metadata
        assert b"inference_engine" in schema_metadata
        assert schema_metadata[b"model_name"] == b"test-model"

        eval_spec.handler_type = lambda: mock_adapter

        with patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": eval_spec}):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

            mock_adapter.prepare_for_benchmark.assert_called_once()
            mock_adapter.run_benchmark.assert_called_once()

            benchmark_call = mock_adapter.run_benchmark.call_args
            assert benchmark_call.kwargs["model_name"] == "test-model"
            assert benchmark_call.kwargs["inference_engine"] == "local"
