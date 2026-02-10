"""Integration tests for the full evals pipeline: convert → infer → benchmark.

These tests validate that data formats and metadata flow correctly through
all three stages of the evaluation workflow using real parquet I/O operations.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pyarrow.parquet as pq
from evals.types import (
    DatasetLoadConfig,
    EvalPrompt,
    InferenceConfig,
    InferenceOutput,
    InternalEvalRecord,
    MessageRole,
    PromptMessage,
)

from lalamo.evals.benchmark.command import benchmark_command_handler
from lalamo.evals.datasets.command import convert_dataset_handler
from lalamo.evals.inference.command import infer_command_handler
from lalamo.evals.inference.engines import CustomAPIEngineConfig, LalamoEngineConfig


class TestFullPipelineIntegration:
    def test_convert_infer_benchmark_pipeline(self, tmp_path: Path) -> None:
        """Test full pipeline with lalamo engine: convert → infer → benchmark.

        Validates:
        - Dataset conversion creates correct parquet schema
        - Inference enriches predictions with ground truth
        - Metadata propagates through all stages
        - Benchmark receives properly formatted predictions
        """
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

        test_prompts = [
            EvalPrompt(
                id="q1",
                messages=[
                    PromptMessage(role=MessageRole.USER, content="What is the capital of France?"),
                ],
            ),
            EvalPrompt(
                id="q2",
                messages=[
                    PromptMessage(role=MessageRole.USER, content="What is 5+3?"),
                ],
            ),
        ]

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(
            temperature=0.0,
            max_output_length=2048,
        )
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = test_prompts
        mock_adapter.prepare_for_benchmark.return_value = tmp_path / "prepared.parquet"
        mock_adapter.run_benchmark.return_value = {"accuracy": 1.0}

        mock_handler_class = Mock()
        mock_handler_class.download_split = Mock(return_value=test_records)

        eval_spec = Mock()
        eval_spec.name = "test_eval"
        eval_spec.repo = "test/repo"
        eval_spec.splits = ["test"]
        eval_spec.handler_type = mock_handler_class

        # Step 1: Convert dataset from external format to internal parquet
        with patch("lalamo.evals.datasets.command.REPO_TO_EVAL", {"test/repo": eval_spec}):
            convert_dataset_handler(
                eval_repo="test/repo",
                output_dir=dataset_dir,
                callbacks=Mock(),
            )

        # Verify conversion output format
        assert (dataset_dir / "test.parquet").exists()
        assert (dataset_dir / "config.json").exists()

        converted_df = pl.read_parquet(dataset_dir / "test.parquet")
        assert len(converted_df) == 2
        assert set(converted_df.columns) == {"id", "question", "answer", "metadata"}
        assert converted_df["id"].to_list() == ["q1", "q2"]

        # Step 2: Run inference on converted dataset
        def mock_run_inference(input_path, output_path, callbacks):
            # Create mock output parquet with expected schema
            output_data = pl.DataFrame({
                "response": ["Paris", "8"],
                "chain_of_thought": ["", ""],
            })
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data.write_parquet(output_path)
            return output_path

        mock_engine = Mock()
        mock_engine.get_conversation_column_name.return_value = "conversation"
        mock_engine.run_inference.side_effect = mock_run_inference

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
                inference_overrides={},
            )

        assert predictions_path.exists()

        # Verify predictions format and ground truth enrichment
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

        # Verify metadata in parquet schema
        table = pq.read_table(predictions_path)
        schema_metadata = table.schema.metadata
        assert b"model_name" in schema_metadata
        assert b"inference_engine" in schema_metadata
        assert schema_metadata[b"model_name"] == b"test-model"

        # Step 3: Run benchmark on predictions
        eval_spec.handler_type = lambda: mock_adapter

        with patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": eval_spec}):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

            # Verify benchmark received correct metadata
            mock_adapter.prepare_for_benchmark.assert_called_once()
            mock_adapter.run_benchmark.assert_called_once()

            benchmark_call = mock_adapter.run_benchmark.call_args
            assert benchmark_call.kwargs["model_name"] == "test-model"
            assert benchmark_call.kwargs["inference_engine"] == "local"

    def test_pipeline_with_custom_api_engine(self, tmp_path: Path) -> None:
        """Test full pipeline with custom API engine: convert → infer → benchmark.

        Validates the same workflow as lalamo engine but with:
        - Custom API engine (different code path)
        - Different model naming (API model name vs file path)
        - Different engine type metadata (custom_api vs local)
        """
        dataset_dir = tmp_path / "dataset"
        infer_output_dir = tmp_path / "infer_output"

        test_records = [
            InternalEvalRecord(
                id="q1",
                question="What is the capital of Spain?",
                answer="Madrid",
                metadata={"category": "geography"},
            ),
            InternalEvalRecord(
                id="q2",
                question="What is 10-3?",
                answer="7",
                metadata={"category": "math"},
            ),
        ]

        test_prompts = [
            EvalPrompt(
                id="q1",
                messages=[
                    PromptMessage(role=MessageRole.USER, content="What is the capital of Spain?"),
                ],
            ),
            EvalPrompt(
                id="q2",
                messages=[
                    PromptMessage(role=MessageRole.USER, content="What is 10-3?"),
                ],
            ),
        ]

        mock_adapter = Mock()
        mock_adapter.get_inference_config.return_value = InferenceConfig(
            temperature=0.7,
            max_output_length=1024,
        )
        mock_adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
        mock_adapter.get_benchmark_split.return_value = "test"
        mock_adapter.format_prompts.return_value = test_prompts
        mock_adapter.prepare_for_benchmark.return_value = tmp_path / "prepared.parquet"
        mock_adapter.run_benchmark.return_value = {"accuracy": 1.0}

        mock_handler_class = Mock()
        mock_handler_class.download_split = Mock(return_value=test_records)

        eval_spec = Mock()
        eval_spec.name = "test_eval"
        eval_spec.repo = "test/repo"
        eval_spec.splits = ["test"]
        eval_spec.handler_type = mock_handler_class

        # Step 1: Convert dataset
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
        assert converted_df["id"].to_list() == ["q1", "q2"]

        # Step 2: Run inference with custom API engine
        def mock_run_inference(input_path, output_path, callbacks):
            # Create mock output parquet with expected schema
            output_data = pl.DataFrame({
                "response": ["Madrid", "7"],
                "chain_of_thought": ["", ""],
            })
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data.write_parquet(output_path)
            return output_path

        mock_engine = Mock()
        mock_engine.get_conversation_column_name.return_value = "messages"
        mock_engine.run_inference.side_effect = mock_run_inference

        eval_spec.handler_type = lambda: mock_adapter

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": eval_spec}),
            patch("lalamo.evals.inference.command.CustomAPIInferenceEngine", return_value=mock_engine),
        ):
            engine_config = CustomAPIEngineConfig(
                base_url="http://localhost:11434/v1",
                model="qwen2.5-coder:0.5b",
                api_key=None,
                max_retries=3,
                timeout=120.0,
            )

            predictions_path = infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=infer_output_dir,
                engine_config=engine_config,
                inference_overrides={},
            )

        assert predictions_path.exists()

        # Verify predictions format
        predictions_df = pl.read_parquet(predictions_path)
        assert len(predictions_df) == 2
        assert predictions_df["model_output"].to_list() == ["Madrid", "7"]
        assert predictions_df["answer"].to_list() == ["Madrid", "7"]

        # Verify API engine metadata
        table = pq.read_table(predictions_path)
        schema_metadata = table.schema.metadata
        assert b"model_name" in schema_metadata
        assert b"inference_engine" in schema_metadata
        assert schema_metadata[b"model_name"] == b"qwen2.5-coder:0.5b"
        assert schema_metadata[b"inference_engine"] == b"custom_api"

        # Step 3: Run benchmark
        eval_spec.handler_type = lambda: mock_adapter

        with patch("lalamo.evals.benchmark.command.REPO_TO_EVAL", {"test/repo": eval_spec}):
            benchmark_command_handler(
                eval_name="test/repo",
                predictions_path=predictions_path,
                callbacks=Mock(),
            )

            # Verify benchmark received custom_api metadata
            mock_adapter.prepare_for_benchmark.assert_called_once()
            mock_adapter.run_benchmark.assert_called_once()

            benchmark_call = mock_adapter.run_benchmark.call_args
            assert benchmark_call.kwargs["model_name"] == "qwen2.5-coder:0.5b"
            assert benchmark_call.kwargs["inference_engine"] == "custom_api"
