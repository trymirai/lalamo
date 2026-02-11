from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

pytest.importorskip("evals")
pytest.importorskip("openai")

from evals.types import (
    DatasetLoadConfig,
    EvalPrompt,
    InferenceConfig,
    InferenceEngineType,
    InternalEvalRecord,
    MessageRole,
    PromptMessage,
)

from lalamo.evals.inference.command import infer_command_handler
from lalamo.evals.inference.engines import CustomAPIEngineConfig, LalamoEngineConfig


@pytest.fixture
def adapter_default_config() -> InferenceConfig:
    return InferenceConfig(
        temperature=0.0,
        max_output_length=2048,
        max_model_len=4096,
        top_p=1.0,
        top_k=None,
        stop_tokens=["Question:"],
    )


@pytest.fixture
def test_prompts() -> list[EvalPrompt]:
    return [
        EvalPrompt(
            id="q1",
            messages=[PromptMessage(role=MessageRole.USER, content="Test question")],
        ),
    ]


@pytest.fixture
def test_records() -> list[InternalEvalRecord]:
    return [
        InternalEvalRecord(
            id="q1",
            question="Test question",
            answer="Test answer",
            metadata={"test": "value"},
        ),
    ]


@pytest.fixture
def mock_adapter(adapter_default_config: InferenceConfig, test_prompts: list[EvalPrompt]) -> Mock:
    adapter = Mock()
    adapter.get_inference_config.return_value = adapter_default_config
    adapter.get_loading_config.return_value = [DatasetLoadConfig(split="test", limit=None)]
    adapter.get_benchmark_split.return_value = "test"
    adapter.format_prompts.return_value = test_prompts
    return adapter


@pytest.fixture
def mock_engine(tmp_path: Path) -> Mock:
    def mock_run_inference(input_path, output_path, callbacks):
        output_data = pl.DataFrame({
            "response": ["Test response"],
            "chain_of_thought": [""],
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data.write_parquet(output_path)
        return output_path

    engine = Mock()
    engine.get_conversation_column_name.return_value = "messages"
    engine.run_inference.side_effect = mock_run_inference
    return engine


class TestInferenceConfigMerging:
    def test_uses_adapter_defaults_when_no_overrides(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        adapter_default_config: InferenceConfig,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine),
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )
            overrides = {}

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides=overrides,
            )

            mock_adapter.get_inference_config.assert_called_once()

    def test_overrides_apply_correctly(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        captured_config = None

        def capture_engine_init(config, inference_config):
            nonlocal captured_config
            captured_config = inference_config
            return mock_engine

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", side_effect=capture_engine_init),
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )
            overrides = {
                "temperature": 0.7,
                "max_output_length": 1024,
            }

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides=overrides,
            )

            assert captured_config is not None
            assert captured_config.temperature == 0.7
            assert captured_config.max_output_length == 1024
            assert captured_config.max_model_len == 4096
            assert captured_config.stop_tokens == ["Question:"]

    def test_none_values_dont_override_defaults(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        adapter_default_config: InferenceConfig,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        captured_config = None

        def capture_engine_init(config, inference_config):
            nonlocal captured_config
            captured_config = inference_config
            return mock_engine

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", side_effect=capture_engine_init),
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )
            overrides = {"temperature": None}

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides=overrides,
            )

            assert captured_config is not None
            assert captured_config.temperature == 0.0

    def test_all_params_can_be_overridden(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        captured_config = None

        def capture_engine_init(config, inference_config):
            nonlocal captured_config
            captured_config = inference_config
            return mock_engine

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", side_effect=capture_engine_init),
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )
            overrides = {
                "temperature": 0.9,
                "max_output_length": 512,
                "max_model_len": 2048,
                "top_p": 0.95,
                "top_k": 50,
                "stop_tokens": ["END"],
            }

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides=overrides,
            )

            assert captured_config is not None
            assert captured_config.temperature == 0.9
            assert captured_config.max_output_length == 512
            assert captured_config.max_model_len == 2048
            assert captured_config.top_p == 0.95
            assert captured_config.top_k == 50
            assert captured_config.stop_tokens == ["END"]

    def test_partial_overrides_merge_correctly(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        captured_config = None

        def capture_engine_init(config, inference_config):
            nonlocal captured_config
            captured_config = inference_config
            return mock_engine

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", side_effect=capture_engine_init),
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )
            overrides = {
                "temperature": 0.5,
                "stop_tokens": ["STOP"],
            }

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides=overrides,
            )

            assert captured_config is not None
            assert captured_config.temperature == 0.5
            assert captured_config.stop_tokens == ["STOP"]
            assert captured_config.max_output_length == 2048
            assert captured_config.max_model_len == 4096
            assert captured_config.top_p == 1.0


class TestEngineSelection:
    def test_selects_lalamo_engine(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine", return_value=mock_engine) as mock_lalamo,
            patch("lalamo.evals.inference.command.CustomAPIInferenceEngine") as mock_custom,
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = LalamoEngineConfig(
                model_path=Path("model"),
                batch_size=1,
                vram_gb=None,
            )

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides={},
            )

            mock_lalamo.assert_called_once()
            mock_custom.assert_not_called()

    def test_selects_custom_api_engine(
        self,
        tmp_path: Path,
        mock_adapter: Mock,
        mock_engine: Mock,
        test_records: list[InternalEvalRecord],
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "test.parquet").touch()

        with (
            patch("lalamo.evals.inference.command.REPO_TO_EVAL", {"test/repo": Mock(handler_type=lambda: mock_adapter)}),
            patch("lalamo.evals.inference.command.LalamoInferenceEngine") as mock_lalamo,
            patch("lalamo.evals.inference.command.CustomAPIInferenceEngine", return_value=mock_engine) as mock_custom,
            patch("lalamo.evals.inference.command._load_internal_dataset", return_value=test_records),
        ):
            engine_config = CustomAPIEngineConfig(
                base_url="http://localhost:8000",
                model="test-model",
                api_key=None,
                max_retries=0,
                timeout=60.0,
            )

            infer_command_handler(
                eval_repo="test/repo",
                dataset_dir=dataset_dir,
                output_dir=tmp_path / "output",
                engine_config=engine_config,
                inference_overrides={},
            )

            mock_custom.assert_called_once()
            mock_lalamo.assert_not_called()
