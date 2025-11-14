import json
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.flax import save_file

from lalamo.common import flatten_parameters
from lalamo.model_import import REPO_TO_MODEL, ModelMetadata, import_model
from lalamo.model_import.common import LanguageModel, ModelType, RouterModel
from lalamo.modules import config_converter
from tests.test_models import DType, ModelTestSpec

MODEL_LIST: list[ModelTestSpec] = [
    ModelTestSpec("trymirai/chat-moderation-router", DType.FLOAT32),
    ModelTestSpec("Qwen/Qwen3-0.6B", DType.FLOAT32),
    ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32),
    ModelTestSpec("google/gemma-3-1b-it", DType.FLOAT32),
    ModelTestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", DType.FLOAT32),
    ModelTestSpec("meta-llama/Llama-3.2-1B-Instruct", DType.FLOAT32),
    ModelTestSpec("cartesia-ai/Llamba-1B", DType.FLOAT32),
    ModelTestSpec("cartesia-ai/Llamba-1B-4bit-mlx", DType.FLOAT32),
]

MODEL_LIST += (
    [
        ModelTestSpec("Qwen/Qwen3-4B-AWQ", DType.FLOAT16),
        ModelTestSpec("openai/gpt-oss-20b", DType.FLOAT16),
    ]
    if torch.cuda.is_available()
    else []
)


@contextmanager
def temporary_directory() -> Generator[Path, Any, None]:
    """
    Context manager that creates a temporary directory and ensures cleanup.

    Yields:
        Path: Path to the temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp())

    try:
        yield temp_dir
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.mark.parametrize(
    "test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST]
)
def test_model_conversion(test_spec: ModelTestSpec) -> None:
    """
    Test the conversion -> export -> import flow for a given model.
    """

    model_repo_name = test_spec.model_repo
    model_dtype = test_spec.dtype
    assert model_dtype is not None

    with temporary_directory() as temp_dir:
        model_repo = REPO_TO_MODEL.get(model_repo_name)
        assert model_repo, f"Unknown model specified: {model_repo_name}"

        # Step 1: Import model from HF format
        model, metadata = import_model(
            model_spec=model_repo,
            precision=model_dtype.jax_dtype,
        )

        # Step 2: Export model
        model.message_processor.tokenizer.save(str(temp_dir / "tokenizer.json"))
        weights = flatten_parameters(model.export_weights())
        del model
        save_file(weights, temp_dir / "model.safetensors")
        config_json = config_converter.unstructure(metadata, ModelMetadata)
        with open(temp_dir / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)

        # Step 3: Re-import model from Lalamo format
        model = None
        match metadata.model_type:
            case ModelType.LANGUAGE_MODEL:
                model = LanguageModel.load(temp_dir)
            case ModelType.ROUTER_MODEL:
                model = RouterModel.load(temp_dir)
        assert model is not None, f"Failed to load model {model_repo_name}"
        del model
