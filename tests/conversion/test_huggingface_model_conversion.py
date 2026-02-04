import json
import pathlib
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch

from lalamo.common import flatten_parameters
from lalamo.model_import import REPO_TO_MODEL, ModelMetadata, import_model
from lalamo.model_import.model_specs import ModelType
from lalamo.models import ClassifierModelConfig, LanguageModelConfig
from lalamo.modules import config_converter
from lalamo.safetensors import safe_write
from tests.helpers import limit_memory, unsi
from tests.tracer.tracer import DType, ModelTestSpec

MODEL_LIST: list[ModelTestSpec] = [
    ModelTestSpec("trymirai/chat-moderation-router", convert_memory_limit=unsi("400 M")),
    ModelTestSpec("Qwen/Qwen3-0.6B", convert_memory_limit=unsi("1.3 G")),
    ModelTestSpec("Qwen/Qwen3-8B", convert_memory_limit=unsi("16 G")),
    ModelTestSpec("Qwen/Qwen3-4B-AWQ"),
    ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct"),
    ModelTestSpec("google/gemma-3-1b-it", convert_memory_limit=unsi("2.5 G")),
    ModelTestSpec("google/gemma-3-4b-it"),
    ModelTestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ModelTestSpec("meta-llama/Llama-3.2-1B-Instruct"),
    ModelTestSpec("cartesia-ai/Llamba-1B", convert_memory_limit=unsi("2.8 G")),
    ModelTestSpec("cartesia-ai/Llamba-1B-4bit-mlx"),
    ModelTestSpec("LiquidAI/LFM2-350M", convert_memory_limit=unsi("800 M")),
    ModelTestSpec("mlx-community/LFM2-350M-4bit", convert_memory_limit=unsi("1.2 G")),
    ModelTestSpec("LiquidAI/LFM2-2.6B"),
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


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.test_id for m in MODEL_LIST])
def test_model_conversion(test_spec: ModelTestSpec, tmp_path: pathlib.Path) -> None:
    """
    Test the conversion -> export -> import flow for a given model.
    NOTE: Using tmp_path fixture provided by pytest runtime environment.
    """

    if test_spec.convert_memory_limit is not None:
        memory_limit = limit_memory(test_spec.convert_memory_limit)
    else:
        memory_limit = nullcontext()

    with memory_limit:
        model_repo = test_spec.model_repo
        model_dtype = test_spec.dtype
        model_spec = REPO_TO_MODEL.get(model_repo)
        assert model_spec, f"Unknown model specified: {model_repo}"

        # Step 1: Import model from HF format
        model, metadata = import_model(
            model_spec=model_spec,
            precision=model_dtype.jax_dtype if model_dtype is not None else None,
        )

        # Step 2: Export model
        model.message_processor.tokenizer.save(str(tmp_path / "tokenizer.json"))
        weights = flatten_parameters(model.export_weights())
        del model
        with (tmp_path / "model.safetensors").open("wb") as fd:
            safe_write(fd, weights)
        config_json = config_converter.unstructure(metadata, ModelMetadata)
        with open(tmp_path / "config.json", "w") as file:
            json.dump(config_json, file, indent=4)

    # Step 3: Re-import model from Lalamo format
    model = None
    match metadata.model_type:
        case ModelType.LANGUAGE_MODEL:
            model = LanguageModelConfig.load_model(tmp_path)
        case ModelType.CLASSIFIER_MODEL:
            model = ClassifierModelConfig.load_model(tmp_path)
    assert model is not None, f"Failed to load model {model_repo}"
    del model
