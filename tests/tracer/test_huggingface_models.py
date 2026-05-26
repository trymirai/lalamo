import json
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lalamo.model_import.common import _import_chat_codec, import_model
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_registry import ModelRegistry
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.safetensors import safe_write
from lalamo.trace_comparator import compare_trace_files
from lalamo.tracer import (
    export_trace_result,
    record_checked_token_trace,
    record_tokenization_trace,
    trace_sharding_config,
)
from tests.helpers import unsi
from tests.tracer.tracer import DType, ModelTestSpec, _test_model, configure_precision_for_tests
from tests.tracer.tracer_huggingface import HFDecoderTracer, ModernBertTracer

MODEL_LIST = [
    ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32),
    ModelTestSpec("google/gemma-3-1b-it", DType.FLOAT32),
    ModelTestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", DType.FLOAT32),
    ModelTestSpec("meta-llama/Llama-3.2-1B-Instruct", DType.FLOAT32),
    ModelTestSpec("HuggingFaceTB/SmolLM3-3B", DType.FLOAT32, minimum_memory_for_trace=unsi("32 G")),
    ModelTestSpec("Qwen/Qwen3-0.6B", DType.FLOAT32),
    ModelTestSpec("Qwen/Qwen3.5-0.8B", DType.FLOAT32),
    ModelTestSpec("Qwen/Qwen3-Next-80B-A3B-Instruct", DType.FLOAT32, minimum_memory_for_trace=unsi("512 G")),
]

MODEL_LIST += (
    [
        ModelTestSpec("openai/gpt-oss-20b", DType.FLOAT16, minimum_memory_for_trace=unsi("64 G")),
    ]
    if torch.cuda.is_available()
    else []
)

CLASSIFIER_MODEL_LIST = [
    ModelTestSpec("trymirai/chat-moderation-router", DType.FLOAT32),
]


def test_hf_tracer_models_are_registered() -> None:
    repo_to_model = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model
    missing_models = [
        spec.model_repo for spec in [*MODEL_LIST, *CLASSIFIER_MODEL_LIST] if spec.model_repo not in repo_to_model
    ]
    assert not missing_models


def test_hf_chat_tokenization_matches_lalamo() -> None:
    model_repo = "Qwen/Qwen2.5-0.5B-Instruct"
    model_spec = cast(
        "LanguageModelSpec",
        ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[model_repo],
    )
    tokenizer, token_codec_config = _import_chat_codec(model_spec)
    trace_result = record_tokenization_trace(token_codec_config.init(tokenizer), [UserMessage("Tell me about London")])
    request = json.loads(trace_result.metadata["request"])
    hf_tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model_repo))

    hf_rendered = hf_tokenizer.apply_chat_template(
        request["messages"],
        add_generation_prompt=request["add_generation_prompt"],
        tokenize=False,
        enable_thinking=request["enable_thinking"],
    )
    hf_token_ids = hf_tokenizer.apply_chat_template(
        request["messages"],
        add_generation_prompt=request["add_generation_prompt"],
        tokenize=True,
        return_dict=False,
        enable_thinking=request["enable_thinking"],
    )

    assert trace_result.metadata["rendered_request"] == hf_rendered
    assert trace_result.arrays["token_ids"].tolist()[0] == hf_token_ids


def test_hf_lm_trace_files_compare(tmp_path: Path) -> None:
    test_spec = ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32, num_tokens=32, token_stride=1)
    assert test_spec.dtype is not None
    configure_precision_for_tests()
    sharding_config = trace_sharding_config()
    token_sharding = sharding_config.make_sharding((None, None))
    token_ids = jax.device_put(jnp.arange(test_spec.num_tokens, dtype=jnp.int32)[None, :], token_sharding)
    token_positions = jax.device_put(
        jnp.arange(test_spec.num_tokens, dtype=jnp.int32)[None, :],
        token_sharding,
    )

    hf_tracer = HFDecoderTracer.load(test_spec.model_repo, test_spec.dtype)
    imported_model = import_model(
        test_spec.model_repo,
        sharding_config=sharding_config,
        context_length=test_spec.num_tokens,
        dtype=test_spec.dtype.jax_dtype,
    )
    model = imported_model.model
    assert isinstance(model, LanguageModel)
    lalamo_result = record_checked_token_trace(model, token_ids, token_positions)

    reference_path = tmp_path / "hf.safetensors"
    result_path = tmp_path / "lalamo.safetensors"
    with reference_path.open("wb") as fd:
        safe_write(fd, hf_tracer.export_trace(token_ids, token_positions))
    with result_path.open("wb") as fd:
        safe_write(fd, export_trace_result(lalamo_result))

    comparison = compare_trace_files(reference_path, result_path)

    assert comparison.passed, "\n".join(f"{result.path}: {result.message}" for result in comparison.failed_results)


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_hf_lm_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, HFDecoderTracer)


@pytest.mark.parametrize(
    "test_spec",
    CLASSIFIER_MODEL_LIST,
    ids=[m.model_repo for m in CLASSIFIER_MODEL_LIST],
)
def test_hf_classifier_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, ModernBertTracer)
