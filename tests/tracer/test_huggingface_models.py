import pytest
import torch

from tests.helpers import unsi
from tests.tracer.tracer import DType, ModelTestSpec, _test_model
from tests.tracer.tracer_huggingface import HFDecoderTracer, ModernBertTracer

MODEL_LIST = [
    ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32),
    ModelTestSpec("google/gemma-3-1b-it", DType.FLOAT32),
    ModelTestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", DType.FLOAT32),
    ModelTestSpec("meta-llama/Llama-3.2-1B-Instruct", DType.FLOAT32),
    ModelTestSpec("google/gemma-4-E2B-it", DType.FLOAT32),
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

BIG_MODEL_TRACE_DTYPE = (
    DType.BFLOAT16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else DType.FLOAT32
)
BIG_MODEL_MEMORY_BYTES = unsi("160 G") if BIG_MODEL_TRACE_DTYPE == DType.BFLOAT16 else unsi("320 G")
BIG_MODEL_LIST = [
    ModelTestSpec(
        "google/gemma-4-31B",
        BIG_MODEL_TRACE_DTYPE,
        num_tokens=32,
        token_stride=16,
        minimum_memory_for_trace=BIG_MODEL_MEMORY_BYTES,
    ),
    ModelTestSpec(
        "google/gemma-4-26B-A4B",
        BIG_MODEL_TRACE_DTYPE,
        num_tokens=32,
        token_stride=16,
        minimum_memory_for_trace=BIG_MODEL_MEMORY_BYTES,
    ),
]


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_hf_lm_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, HFDecoderTracer)


@pytest.mark.parametrize("test_spec", BIG_MODEL_LIST, ids=[m.test_id for m in BIG_MODEL_LIST])
def test_big_hf_lm_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, HFDecoderTracer)


@pytest.mark.parametrize(
    "test_spec",
    CLASSIFIER_MODEL_LIST,
    ids=[m.model_repo for m in CLASSIFIER_MODEL_LIST],
)
def test_hf_classifier_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, ModernBertTracer)
