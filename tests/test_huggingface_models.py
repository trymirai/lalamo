import pytest
import torch

from tests.huggingface_tracer import HFDecoderTracer
from tests.test_models import DType, ModelTestSpec, _test_model

MODEL_LIST = [
    ModelTestSpec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32),
    ModelTestSpec("google/gemma-3-1b-it", DType.FLOAT32),
    ModelTestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", DType.FLOAT32),
    ModelTestSpec("meta-llama/Llama-3.2-1B-Instruct", DType.FLOAT32),
    # ModelTestSpec("PleIAs/Pleias-RAG-1B", DType.FLOAT32),
    ModelTestSpec("Qwen/Qwen3-0.6B", DType.FLOAT32),
]

MODEL_LIST += [
    ModelTestSpec("Qwen/Qwen3-4B-AWQ", DType.FLOAT16),
    ModelTestSpec("openai/gpt-oss-20b", DType.FLOAT16),
] if torch.cuda.is_available() else []


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_hf_model(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, HFDecoderTracer)
