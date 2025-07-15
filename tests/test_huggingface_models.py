from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import pytest
import torch

from lalamo import REPO_TO_MODEL, import_model
from tests.common import checkify_forward
from tests.huggingface_tracer import load_hf_tracer


class DType(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.value)

    @property
    def jax_dtype(self) -> jnp.dtype:
        return jnp.dtype(self.value)


@dataclass
class Spec:
    model_repo: str
    dtype: DType
    requires_gpu: bool = False


MODEL_LIST = [
    Spec("Qwen/Qwen2.5-0.5B-Instruct", DType.FLOAT32),
    Spec("google/gemma-3-1b-it", DType.FLOAT32),
    Spec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", DType.FLOAT32),
    Spec("meta-llama/Llama-3.2-1B-Instruct", DType.FLOAT32),
    Spec("PleIAs/Pleias-RAG-1B", DType.FLOAT32),
    Spec("Qwen/Qwen3-0.6B", DType.FLOAT32),
    Spec("Qwen/Qwen3-4B-AWQ", DType.FLOAT16, requires_gpu=True),
]


NUM_TOKENS = 512
TOKEN_STRIDE = 64


@pytest.fixture
def configure_precision_for_tests() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


@pytest.mark.parametrize("test_spec", MODEL_LIST)
def test_hf_model(test_spec: Spec, configure_precision_for_tests: None) -> None:  # noqa: ARG001
    if test_spec.requires_gpu and not torch.cuda.is_available():
        pytest.skip("GPU is required for this test")

    llm_model, *_ = import_model(
        REPO_TO_MODEL[test_spec.model_repo],
        context_length=NUM_TOKENS * TOKEN_STRIDE,
        precision=test_spec.dtype.jax_dtype,
    )
    hf_tracer = load_hf_tracer(test_spec.model_repo, torch_dtype=test_spec.dtype.torch_dtype)

    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)
    token_positions = jnp.arange(0, NUM_TOKENS * TOKEN_STRIDE, TOKEN_STRIDE, dtype=jnp.int32)

    err, llm_result = checkify_forward(llm_model)(
        token_ids=token_ids,
        token_positions=token_positions,
        return_updated_kv_cache=True,
        return_activation_trace=True,
    )
    err.throw()

    hf_tracer.match_activations(llm_result)
