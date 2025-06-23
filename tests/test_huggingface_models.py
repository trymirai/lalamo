from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import pytest
import torch

from lalamo import REPO_TO_MODEL, import_model
from tests.common import checkify_forward
from tests.huggingface_tracer import load_hf_tracer


class TestDType(Enum):
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
class TestSpec:
    model_repo: str
    dtype: TestDType


MODEL_LIST = [
    TestSpec("Qwen/Qwen2.5-0.5B-Instruct", TestDType.FLOAT32),
    TestSpec("google/gemma-3-1b-it", TestDType.FLOAT32),
    TestSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", TestDType.FLOAT32),
    TestSpec("meta-llama/Llama-3.2-1B-Instruct", TestDType.FLOAT32),
    TestSpec("PleIAs/Pleias-RAG-1B", TestDType.FLOAT32),
    TestSpec("Qwen/Qwen3-0.6B", TestDType.FLOAT32),
    TestSpec("Qwen/Qwen3-4B-AWQ", TestDType.FLOAT16),
]


NUM_TOKENS = 512
TOKEN_STRIDE = 64


@pytest.mark.parametrize("test_spec", MODEL_LIST)
def test_hf_model(test_spec: TestSpec) -> None:
    hf_tracer = load_hf_tracer(test_spec.model_repo, torch_dtype=test_spec.dtype.torch_dtype)
    llm_model, *_ = import_model(
        REPO_TO_MODEL[test_spec.model_repo],
        context_length=NUM_TOKENS * TOKEN_STRIDE,
        precision=test_spec.dtype.jax_dtype,
    )

    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)
    token_positions = jnp.arange(0, NUM_TOKENS * TOKEN_STRIDE, TOKEN_STRIDE, dtype=jnp.int32)
    mask = jnp.tril(jnp.ones((NUM_TOKENS, NUM_TOKENS), dtype=jnp.bool))

    err, llm_result = checkify_forward(llm_model)(
        token_ids=token_ids,
        token_positions=token_positions,
        mask=mask,
        return_updated_kv_cache=True,
        return_activation_trace=True,
    )
    err.throw()

    hf_tracer.match_activations(llm_result)
