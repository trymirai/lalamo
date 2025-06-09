import jax.numpy as jnp
import pytest
import torch

from fartsovka import REPO_TO_MODEL, import_model
from tests.common import checkify_forward
from tests.huggingface_tracer import load_hf_tracer

MODEL_LIST = [
    "Qwen/Qwen2.5-?0.5B-Instruct",
    "google/gemma-3-1b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "PleIAs/Pleias-RAG-1B",
]


NUM_TOKENS = 512
TOKEN_STRIDE = 64


@pytest.mark.parametrize("model_repo", MODEL_LIST)
def test_hf_model(model_repo: str) -> None:
    hf_tracer = load_hf_tracer(model_repo, torch_dtype=torch.float32)
    fs_model, *_ = import_model(
        REPO_TO_MODEL[model_repo],
        context_length=NUM_TOKENS * TOKEN_STRIDE,
        precision=jnp.float32,
    )

    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)
    token_positions = jnp.arange(0, NUM_TOKENS * TOKEN_STRIDE, TOKEN_STRIDE, dtype=jnp.int32)
    mask = jnp.tril(jnp.ones((NUM_TOKENS, NUM_TOKENS), dtype=jnp.bool))

    err, fs_result = checkify_forward(fs_model)(
        token_ids=token_ids,
        token_positions=token_positions,
        mask=mask,
        return_updated_kv_cache=True,
        return_activation_trace=True,
    )
    err.throw()

    hf_tracer.match_activations(fs_result)
