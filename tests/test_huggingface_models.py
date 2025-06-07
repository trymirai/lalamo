import jax.numpy as jnp
import pytest

from fartsovka import REPO_TO_MODEL, import_model
from tests.common import checkify_forward
from tests.huggingface_utils import load_hf_tracer

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-1b-it",
]


NUM_TOKENS = 512
TOKEN_STRIDE = 512


@pytest.mark.parametrize("model_repo", MODEL_LIST)
def test_hf_model(model_repo: str) -> None:
    hf_tracer = load_hf_tracer(model_repo)
    fs_model, *_ = import_model(REPO_TO_MODEL[model_repo])

    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)
    token_positions = jnp.arange(0, NUM_TOKENS, TOKEN_STRIDE, dtype=jnp.int32)
    mask = jnp.tril(jnp.ones((NUM_TOKENS, NUM_TOKENS), dtype=jnp.bool))

    err, fs_result = checkify_forward(fs_model)(
        token_ids=token_ids,
        token_positions=token_positions,
        mask=mask,
        return_updated_kv_cache=True,
        return_activation_trace=True,
    )
    err.throw()
    assert fs_result.activation_trace is not None

    hf_tracer.match_activation_trace(fs_result.activation_trace)
