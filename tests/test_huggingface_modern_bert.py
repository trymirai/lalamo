import gc
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import pytest
import torch

from lalamo import import_language_model
from lalamo.model_import.common import import_router_model
from lalamo.router_model import RouterModel
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


@dataclass(frozen=True)
class Spec:
    model_repo: str
    dtype: DType
    requires_gpu: bool = False


MODEL_LIST = [
    Spec("trymirai/flo-bert-classifier", DType.FLOAT32),
]


NUM_TOKENS = 2
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

    router_model, *_ = import_router_model(
        test_spec.model_repo,
        context_length=NUM_TOKENS * TOKEN_STRIDE,
        precision=test_spec.dtype.jax_dtype,
    )
    assert isinstance(router_model, RouterModel)
    hf_tracer = load_hf_tracer(test_spec.model_repo, dtype=test_spec.dtype.torch_dtype)

    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)[None, :]
    token_positions = jnp.arange(0, NUM_TOKENS * TOKEN_STRIDE, TOKEN_STRIDE, dtype=jnp.int32)[None, :]

    with jax.disable_jit():
        err, llm_result = checkify_forward(router_model.classifier)(
            token_ids=token_ids,
            token_positions=token_positions,
            return_updated_kv_cache=True,
            return_activation_trace=True,
        )
        err.throw()

    del router_model
    gc.collect()

    hf_tracer.match_activations(llm_result)
