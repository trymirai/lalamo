import unittest

import pytest

from tests.tracer.tracer import MLX_AVAILABLE, DType, ModelTestSpec, _test_model

if MLX_AVAILABLE:
    from tests.tracer.tracer_mlx import MLXDecoderTracer

# token_stride=1 is required because mlx doesn't accept token positions
MODEL_LIST = [
    ModelTestSpec("Qwen/Qwen3-0.6B-MLX-4bit", DType.FLOAT32, token_stride=1),
    ModelTestSpec("Qwen/Qwen3-4B-MLX-4bit", DType.FLOAT32, token_stride=1),
]


@unittest.skipUnless(MLX_AVAILABLE, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_mlx_model(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, MLXDecoderTracer)  # type: ignore
