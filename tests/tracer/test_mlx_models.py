import unittest

import pytest

from tests.tracer.tracer import DType, ModelTestSpec, _test_model

try:
    from tests.tracer.tracer_mlx import MLXDecoderTracer
except ImportError:
    MLXDecoderTracer = None

MODEL_LIST = [
    ModelTestSpec("Qwen/Qwen3-0.6B-MLX-4bit", DType.FLOAT32, token_stride=1),
    ModelTestSpec("Qwen/Qwen3-4B-MLX-4bit", DType.FLOAT32, token_stride=1),
]


@unittest.skipUnless(MLXDecoderTracer is not None, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_mlx_model(test_spec: ModelTestSpec) -> None:
    assert MLXDecoderTracer is not None
    _test_model(test_spec, MLXDecoderTracer)
