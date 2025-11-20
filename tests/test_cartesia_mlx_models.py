import importlib.util
import unittest

import pytest

from tests.test_models import DType, ModelTestSpec, _test_model

CMX_AVAILABLE = importlib.util.find_spec("cartesia_mlx")

if CMX_AVAILABLE:
    from tests.cartesia_mlx_tracer import CartesiaMLXDecoderTracer

MODEL_LIST = [
    ModelTestSpec("cartesia-ai/Llamba-1B-4bit-mlx", DType.FLOAT32, token_stride=1),
]


@unittest.skipUnless(CMX_AVAILABLE, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_cartesia_mlx_model(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, CartesiaMLXDecoderTracer) # type: ignore

