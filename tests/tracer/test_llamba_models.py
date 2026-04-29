import unittest

import pytest

from tests.tracer.tracer import DType, ModelTestSpec, _test_model

try:
    from tests.tracer.tracer_llamba import LlambaDecoderTracer
except ImportError:
    LlambaDecoderTracer = None

pytestmark = pytest.mark.usefixtures("tracer_mesh")

MODEL_LIST = [
    ModelTestSpec("cartesia-ai/Llamba-1B-4bit-mlx", DType.FLOAT32, token_stride=1),
]


@unittest.skipUnless(LlambaDecoderTracer is not None, "requires cartesia_mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_llamba_models(test_spec: ModelTestSpec) -> None:
    assert LlambaDecoderTracer is not None
    _test_model(test_spec, LlambaDecoderTracer)
