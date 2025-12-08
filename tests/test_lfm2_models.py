import pytest

from tests.lfm2_tracer import LFM2DecoderTracer
from tests.test_models import DType, ModelTestSpec, _test_model

MODEL_LIST = [
    ModelTestSpec("LiquidAI/LFM2-2.6B", DType.FLOAT32),
]


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_lfm2_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, LFM2DecoderTracer)

