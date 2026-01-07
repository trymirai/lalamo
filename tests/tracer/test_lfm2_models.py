import pytest

from lalamo.model_import.model_specs.lfm2 import LFM2_MODELS
from tests.tracer.tracer import DType, ModelTestSpec, _test_model
from tests.tracer.tracer_lfm2 import LFM2DecoderTracer

MODEL_LIST = [ModelTestSpec(model.repo, DType.FLOAT32) for model in LFM2_MODELS if model.quantization is None]


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_lfm2_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, LFM2DecoderTracer)

