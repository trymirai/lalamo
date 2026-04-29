import pytest

from lalamo.model_import.model_configs import HFLFM2Config
from lalamo.model_import.model_specs.common import ModelType
from tests.conftest import filter_specs
from tests.model_test_tiers import ModelTier
from tests.tracer.tracer import DType, ModelTestSpec, _test_model
from tests.tracer.tracer_lfm2 import LFM2DecoderTracer

pytestmark = pytest.mark.usefixtures("tracer_mesh")

MODEL_LIST = [
    ModelTestSpec(model.origin.description, DType.FLOAT32)
    for model in LFM2_MODELS
    if model.origin.description.startswith("LiquidAI/")
]


@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_lfm2_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, LFM2DecoderTracer)
