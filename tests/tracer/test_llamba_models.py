import unittest

import pytest

from lalamo.model_import.model_configs import HFLlambaConfig
from lalamo.model_import.model_specs.common import ModelType
from tests.conftest import filter_specs
from tests.model_test_tiers import ModelTier
from tests.tracer.tracer import DType, ModelTestSpec, _test_model

try:
    from tests.tracer.tracer_llamba import LlambaDecoderTracer
except ImportError:
    LlambaDecoderTracer = None

pytestmark = pytest.mark.usefixtures("tracer_mesh")

MODEL_LIST = [
    ModelTestSpec(spec.repo, DType.FLOAT32, token_stride=1)
    for spec in filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)
    if spec.config_type is HFLlambaConfig
]


@unittest.skipUnless(LlambaDecoderTracer is not None, "requires cartesia_mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_llamba_models(test_spec: ModelTestSpec) -> None:
    assert LlambaDecoderTracer is not None
    _test_model(test_spec, LlambaDecoderTracer)
