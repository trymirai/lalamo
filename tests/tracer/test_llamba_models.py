import importlib.util
import unittest

import pytest

from lalamo.model_import.model_configs import HFLlambaConfig
from lalamo.model_import.model_specs.common import ModelType
from tests.conftest import filter_specs
from tests.model_test_tiers import ModelTier
from tests.tracer.tracer import DType, ModelTestSpec, _test_model

CMX_AVAILABLE = importlib.util.find_spec("cartesia_mlx")

if CMX_AVAILABLE:
    from tests.tracer.tracer_llamba import LlambaDecoderTracer

MODEL_LIST = [
    ModelTestSpec(spec.repo, DType.FLOAT32, token_stride=1)
    for spec in filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)
    if spec.config_type is HFLlambaConfig
]


@unittest.skipUnless(CMX_AVAILABLE, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_llamba_models(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, LlambaDecoderTracer)  # type: ignore
