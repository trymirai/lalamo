import unittest

import pytest

from lalamo.model_import.model_configs import HFLFM2Config, HFLlambaConfig
from lalamo.model_import.model_specs.common import ModelType
from tests.conftest import filter_specs
from tests.model_test_tiers import ModelTier
from tests.tracer.tracer import MLX_AVAILABLE, DType, ModelTestSpec, _test_model

if MLX_AVAILABLE:
    from tests.tracer.tracer_mlx import MLXDecoderTracer

# token_stride=1 is required because mlx doesn't accept token positions
MODEL_LIST = [
    ModelTestSpec(spec.repo, DType.FLOAT32, token_stride=1)
    for spec in filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)
    if spec.quantization is not None and spec.config_type not in (HFLFM2Config, HFLlambaConfig)
]


@unittest.skipUnless(MLX_AVAILABLE, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_mlx_model(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, MLXDecoderTracer)  # type: ignore
