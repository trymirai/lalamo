import unittest

import pytest

from tests.tracer.tracer import DType, ModelTestSpec, _test_model

try:
    from tests.tracer.tracer_mlx import MLXDecoderTracer
except ImportError:
    MLXDecoderTracer = None

pytestmark = pytest.mark.usefixtures("tracer_mesh")

# token_stride=1 is required because mlx doesn't accept token positions
MODEL_LIST = [
    ModelTestSpec(spec.repo, DType.FLOAT32, token_stride=1)
    for spec in filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CORE)
    if spec.quantization is not None and spec.config_type not in (HFLFM2Config, HFLlambaConfig)
]


@unittest.skipUnless(MLXDecoderTracer is not None, "requires mlx")
@pytest.mark.parametrize("test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST])
def test_mlx_model(test_spec: ModelTestSpec) -> None:
    assert MLXDecoderTracer is not None
    _test_model(test_spec, MLXDecoderTracer)
