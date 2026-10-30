import pytest

from tests.tracer.tracer import DType
from tests.tracer.tracer_vision import VisionModelTestSpec, run_vision_tracer

VISION_MODEL_LIST = [
    VisionModelTestSpec("Qwen/Qwen3.5-0.8B", DType.FLOAT32),
    VisionModelTestSpec("Qwen/Qwen3.5-27B", DType.FLOAT32),
]


@pytest.mark.parametrize("test_spec", VISION_MODEL_LIST, ids=[spec.test_id for spec in VISION_MODEL_LIST])
def test_qwen3_5_vision_models(test_spec: VisionModelTestSpec) -> None:
    run_vision_tracer(test_spec)
