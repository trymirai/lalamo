import pytest

from tests.tracer.tracer_vision import VisionModelTestSpec, run_vision_tracer

VISION_MODEL_LIST = [
    VisionModelTestSpec("Qwen/Qwen3.5-0.8B"),
    VisionModelTestSpec("Qwen/Qwen3.5-9B"),
    VisionModelTestSpec("Qwen/Qwen3.8-27B"),
]


@pytest.mark.parametrize("test_spec", VISION_MODEL_LIST, ids=[spec.model_repo for spec in VISION_MODEL_LIST])
def test_qwen3_5_vision_models(test_spec: VisionModelTestSpec) -> None:
    run_vision_tracer(test_spec)
