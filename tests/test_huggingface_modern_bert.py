# TODO: currently ModernBERT test is singled out because it will fail unless
# special temporary 'hacks' are uncommented in mlp.py and kv_cache.py. WIP to
# fix this.

import pytest

from tests.huggingface_tracer import HFClassifierTracer
from tests.test_models import DType, ModelTestSpec, _test_model

MODEL_LIST = [
    ModelTestSpec("trymirai/chat-moderation-router", DType.FLOAT32),
]


@pytest.mark.parametrize(
    "test_spec", MODEL_LIST, ids=[m.model_repo for m in MODEL_LIST]
)
def test_hf_model(test_spec: ModelTestSpec) -> None:
    _test_model(test_spec, HFClassifierTracer)


if __name__ == "__main__":
    test_hf_model(MODEL_LIST[0])
