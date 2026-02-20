import pytest

from lalamo.model_import.model_configs.huggingface import HuggingFaceLMConfig
from lalamo.model_import.model_specs.common import ModelSpec
from lalamo.model_registry import ModelRegistry

_hf_model_specs = [
    spec
    for spec in ModelRegistry.build(allow_third_party_plugins=False).models
    if issubclass(spec.config_type, HuggingFaceLMConfig)
]


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return ModelRegistry.build(allow_third_party_plugins=False)


@pytest.fixture(params=_hf_model_specs, ids=[spec.repo for spec in _hf_model_specs])
def hf_model_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param
