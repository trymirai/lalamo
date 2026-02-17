import pytest

from lalamo.model_registry import ModelRegistry


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return ModelRegistry.build(allow_third_party_plugins=False)
