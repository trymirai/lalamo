import pytest

from lalamo.registry import ModelRegistry


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return ModelRegistry()
