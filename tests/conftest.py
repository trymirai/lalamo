import pytest

from lalamo.registry import ModelRegistry, get_model_registry


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return get_model_registry()
