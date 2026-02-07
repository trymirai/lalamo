import pytest

from lalamo.model_registry import ModelRegistry, get_model_registry


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    return get_model_registry()
