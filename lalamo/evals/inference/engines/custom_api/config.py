from dataclasses import dataclass

from evals.types import InferenceEngineType


@dataclass
class CustomAPIEngineConfig:
    base_url: str
    model: str
    api_key: str | None = None
    max_retries: int = 0
    timeout: float = 60.0

    @property
    def engine_type(self) -> InferenceEngineType:
        return InferenceEngineType.CUSTOM_API

    def get_model_name(self) -> str:
        return self.model
