from dataclasses import dataclass


@dataclass
class CustomAPIEngineConfig:
    base_url: str
    model: str
    api_key: str | None = None
    max_retries: int = 0
    timeout: float = 60.0

    def get_model_name(self) -> str:
        return self.model
