from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from evals.types import InferenceConfig

from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks


class InferenceEngine(ABC):

    def _check_unsupported_params(
        self,
        inference_config: InferenceConfig,
        supported_params: set[str],
    ) -> list[str]:
        config_dict = asdict(inference_config)
        return [
            f"{k}={v}"
            for k, v in config_dict.items()
            if k not in supported_params and v is not None
        ]

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return the engine name for metadata."""
        ...

    @abstractmethod
    def get_conversation_column_name(self) -> str:
        """Return the column name this engine expects for conversation data."""
        ...

    @abstractmethod
    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        """Run inference and save outputs."""
        ...
