from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from evals.types import EvalPrompt, InferenceOutput, InternalEvalRecord

from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks


@dataclass(frozen=True)
class InferenceEngine(ABC):
    @abstractmethod
    def prepare_input(
        self,
        prompts: list[EvalPrompt],
        records: list[InternalEvalRecord],
        output_path: Path,
    ) -> Path:
        """Convert prompts to engine-specific format and save. """
        ...

    @abstractmethod
    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        """Run inference and save outputs. """
        ...

    @abstractmethod
    def parse_output(
        self,
        output_path: Path,
        input_path: Path,
    ) -> list[InferenceOutput]:
        """Parse engine output to standardized format. """
        ...
