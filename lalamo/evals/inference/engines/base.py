from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.types import EvalPrompt, InferenceOutput


@dataclass(frozen=True)
class InferenceEngine(ABC):
    """Abstract interface for inference engines.

    Supports multiple engines: lalamo, uzu, llama.cpp, OpenAI API, etc.
    """

    @abstractmethod
    def prepare_input(
        self,
        prompts: list[EvalPrompt],
        output_path: Path,
    ) -> Path:
        """Convert prompts to engine-specific format and save.

        Examples:
        - lalamo: writes parquet with 'conversation' column
        - llama.cpp: writes JSONL with raw strings
        - OpenAI: returns in-memory format, no file needed

        Args:
            prompts: List of prompts to convert
            output_path: Where to save the engine-specific format

        Returns:
            Path to prepared input file
        """
        ...

    @abstractmethod
    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        **engine_params: Any,  # noqa: ANN401
    ) -> Path:
        """Run inference and save outputs.

        Args:
            input_path: Path to prepared input file
            output_path: Where to save inference outputs
            **engine_params: Engine-specific parameters

        Returns:
            Path to output file with responses
        """
        ...

    @abstractmethod
    def parse_output(
        self,
        output_path: Path,
        input_path: Path,
    ) -> list[InferenceOutput]:
        """Parse engine output to standardized format.

        Args:
            output_path: Path to engine output file
            input_path: Path to original input (for ID matching)

        Returns:
            List of standardized inference outputs
        """
        ...
