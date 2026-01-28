from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from lalamo.eval_import.internal_format import InternalEvalRecord
from lalamo.eval_import.prediction_format import BenchmarkMetrics, PredictionRecord


@dataclass(frozen=True)
class EvalHandler(ABC):
    @abstractmethod
    def convert_record(self, record: dict) -> InternalEvalRecord:
        """Convert single foreign format record to internal format. """
        ...

    @abstractmethod
    def convert_split(self, parquet_path: Path) -> list[InternalEvalRecord]:
        """Convert entire split file to internal format. """
        ...

    @abstractmethod
    def prepare_for_benchmark(
        self,
        predictions: list[PredictionRecord],
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
    ) -> Path:
        """Convert our internal format to what the official eval script expects. """
        ...

    @abstractmethod
    def run_official_benchmark(
        self,
        prepared_data_path: Path,
        eval_name: str,
        model_name: str,
        split: str,
    ) -> BenchmarkMetrics:
        """ Run the official evaluation code. """
        ...
