from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRecord:
    """Universal format for model predictions."""

    id: str
    model_output: str
    extracted_answer: str | None = None
    is_correct: bool | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Universal format for benchmark results."""

    eval_name: str
    model_name: str
    split: str
    overall_accuracy: float
    total_examples: int
    correct: int
    incorrect: int
    category_metrics: dict[str, float] | None = None
    custom_metrics: dict[str, float] | None = None
