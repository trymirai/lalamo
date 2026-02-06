from lalamo.evals.cli import eval_app
from lalamo.evals.datasets import REPO_TO_EVAL, EvalSpec, convert_dataset_handler
from lalamo.evals.inference import InferenceEngine, LalamoInferenceEngine

__all__ = [
    "REPO_TO_EVAL",
    "EvalSpec",
    "InferenceEngine",
    "LalamoInferenceEngine",
    "convert_dataset_handler",
    "eval_app",
]
