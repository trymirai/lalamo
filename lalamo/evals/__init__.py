from lalamo.evals.cli import eval_app
from lalamo.evals.datasets import REPO_TO_EVAL, EvalSpec, convert_dataset
from lalamo.evals.inference import InferenceEngine, LalamoInferenceEngine, run_batch_generation, run_inference

__all__ = [
    "REPO_TO_EVAL",
    "EvalSpec",
    "InferenceEngine",
    "LalamoInferenceEngine",
    "convert_dataset",
    "eval_app",
    "run_batch_generation",
    "run_inference",
]
