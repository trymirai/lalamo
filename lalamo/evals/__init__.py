"""Evaluation tools for lalamo models."""

from .cli import eval_app
from .datasets import convert_dataset, EvalSpec, REPO_TO_EVAL
from .inference import run_inference, generate_replies, InferenceEngine, LalamoInferenceEngine

__all__ = [
    "eval_app",
    "convert_dataset",
    "EvalSpec",
    "REPO_TO_EVAL",
    "run_inference",
    "generate_replies",
    "InferenceEngine",
    "LalamoInferenceEngine",
]
