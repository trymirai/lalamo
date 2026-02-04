from evals import MMLUProAdapter

from lalamo.evals.datasets.specs.base import EvalSpec

MMLU_PRO = EvalSpec(
    name="MMLU-Pro",
    repo="TIGER-Lab/MMLU-Pro",
    splits=["test", "validation"],
    handler_type=MMLUProAdapter,
)

MMLU_EVALS = [MMLU_PRO]
