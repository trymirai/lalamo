from evals import MMLUProAdapter

from lalamo.evals.datasets.specs.base import EvalSpec

MMLU_PRO = EvalSpec(
    name="mmlu-pro",
    repo="TIGER-Lab/MMLU-Pro",
    handler_type=MMLUProAdapter,
)

MMLU_EVALS = [MMLU_PRO]
