from evals import MMLUProHandler

from .common import EvalSpec

MMLU_PRO = EvalSpec(
    name="MMLU-Pro",
    repo="TIGER-Lab/MMLU-Pro",
    splits=["test", "validation"],
    handler_type=MMLUProHandler,
)

MMLU_EVALS = [MMLU_PRO]
