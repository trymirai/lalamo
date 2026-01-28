from .common import EvalSpec

MMLU_PRO = EvalSpec(
    name="MMLU-Pro",
    repo="TIGER-Lab/MMLU-Pro",
    splits=["test", "validation"],
)

MMLU_EVALS = [MMLU_PRO]
