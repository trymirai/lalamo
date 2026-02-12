from lalamo.evals.datasets.specs.base import EvalSpec
from lalamo.evals.datasets.specs.ifeval import IFEVAL
from lalamo.evals.datasets.specs.mmlu import MMLU_EVALS, MMLU_PRO

REPO_TO_EVAL = {
    MMLU_PRO.name: MMLU_PRO,
    IFEVAL.name: IFEVAL,
}

__all__ = ["REPO_TO_EVAL", "EvalSpec"]
