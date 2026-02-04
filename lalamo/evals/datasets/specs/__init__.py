from lalamo.evals.datasets.specs.base import EvalSpec
from lalamo.evals.datasets.specs.ifeval import IFEVAL
from lalamo.evals.datasets.specs.mmlu import MMLU_PRO

REPO_TO_EVAL = {
    spec.repo: spec
    for spec in [MMLU_PRO, IFEVAL]
}

__all__ = ["REPO_TO_EVAL", "EvalSpec"]
