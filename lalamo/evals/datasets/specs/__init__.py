from .base import EvalSpec
from .mmlu import MMLU_PRO

# Build registry
REPO_TO_EVAL = {
    spec.repo: spec
    for spec in [MMLU_PRO]
}

__all__ = ["REPO_TO_EVAL", "EvalSpec"]
