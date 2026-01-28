from .common import EvalSpec
from .mmlu import MMLU_EVALS

ALL_EVALS = [*MMLU_EVALS]

REPO_TO_EVAL = {eval_spec.repo: eval_spec for eval_spec in ALL_EVALS}
