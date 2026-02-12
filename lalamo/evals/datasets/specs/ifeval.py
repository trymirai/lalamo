from evals import IFEvalAdapter

from lalamo.evals.datasets.specs.base import EvalSpec

IFEVAL = EvalSpec(
    name="ifeval",
    repo="google/IFEval",
    handler_type=IFEvalAdapter,
)
