from evals import IFEvalAdapter

from lalamo.evals.datasets.specs.base import EvalSpec

IFEVAL = EvalSpec(
    name="IFEval",
    repo="google/IFEval",
    splits=["train"],
    handler_type=IFEvalAdapter,
)
