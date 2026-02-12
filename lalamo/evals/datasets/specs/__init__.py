from evals import IFEvalAdapter, MMLUProAdapter

EVAL_ADAPTERS = {
    "ifeval": IFEvalAdapter,
    "mmlu-pro": MMLUProAdapter,
}

__all__ = ["EVAL_ADAPTERS"]
