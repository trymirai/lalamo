from dataclasses import dataclass

from evals import EvalAdapter


@dataclass(frozen=True)
class EvalSpec:
    name: str
    repo: str
    splits: list[str]
    handler_type: type[EvalAdapter]
