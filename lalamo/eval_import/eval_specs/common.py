from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSpec:
    name: str
    repo: str
    splits: list[str]
