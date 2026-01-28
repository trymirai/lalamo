from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals import EvalHandler


@dataclass(frozen=True)
class EvalSpec:
    name: str
    repo: str
    splits: list[str]
    handler_type: type["EvalHandler"]
