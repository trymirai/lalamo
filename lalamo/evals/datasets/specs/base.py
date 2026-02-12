from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.protocols import EvalAdapter


@dataclass(frozen=True)
class EvalSpec:
    name: str
    repo: str
    handler_type: type["EvalAdapter"]
