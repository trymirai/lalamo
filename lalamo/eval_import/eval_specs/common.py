from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lalamo.eval_import.eval_handlers import EvalHandler


@dataclass(frozen=True)
class EvalSpec:
    name: str
    repo: str
    splits: list[str]
    handler_type: type["EvalHandler"]
