import math
from collections.abc import Sequence

UNITS = ["", "K", "M", "G", "T", "P", "E"]


def si(x: int, base: int = 1024, units: Sequence[str] = UNITS) -> str:
    precision = math.ceil(math.log10(base))
    power = min(math.trunc(math.log(math.fabs(x), base)), len(units) - 1) if x != 0 else 0
    return f"{x / (base**power):.{precision}f} {units[power]}"


def unsi(x: str, base: int = 1024, units: Sequence[str] = UNITS) -> int:
    val, unit, *_ = [*x.split(" ", 1), ""]
    return int(float(val) * (base ** units.index(unit)))


__all__ = ["UNITS", "si", "unsi"]
