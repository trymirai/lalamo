import contextlib
import math
from collections.abc import Generator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from memray import FileReader, Tracker


class MemoryLimitExceededError(RuntimeError):
    pass


@contextlib.contextmanager
def limit_memory(maximum_memory: int) -> Generator[None, Any, None]:
    with TemporaryDirectory() as tempdir:
        tempfile = Path(tempdir) / "memray.bin"
        with Tracker(tempfile, native_traces=True):
            yield
        reader = FileReader(tempfile)
        actual_memory = sum(r.size for r in reader.get_high_watermark_allocation_records(merge_threads=True))
    if actual_memory > maximum_memory:
        raise MemoryLimitExceededError(
            f"Memory limit exceeded: {si(actual_memory)}B used but only {si(maximum_memory)}B allowed",
        )


UNITS = ["", "K", "M", "G", "T", "P", "E"]


def si(x: int, base: int = 1024, units: Sequence[str] = UNITS) -> str:
    precision = math.ceil(math.log10(base))
    power = min(math.trunc(math.log(math.fabs(x), base)), len(units) - 1) if x != 0 else 0
    return f"{x / (base**power):.{precision}f} {units[power]}"


def unsi(x: str, base: int = 1024, units: Sequence[str] = UNITS) -> int:
    val, unit, *_ = [*x.split(" ", 1), ""]
    return int(float(val) * (base ** units.index(unit)))


__all__ = ["UNITS", "limit_memory", "si", "unsi"]
