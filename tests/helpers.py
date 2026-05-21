import math
from collections import defaultdict
from collections.abc import Sequence
from functools import cache

import jax
from jax.sharding import AxisType, NamedSharding

from lalamo.utils.sharding import LogicalAxis, ShardingConfig

UNITS = ["", "K", "M", "G", "T", "P", "E"]


def si(x: int, base: int = 1024, units: Sequence[str] = UNITS) -> str:
    precision = math.ceil(math.log10(base))
    power = min(math.trunc(math.log(math.fabs(x), base)), len(units) - 1) if x != 0 else 0
    return f"{x / (base**power):.{precision}f} {units[power]}"


def unsi(x: str, base: int = 1024, units: Sequence[str] = UNITS) -> int:
    val, unit, *_ = [*x.split(" ", 1), ""]
    return int(float(val) * (base ** units.index(unit)))


@cache
def make_test_sharding_config() -> ShardingConfig:
    mesh = jax.make_mesh(
        (2, 2, 2),
        (LogicalAxis.BATCH.value, LogicalAxis.MATRIX.value, LogicalAxis.MIXTURE.value),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices("cpu")[:8],
    )
    return ShardingConfig(
        mesh=mesh,
        logical_to_physical=defaultdict(
            lambda: None,
            {
                LogicalAxis.BATCH: LogicalAxis.BATCH.value,
                LogicalAxis.MATRIX: LogicalAxis.MATRIX.value,
                LogicalAxis.MIXTURE: LogicalAxis.MIXTURE.value,
            },
        ),
    )


def make_sharding(logical_axes: tuple[LogicalAxis | None, ...]) -> NamedSharding:
    sharding_config = make_test_sharding_config()
    return sharding_config.resolve_sharding(logical_axes)


__all__ = ["UNITS", "make_sharding", "make_test_sharding_config", "si", "unsi"]
