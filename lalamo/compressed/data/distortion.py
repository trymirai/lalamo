import csv
from functools import cache
from importlib.resources import files
from typing import NamedTuple

DISTORTION_CSV = "distortion_estimates.csv"


class DistortionKey(NamedTuple):
    format_name: str
    bits: int
    group_size: int
    bias_bits: int | None = None
    scale_mode: str = ""
    scale_normalization: float | None = None
    residual_scale: float | None = None


def _parse_optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _key_from_csv_row(row: dict[str, str]) -> DistortionKey:
    return DistortionKey(
        format_name=row["format"],
        bits=int(row["bits"]),
        group_size=int(row["group_size"]),
        bias_bits=_parse_optional_int(row["bias_bits"]),
        scale_mode=row["scale_mode"],
        scale_normalization=_parse_optional_float(row["scale_normalization"]),
        residual_scale=_parse_optional_float(row["residual_scale"]),
    )


@cache
def _csv_distortions() -> dict[DistortionKey, float]:
    resource = files("lalamo.compressed.data").joinpath(DISTORTION_CSV)
    with resource.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {_key_from_csv_row(row): float(row["distortion"]) for row in reader}


def distortion_estimate(
    *,
    format_name: str,
    bits: int,
    group_size: int,
    bias_bits: int | None = None,
    scale_mode: str = "",
    scale_normalization: float | None = None,
    residual_scale: float | None = None,
) -> float:
    key = DistortionKey(
        format_name=format_name,
        bits=bits,
        group_size=group_size,
        bias_bits=bias_bits,
        scale_mode=scale_mode,
        scale_normalization=scale_normalization,
        residual_scale=residual_scale,
    )
    distortions = _csv_distortions()
    if key not in distortions:
        raise ValueError(f"Distortion estimate is not available for {key=}")
    return distortions[key]
