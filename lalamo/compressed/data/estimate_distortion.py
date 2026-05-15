import csv
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import typer

from lalamo.compressed.data.distortion import DISTORTION_CSV, DistortionKey
from lalamo.compressed.e8p import E8PSpec
from lalamo.compressed.lloyd_max import LloydMaxSpec
from lalamo.compressed.microfloat import MicrofloatScaleMode, MicrofloatSpec
from lalamo.compressed.quantized_spec import QuantizedSpec
from lalamo.weight_matrix import CompressionImplementation

DEFAULT_E8P_BITS = (2, 3, 4)
DEFAULT_LLOYD_MAX_BITS = (2, 3, 4, 6, 8)
DEFAULT_GROUP_SIZES = (2, 4, 16, 32, 64, 128)
DEFAULT_BIAS_BITS = (2, 3, 4, 6, 8)
DEFAULT_MICROFLOAT_SCALE_MODES = ("mxfp4", "nvfp4")
DEFAULT_SAMPLE_GROUPS = 8192
MAX_BIAS_SEARCH_ELEMENTS_PER_CHUNK = 8_388_608


class _DistortionRow(NamedTuple):
    key: DistortionKey
    distortion: float


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.17g}"


def _default_configs() -> tuple[DistortionKey, ...]:
    configs = [DistortionKey(format_name="e8p", bits=bits, group_size=8) for bits in DEFAULT_E8P_BITS]
    configs.extend(
        DistortionKey(
            format_name="microfloat",
            bits=4,
            group_size=group_size,
            scale_mode=scale_mode,
        )
        for group_size in DEFAULT_GROUP_SIZES
        for scale_mode in DEFAULT_MICROFLOAT_SCALE_MODES
    )
    configs.extend(
        DistortionKey(
            format_name="lloyd_max",
            bits=bits,
            group_size=group_size,
            bias_bits=bias_bits,
        )
        for group_size in DEFAULT_GROUP_SIZES
        for bits in DEFAULT_LLOYD_MAX_BITS
        for bias_bits in (None, *DEFAULT_BIAS_BITS)
    )
    return tuple(configs)


def _spec_from_key(key: DistortionKey) -> QuantizedSpec:
    if key.format_name == "e8p":
        return E8PSpec(
            bits=cast("Literal[2, 3, 4]", key.bits),
            scale_normalization=key.scale_normalization,
            residual_scale=key.residual_scale,
        )

    if key.format_name == "microfloat":
        return MicrofloatSpec(group_size=key.group_size, scale_mode=MicrofloatScaleMode(key.scale_mode))

    if key.format_name == "lloyd_max":
        return LloydMaxSpec(
            bits=cast("Literal[2, 3, 4, 6, 8]", key.bits),
            group_size=key.group_size,
            bias_bits=cast("Literal[2, 3, 4, 6, 8] | None", key.bias_bits),
        )

    raise ValueError(f"Unsupported quantized format {key.format_name!r}")


def _estimate_distortion(key: DistortionKey, sample_groups: int) -> float:
    spec = _spec_from_key(key)
    chunk_size = sample_groups
    if key.format_name == "lloyd_max" and key.bias_bits is not None:
        bias_levels = 2**key.bias_bits
        chunk_size = MAX_BIAS_SEARCH_ELEMENTS_PER_CHUNK // (key.group_size * bias_levels)
        chunk_size = max(1, min(sample_groups, chunk_size))

    squared_error_sum = 0.0
    value_count = 0
    random_key = jax.random.PRNGKey(0)
    for chunk_start in range(0, sample_groups, chunk_size):
        current_chunk_size = min(chunk_size, sample_groups - chunk_start)
        chunk_key = jax.random.fold_in(random_key, chunk_start)
        weights = jax.random.normal(chunk_key, (current_chunk_size, key.group_size), dtype=jnp.float32)
        compressed = spec.compress(weights, implementation=CompressionImplementation.TRAINING, is_sharded=False)
        squared_errors = jnp.square(weights - compressed.decompress())
        squared_error_sum += float(jax.device_get(jnp.sum(squared_errors)))
        value_count += weights.size

    return squared_error_sum / value_count


def _write_distortions(path: Path, rows: list[_DistortionRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            (
                "format",
                "bits",
                "group_size",
                "bias_bits",
                "scale_mode",
                "scale_normalization",
                "residual_scale",
                "distortion",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.key.format_name,
                    row.key.bits,
                    row.key.group_size,
                    _format_optional_int(row.key.bias_bits),
                    row.key.scale_mode,
                    _format_optional_float(row.key.scale_normalization),
                    _format_optional_float(row.key.residual_scale),
                    f"{row.distortion:.17g}",
                )
            )


def _generate_distortions(*, output_dir: Path, sample_groups: int) -> None:
    rows: list[_DistortionRow] = []
    for key in _default_configs():
        typer.echo(f"Estimating distortion: {key}")
        rows.append(_DistortionRow(key=key, distortion=_estimate_distortion(key, sample_groups)))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_distortions(output_dir / DISTORTION_CSV, rows)


def main() -> None:
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.callback()
    def _main() -> None:
        pass

    @app.command()
    def generate(
        output_dir: Annotated[
            Path,
            typer.Option(file_okay=False, dir_okay=True, help="Directory for generated CSV tables."),
        ] = Path(__file__).resolve().parent,
        sample_groups: Annotated[
            int,
            typer.Option(help="Number of Gaussian groups sampled for each distortion estimate."),
        ] = DEFAULT_SAMPLE_GROUPS,
    ) -> None:
        _generate_distortions(output_dir=output_dir, sample_groups=sample_groups)

    app()


if __name__ == "__main__":
    main()
