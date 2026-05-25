import csv
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated, NamedTuple

import jax
import jax.numpy as jnp
import typer
from jaxtyping import Array, Float, Int

from lalamo.compressed.utils.rounding import (
    lut_values_at,
    pack_e4m3_scales,
)

CODEBOOK_CSV = "lloyd_max_codebooks.csv"
BIAS_LUT_CSV = "lloyd_max_bias_luts.csv"

DEFAULT_BITS = (2, 3, 4, 6, 8)
DEFAULT_GROUP_SIZES = (2, 4, 16, 32, 64, 128)
DEFAULT_BIAS_BITS = (2, 3, 4, 6, 8)


class _LutKey(NamedTuple):
    bits: int
    group_size: int
    bias_bits: int | None


class _IndexedValue(NamedTuple):
    index: int
    value: float


class _LutRow(NamedTuple):
    bits: int
    group_size: int
    bias_bits: int | None
    index: int
    value: float


class _BiasedLutState(NamedTuple):
    codebook: Float[Array, " levels"]
    center_indices: Int[Array, " bias_levels"]
    objective: Float[Array, ""]


def _parse_bias_bits(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _format_bias_bits(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


@cache
def _csv_values(filename: str) -> dict[_LutKey, tuple[float, ...]]:
    resource = files("lalamo.compressed.data").joinpath(filename)
    rows_by_key: dict[_LutKey, list[_IndexedValue]] = {}
    with resource.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = _LutKey(
                bits=int(row["bits"]),
                group_size=int(row["group_size"]),
                bias_bits=_parse_bias_bits(row["bias_bits"]),
            )
            rows_by_key.setdefault(key, []).append(_IndexedValue(index=int(row["index"]), value=float(row["value"])))

    return {
        key: tuple(
            indexed_value.value for indexed_value in sorted(rows, key=lambda indexed_value: indexed_value.index)
        )
        for key, rows in rows_by_key.items()
    }


def codebook_values(
    *,
    bits: int,
    group_size: int,
    bias_bits: int | None,
) -> tuple[float, ...]:
    key = _LutKey(bits=bits, group_size=group_size, bias_bits=bias_bits)
    values = _csv_values(CODEBOOK_CSV)
    if key not in values:
        raise ValueError(f"LloydMax codebook table is not available for {key=}")
    return values[key]


def bias_lut_values(
    *,
    bits: int,
    group_size: int,
    bias_bits: int,
) -> tuple[float, ...]:
    key = _LutKey(bits=bits, group_size=group_size, bias_bits=bias_bits)
    values = _csv_values(BIAS_LUT_CSV)
    if key not in values:
        raise ValueError(f"LloydMax bias table is not available for {key=}")
    return values[key]


def _sample_values_to_lloyd_lut_values(
    values: Float[Array, " samples"],
    weights: Float[Array, " samples"],
    *,
    num_levels: int,
    steps: int,
    initial_centers: Float[Array, " levels"] | None = None,
) -> Float[Array, " levels"]:
    if initial_centers is None:
        sorted_indices = jnp.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = jnp.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        center_weight_targets = (jnp.arange(num_levels) + 0.5) * total_weight / num_levels
        center_indices = jnp.searchsorted(cumulative_weights, center_weight_targets, side="left", method="compare_all")
        center_indices = jnp.minimum(center_indices, values.size - 1)
        centers = sorted_values[center_indices]
    else:
        centers = initial_centers

    for _ in range(steps):
        thresholds = (centers[:-1] + centers[1:]) / 2
        assignments = jnp.searchsorted(thresholds, values, side="left", method="compare_all")
        weight_sums = jnp.bincount(assignments, weights=weights, length=num_levels)
        value_sums = jnp.bincount(assignments, weights=weights * values, length=num_levels)
        safe_weight_sums = jnp.maximum(weight_sums, jnp.finfo(values.dtype).tiny)
        centers = jnp.where(weight_sums > 0, value_sums / safe_weight_sums, centers)
        centers = jnp.sort(centers)
    return centers


def _unbiased_codebook_values(
    bits: int,
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
) -> Float[Array, " levels"]:
    num_levels = 2**bits
    flat_normalized_weights = normalized_weights.reshape(-1)
    unweighted_centers = _sample_values_to_lloyd_lut_values(
        flat_normalized_weights,
        jnp.ones_like(flat_normalized_weights),
        num_levels=num_levels,
        steps=32,
    )
    half_levels = num_levels // 2
    negative_centers = unweighted_centers[:half_levels]
    positive_centers = unweighted_centers[half_levels:]
    initial_positive_values = (jnp.flip(jnp.abs(negative_centers)) + positive_centers) / 2
    initial_positive_values = jnp.sort(initial_positive_values)
    weights = jnp.broadcast_to(jnp.square(scale_values), normalized_weights.shape)
    positive_values = _sample_values_to_lloyd_lut_values(
        jnp.abs(flat_normalized_weights),
        weights.reshape(-1),
        num_levels=half_levels,
        steps=32,
        initial_centers=initial_positive_values,
    )
    return jnp.concatenate([-jnp.flip(positive_values), positive_values])


def _deterministic_lut_indices(
    values: Float[Array, "..."],
    table: Float[Array, " levels"],
) -> Int[Array, "..."]:
    thresholds = (table[:-1] + table[1:]).astype(values.dtype) / 2
    return jnp.searchsorted(thresholds, values, side="left", method="scan_unrolled").astype(jnp.int32)


def _weighted_bias_costs(
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
    codebook: Float[Array, " levels"],
    bias_candidates: Float[Array, " candidates"],
    chunk_size: int,
) -> Float[Array, " groups candidates"]:
    block_weights = jnp.square(scale_values[:, 0])
    weighted_mse_chunks = []
    for candidate_start in range(0, bias_candidates.size, chunk_size):
        candidate_stop = candidate_start + chunk_size
        candidate_biases = bias_candidates[candidate_start:candidate_stop]
        shifted_weights = normalized_weights[:, None, :] + candidate_biases[None, :, None]
        shifted_weight_indices = _deterministic_lut_indices(shifted_weights, codebook)
        code_values = lut_values_at(shifted_weight_indices, codebook)
        residuals = code_values - candidate_biases[None, :, None] - normalized_weights[:, None, :]
        block_mse = jnp.mean(residuals * residuals, axis=-1)
        weighted_mse_chunks.append(block_mse * block_weights[:, None])
    return jnp.concatenate(weighted_mse_chunks, axis=-1)


def _strictly_increasing_indices(
    indices: Int[Array, " levels"],
    *,
    min_value: int,
    max_value: int,
) -> Int[Array, " levels"]:
    host_indices = [int(index) for index in jax.device_get(indices)]
    adjusted_indices = []
    for offset, index in enumerate(host_indices):
        remaining_indices = len(host_indices) - offset - 1
        lower_bound = min_value + offset
        if adjusted_indices:
            lower_bound = max(lower_bound, adjusted_indices[-1] + 1)
        upper_bound = max_value - remaining_indices
        adjusted_indices.append(min(max(index, lower_bound), upper_bound))
    return jnp.array(adjusted_indices, dtype=indices.dtype)


def _symmetric_center_indices(
    positive_center_indices: Int[Array, " half_levels"],
    num_candidates: int,
) -> Int[Array, " levels"]:
    positive_center_indices = _strictly_increasing_indices(
        jnp.sort(positive_center_indices),
        min_value=num_candidates // 2,
        max_value=num_candidates - 1,
    )
    negative_center_indices = num_candidates - 1 - jnp.flip(positive_center_indices)
    return jnp.concatenate([negative_center_indices, positive_center_indices])


def _center_indices_from_values(
    centers: Float[Array, " levels"],
    bias_candidates: Float[Array, " candidates"],
) -> Int[Array, " levels"]:
    candidate_thresholds = (bias_candidates[:-1] + bias_candidates[1:]) / 2
    center_indices = jnp.searchsorted(
        candidate_thresholds,
        centers,
        side="right",
        method="compare_all",
    )
    return _strictly_increasing_indices(
        jnp.sort(center_indices),
        min_value=0,
        max_value=bias_candidates.size - 1,
    )


def _initial_bias_center_indices(
    best_biases: Float[Array, " groups"],
    scale_values: Float[Array, " groups"],
    bias_candidates: Float[Array, " candidates"],
    num_bias_levels: int,
) -> Int[Array, " bias_levels"]:
    positive_centers = _sample_values_to_lloyd_lut_values(
        jnp.abs(best_biases),
        scale_values,
        num_levels=num_bias_levels // 2,
        steps=16,
    )
    positive_center_indices = _center_indices_from_values(positive_centers, bias_candidates)
    return _symmetric_center_indices(positive_center_indices, bias_candidates.size)


def _signed_bias_center_indices(
    best_biases: Float[Array, " groups"],
    scale_values: Float[Array, " groups"],
    bias_candidates: Float[Array, " candidates"],
    num_bias_levels: int,
) -> Int[Array, " bias_levels"]:
    centers = _sample_values_to_lloyd_lut_values(
        best_biases,
        scale_values,
        num_levels=num_bias_levels,
        steps=16,
    )
    return _center_indices_from_values(centers, bias_candidates)


def _update_codebook(
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
    biases: Float[Array, " groups"],
    codebook: Float[Array, " levels"],
) -> Float[Array, " levels"]:
    shifted_weights = normalized_weights + biases[:, None]
    weight_indices = _deterministic_lut_indices(shifted_weights, codebook)
    (num_levels,) = codebook.shape
    half_levels = num_levels // 2
    pair_indices = jnp.where(
        weight_indices < half_levels,
        half_levels - 1 - weight_indices,
        weight_indices - half_levels,
    )
    signs = jnp.where(weight_indices < half_levels, -1, 1).astype(shifted_weights.dtype)
    weights = jnp.broadcast_to(jnp.square(scale_values), normalized_weights.shape)
    pair_weights = jnp.bincount(pair_indices.reshape(-1), weights=weights.reshape(-1), length=half_levels)
    pair_sums = jnp.bincount(
        pair_indices.reshape(-1),
        weights=(weights * signs * shifted_weights).reshape(-1),
        length=half_levels,
    )
    safe_pair_weights = jnp.maximum(pair_weights, jnp.finfo(codebook.dtype).tiny)
    positive_values = pair_sums / safe_pair_weights
    positive_values = jnp.where(pair_weights > 0, positive_values, codebook[half_levels:])
    positive_values = jnp.maximum(positive_values, jnp.finfo(codebook.dtype).tiny)
    positive_values = jnp.sort(positive_values)
    return jnp.concatenate([-jnp.flip(positive_values), positive_values])


def _update_bias_indices(
    weighted_bias_costs: Float[Array, " groups candidates"],
    center_indices: Int[Array, " bias_levels"],
) -> Int[Array, " bias_levels"]:
    (num_bias_levels,) = center_indices.shape
    center_costs = weighted_bias_costs[:, center_indices]
    assignments = jnp.argmin(center_costs, axis=-1)
    assignment_counts = jnp.bincount(assignments, length=num_bias_levels)
    assignment_weights = jax.nn.one_hot(assignments, num_bias_levels, dtype=weighted_bias_costs.dtype)
    candidate_costs = assignment_weights.T @ weighted_bias_costs
    updated_center_indices = jnp.argmin(candidate_costs, axis=-1)
    center_indices = jnp.where(assignment_counts > 0, updated_center_indices, center_indices)
    return _strictly_increasing_indices(
        jnp.sort(center_indices),
        min_value=0,
        max_value=weighted_bias_costs.shape[-1] - 1,
    )


def _refine_biased_lut_state(
    normalized_weights: Float[Array, "groups group_size"],
    scale_values: Float[Array, "groups 1"],
    codebook: Float[Array, " levels"],
    bias_candidates: Float[Array, " candidates"],
    weighted_bias_costs: Float[Array, " groups candidates"],
    center_indices: Int[Array, " bias_levels"],
    *,
    bias_candidate_chunk_size: int,
    steps: int,
) -> _BiasedLutState:
    for _ in range(steps):
        bias_table = bias_candidates[center_indices]
        center_costs = weighted_bias_costs[:, center_indices]
        assigned_bias_indices = jnp.argmin(center_costs, axis=-1)
        biases = lut_values_at(assigned_bias_indices.astype(jnp.uint8), bias_table)
        codebook = _update_codebook(normalized_weights, scale_values, biases, codebook)
        weighted_bias_costs = _weighted_bias_costs(
            normalized_weights,
            scale_values,
            codebook,
            bias_candidates,
            bias_candidate_chunk_size,
        )
        center_indices = _update_bias_indices(weighted_bias_costs, center_indices)
    objective = jnp.sum(jnp.min(weighted_bias_costs[:, center_indices], axis=-1))
    return _BiasedLutState(codebook=codebook, center_indices=center_indices, objective=objective)


@cache
def _normalized_samples(
    group_size: int,
) -> tuple[Float[Array, "groups group_size"], Float[Array, "groups 1"]]:
    sample_groups = 32_768
    key = jax.random.PRNGKey(0)
    samples = jax.random.normal(key, (sample_groups, group_size), dtype=jnp.float32)
    absmax = jnp.max(jnp.abs(samples), axis=-1, keepdims=True)
    scale_values = pack_e4m3_scales(absmax).astype(samples.dtype)
    safe_scale_values = jnp.where(scale_values == 0, 1, scale_values)
    normalized_weights = samples / safe_scale_values
    return normalized_weights, scale_values


@cache
def _unbiased_codebook(
    bits: int,
    group_size: int,
) -> Float[Array, " levels"]:
    normalized_weights, scale_values = _normalized_samples(group_size)
    return _unbiased_codebook_values(bits, normalized_weights, scale_values)


@cache
def _compute_lut_values(
    bits: int,
    group_size: int,
    bias_bits: int | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    normalized_weights, scale_values = _normalized_samples(group_size)
    codebook = _unbiased_codebook(bits, group_size)
    if bias_bits is None:
        codebook_values = tuple(float(value) for value in jax.device_get(codebook))
        return codebook_values, None

    num_bias_levels = 2**bias_bits
    bias_candidates = jnp.linspace(-1, 1, 512, dtype=jnp.float32)
    bias_candidate_chunk_size = 64
    weighted_bias_costs = _weighted_bias_costs(
        normalized_weights,
        scale_values,
        codebook,
        bias_candidates,
        bias_candidate_chunk_size,
    )
    best_bias_indices = jnp.argmin(weighted_bias_costs, axis=-1)
    best_biases = bias_candidates[best_bias_indices]
    scale_weights = jnp.square(scale_values[:, 0])
    initial_center_indices = (
        _initial_bias_center_indices(
            best_biases,
            scale_weights,
            bias_candidates,
            num_bias_levels,
        ),
        _signed_bias_center_indices(
            best_biases,
            scale_weights,
            bias_candidates,
            num_bias_levels,
        ),
    )
    states = tuple(
        _refine_biased_lut_state(
            normalized_weights,
            scale_values,
            codebook,
            bias_candidates,
            weighted_bias_costs,
            center_indices,
            bias_candidate_chunk_size=bias_candidate_chunk_size,
            steps=8,
        )
        for center_indices in initial_center_indices
    )
    best_state = min(states, key=lambda state: float(jax.device_get(state.objective)))

    codebook_values = tuple(float(value) for value in jax.device_get(best_state.codebook))
    bias_values = tuple(float(value) for value in jax.device_get(bias_candidates[best_state.center_indices]))
    return codebook_values, bias_values


def _write_values(
    path: Path,
    rows: list[_LutRow],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("bits", "group_size", "bias_bits", "index", "value"))
        for row in rows:
            writer.writerow(
                (
                    row.bits,
                    row.group_size,
                    _format_bias_bits(row.bias_bits),
                    row.index,
                    f"{row.value:.17g}",
                )
            )


def _generate_tables(
    *,
    bits_values: tuple[int, ...],
    group_sizes: tuple[int, ...],
    bias_bits_values: tuple[int, ...],
    output_dir: Path,
) -> None:
    codebook_rows: list[_LutRow] = []
    bias_rows: list[_LutRow] = []
    for group_size in group_sizes:
        for bits in bits_values:
            typer.echo(f"Generating LloydMax table: bits={bits}, group_size={group_size}, bias_bits=None")
            codebook, _bias_lut = _compute_lut_values(bits, group_size, None)
            codebook_rows.extend(
                _LutRow(bits=bits, group_size=group_size, bias_bits=None, index=index, value=value)
                for index, value in enumerate(codebook)
            )
            for bias_bits in bias_bits_values:
                typer.echo(f"Generating LloydMax table: bits={bits}, group_size={group_size}, bias_bits={bias_bits}")
                codebook, bias_lut = _compute_lut_values(bits, group_size, bias_bits)
                assert bias_lut is not None
                codebook_rows.extend(
                    _LutRow(bits=bits, group_size=group_size, bias_bits=bias_bits, index=index, value=value)
                    for index, value in enumerate(codebook)
                )
                bias_rows.extend(
                    _LutRow(bits=bits, group_size=group_size, bias_bits=bias_bits, index=index, value=value)
                    for index, value in enumerate(bias_lut)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_values(output_dir / CODEBOOK_CSV, codebook_rows)
    _write_values(output_dir / BIAS_LUT_CSV, bias_rows)


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
        bits: Annotated[
            list[int] | None,
            typer.Option(help="Weight precisions to generate. Can be passed multiple times."),
        ] = None,
        group_size: Annotated[
            list[int] | None,
            typer.Option(help="Group sizes to generate. Can be passed multiple times."),
        ] = None,
        bias_bits: Annotated[
            list[int] | None,
            typer.Option(help="Bias precisions to generate. Can be passed multiple times."),
        ] = None,
    ) -> None:
        bits_values = DEFAULT_BITS
        if bits is not None:
            bits_values = tuple(bits)

        group_sizes = DEFAULT_GROUP_SIZES
        if group_size is not None:
            group_sizes = tuple(group_size)

        bias_bits_values = DEFAULT_BIAS_BITS
        if bias_bits is not None:
            bias_bits_values = tuple(bias_bits)

        _generate_tables(
            bits_values=bits_values,
            group_sizes=group_sizes,
            bias_bits_values=bias_bits_values,
            output_dir=output_dir,
        )

    app()


if __name__ == "__main__":
    main()
