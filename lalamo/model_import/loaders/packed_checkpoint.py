import json
import math
from collections.abc import Iterable
from dataclasses import replace
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

import cattrs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, Float32

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec, IncoherenceSigns, KroneckerRotation
from lalamo.compressed.int import IntSpec
from lalamo.compressed.lattice import (
    COLUMNS_PER_LADDER_INDEX,
    LADDER_INDEX_BITS,
    LatticeKind,
    LatticeMatrix,
    LatticeSpec,
    odd_integer_table,
)
from lalamo.compressed.mlx import MLXMatrix, MLXSpec
from lalamo.compressed.qtip_gaussian import (
    COLUMN_CLASSES,
    STATE_BITS,
    QtipGaussianMatrix,
    QtipGaussianSpec,
    states_to_levels,
)
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.utils.packing import unpack_uint8_to_uint
from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import Layout, ShapeDtypeMatrix, WeightMatrix


class GainAxis(StrEnum):
    ROW = "row"
    COLUMN = "column"


def codebook_from_table(table: Float[Array, "states width"]) -> Float[Array, " codebook"]:
    """The [scale, offsets by column class] codebook whose scale * level + offset reproduces the table, else raises."""
    width = table.shape[1]
    values = np.asarray(table, dtype=np.float64)
    levels = np.asarray(states_to_levels(jnp.arange(1 << STATE_BITS, dtype=jnp.uint32)), dtype=np.float64)[:, :width]
    farthest = np.argmax(np.abs(levels[:, 0] - levels[0, 0]))
    scale = (values[farthest, 0] - values[0, 0]) / (levels[farthest, 0] - levels[0, 0])
    offsets = values[0] - scale * levels[0]
    error = np.abs(scale * levels + offsets - values).max()
    if not error <= 1e-5:
        raise ValueError(f"The trellis table is not scale * level + offset (error {error})")
    return jnp.asarray([scale, *offsets[np.arange(COLUMN_CLASSES) % width]], dtype=jnp.float32)


def merged(top: WeightMatrix, bottom: WeightMatrix) -> WeightMatrix | None:
    """One leaf holding the rows of both when they share a format, else None."""
    if (top.spec, top.dtype) != (bottom.spec, bottom.dtype):
        return None
    if isinstance(top, MLXMatrix):
        return jax.tree.map(lambda *planes: jnp.concatenate(planes), top, bottom)
    if (
        isinstance(top, QtipGaussianMatrix)
        and isinstance(bottom, QtipGaussianMatrix)
        and jnp.array_equal(top.codebook, bottom.codebook)
    ):
        return replace(
            top, codes=jnp.concatenate((top.codes, bottom.codes)), scales=jnp.concatenate((top.scales, bottom.scales))
        )
    return None


def native_config(value: JSON) -> JSON:
    if isinstance(value, list):
        return [native_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "pard_token" in value:
        value = dict(value)
        pard_token = value.pop("pard_token")
        assert pard_token is None, "PARD checkpoints are not supported"
    if value.get("type") == "AttentionConfig" and "qkv_projection_config" in value:
        assert not {"qkvg_projection_config", "has_qkvg_biases", "has_gate"} & value.keys()
        value = dict(value)
        value["qkvg_projection_config"] = value.pop("qkv_projection_config")
        gate_projection_config = value.pop("gate_projection_config")
        assert gate_projection_config == value["qkvg_projection_config"]
        value["has_qkvg_biases"] = value.pop("has_qkv_biases")
        assert not value["has_qkvg_biases"]
        value["has_gate"] = True
    return {name: native_config(item) for name, item in value.items()}


def is_packed_checkpoint(metadata: dict[str, JSON], tensor_names: Iterable[str]) -> bool:
    # Lalamo's own saves also tag trellis leaves "QtipGaussianSpec", but keep a five-float codebook under each matrix.
    return any(name.startswith("qtip_shared.") for name in tensor_names) or any(
        isinstance(spec, dict) and spec["type"] in ("D4S4Spec", "I3S4Spec", "I4S4Spec") for spec in metadata.values()
    )


def load_packed_checkpoint(
    directory: Path | str, sharding_config: ShardingConfig, dtype: DTypeLike | None = None
) -> LanguageModel:
    """Import packed checkpoints without refitting their saved dense or packed weights."""
    directory = Path(directory)
    config = BaseModelConfig.from_json(native_config(json.loads((directory / "config.json").read_text())))
    assert isinstance(config, LanguageModelConfig)
    converter = cattrs.Converter(forbid_extra_keys=True)

    with (directory / "model.safetensors").open("rb") as stream:
        metadata, arrays = safe_read(stream)
        assert metadata is not None
        specs = {name: json.loads(spec) for name, spec in metadata.items()}
        template = config.init_from_directory(directory, EmptyInitializer(dtype, sharding_config))
        # RoPE is recomputed from the config, as uzu does; older packages' saved tables match it to one float32 ulp.
        consumed = {name for name in arrays if name.startswith("decoder.transformer.ropes.")}
        consumed_specs: set[str] = set()

        def parameter(name: str) -> Array:
            consumed.add(name)
            return arrays[name]

        def parameter_tuple(path: ParameterPath, count: int) -> tuple[Array, ...]:
            return tuple(parameter(path / index) for index in range(count))

        def saved_spec(path: ParameterPath) -> dict[str, Any]:
            consumed_specs.add(path / "spec")
            return specs[path / "spec"]

        @cache
        def codebook(table_name: str) -> Float[Array, " codebook"]:
            return codebook_from_table(parameter(table_name))

        def folded_row_scale(
            path: ParameterPath, saved: dict[str, Any], scale: Array, *gains: Array
        ) -> Float32[Array, " rows"]:
            # Row post-gains multiply into the row scale in float32; column post-gains only fold when they are 1.
            axes = tuple(map(GainAxis, saved.pop("post_gain_axes", ())))
            post_gains = tuple(zip(axes, parameter_tuple(path / "post_gains", len(axes)), strict=True))
            if not all(bool(jnp.all(gain == 1)) for axis, gain in post_gains if axis == GainAxis.COLUMN):
                raise ValueError(f"Column post-gains other than 1 do not fold into the row scales at {path}")
            row_gains = (*gains, *(gain for axis, gain in post_gains if axis == GainAxis.ROW))
            return math.prod((gain.astype(jnp.float32) for gain in row_gains), start=scale.astype(jnp.float32))

        def row_stack(parts: tuple[WeightMatrix, ...], is_sharded: bool) -> WeightMatrix:
            if len(parts) == 1:
                return parts[0]
            spec = RowStackSpec(tuple((part.shape[0], part.spec) for part in parts))
            return RowStackMatrix(spec=spec, sharding_config=sharding_config, is_sharded=is_sharded, parts=parts)

        def stacked(hybrids: Iterable[WeightMatrix], is_sharded: bool) -> WeightMatrix:
            # Neighbours under one saved rotation become one hybrid, in which neighbouring same-format leaves merge.
            groups: list[tuple[IncoherenceSigns | KroneckerRotation, list[WeightMatrix]]] = []
            for hybrid in hybrids:
                assert isinstance(hybrid, HybridMatrix) and hybrid.incoherence_signs is not None
                if not groups or not eqx.tree_equal(groups[-1][0], hybrid.incoherence_signs):
                    groups.append((hybrid.incoherence_signs, []))
                leaves = groups[-1][1]
                if leaves and (joined := merged(leaves[-1], hybrid.quantized)) is not None:
                    leaves[-1] = joined
                else:
                    leaves.append(hybrid.quantized)
            fused = tuple(
                HybridMatrix.of(row_stack(tuple(leaves), is_sharded), rotation, sharding_config, is_sharded)
                for rotation, leaves in groups
            )
            return row_stack(fused, is_sharded)

        def weight(path: ParameterPath, saved: dict[str, Any], template: ShapeDtypeMatrix) -> WeightMatrix:
            columns = template.shape[1]
            is_sharded = template.is_sharded
            saved = dict(saved)
            matrix: WeightMatrix
            match saved.pop("type"):
                case "FullPrecisionSpec":
                    matrix = template.load_exported(ExportResults(arrays, specs), prefix=path)
                    consumed.add(path / "weights")
                case "QtipGaussianSpec":
                    table_name = saved.pop("table", f"qtip_shared.codebook_v{saved['vector_width']}")
                    assert saved.pop("layout") == Layout.OUTPUT_INPUT, f"QTIP leaves are stored output-input at {path}"
                    scales, gains = parameter(path / "scales"), parameter(path / "gains")
                    assert scales.dtype == saved.pop("scale_dtype", "float16"), f"Scale dtype differs at {path}"
                    pre_gains = parameter_tuple(path / "pre_gains", saved.pop("pre_gain_count", 0))
                    # Every saved gain multiplies a whole row, so one float32 scale per row replaces them all.
                    row_scale = folded_row_scale(path, saved, scales, gains, *pre_gains)
                    leaf = QtipGaussianMatrix(
                        spec=converter.structure(saved, QtipGaussianSpec),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        # The saved gains carry the matrix dtype.
                        dtype_=gains.dtype,
                        columns=columns,
                        codes=parameter(path / "codes"),
                        scales=row_scale,
                        codebook=codebook(table_name),
                    )
                    rotation = KroneckerRotation(
                        signs=parameter(f"qtip_shared.signs_{columns}"), small_q=parameter(f"qtip_shared.q_{columns}")
                    )
                    matrix = HybridMatrix.of(leaf, rotation, sharding_config, is_sharded)
                case "RowStackSpec":
                    assert saved.pop("layout") == Layout.OUTPUT_INPUT, f"Row stacks are stored output-input at {path}"
                    # Each part's saved spec is inline in the stack's; the parts have no spec entries of their own.
                    rows, part_specs = zip(*saved.pop("parts"), strict=True)
                    assert not saved, f"Unexpected row stack {saved} at {path}"
                    parts = [weight(path / "parts" / index, spec, template) for index, spec in enumerate(part_specs)]
                    assert tuple(part.shape[0] for part in parts) == rows
                    matrix = stacked(parts, is_sharded)
                case "HybridSpec":
                    spec = HybridSpec.from_json({"type": "HybridSpec", **saved})
                    assert isinstance(spec.quantization_spec, IntSpec)
                    # Saved as HybridMatrix.export writes it: int scales group-major, signs on the input axis only.
                    quantized = spec.quantization_spec.from_packed_parameters(
                        packed_weights=parameter(path / "quantized" / "weights"),
                        scales=parameter(path / "quantized" / "scales")[:, : template.shape[0]].T,
                        packed_zero_points=None,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                    )
                    signs = IncoherenceSigns(parameter(path / "incoherence_signs" / "input_signs"), None)
                    matrix = HybridMatrix.of(quantized, signs, sharding_config, is_sharded)
                    assert matrix.spec == spec
                case "I4S4Spec":
                    assert saved.pop("layout") == Layout.OUTPUT_INPUT, f"I4S4 leaves are stored output-input at {path}"
                    # Level 2c - 15 times the group scale s = row scale * ladder value is (2s) * c - 15s: affine, once
                    # the nibbles are swapped to low-first. Gains fold into s in float32, exact for powers of two.
                    row_scales = parameter(path / "row_scales")
                    groups = unpack_uint8_to_uint(parameter(path / "ladder_indices"), LADDER_INDEX_BITS)
                    ladder = jnp.take(parameter(path / "ladder"), groups).astype(jnp.float32)
                    scales = folded_row_scale(path, saved, row_scales)[:, None] * ladder
                    codes = parameter(path / "codes")
                    quantized = MLXSpec(4, COLUMNS_PER_LADDER_INDEX, Layout.OUTPUT_INPUT).from_packed_parameters(
                        packed_weights=(codes << 4) | (codes >> 4),
                        scales=(2 * scales).astype(dtype or row_scales.dtype),
                        biases=(-15 * scales).astype(dtype or row_scales.dtype),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                    )
                    signs = IncoherenceSigns(parameter(path / "input_hadamard_factors"), None)
                    matrix = HybridMatrix.of(quantized, signs, sharding_config, is_sharded)
                case "D4S4Spec" | "I3S4Spec" as kind_name:
                    if saved.pop("post_gain_axes", ()):
                        raise ValueError(f"Lattice post-gains do not commute with the Hadamard rotation at {path}")
                    lattice = converter.structure({**saved, "kind": kind_name[:2].lower()}, LatticeSpec)
                    leaf = LatticeMatrix(
                        spec=lattice,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=parameter(path / "table") if lattice.kind == LatticeKind.D4 else odd_integer_table(3),
                    )
                    if lattice.layout == Layout.INPUT_OUTPUT:
                        rotation = IncoherenceSigns(None, parameter(path / "output_hadamard_factors"))
                    else:
                        rotation = IncoherenceSigns(parameter(path / "input_hadamard_factors"), None)
                    matrix = HybridMatrix.of(leaf, rotation, sharding_config, is_sharded)
                case other:
                    raise ValueError(f"Unsupported packed checkpoint weight format {other!r} at {path}")
            return matrix if dtype is None else matrix.astype(dtype)

        def restore(jax_path: tuple[object, ...], leaf: object) -> object:
            path = ParameterPath() / jax_path
            if isinstance(leaf, ShapeDtypeMatrix):
                # Older packages save the attention projection as a qkv leaf and a gate leaf.
                parent = path.removesuffix("qkvg_projection.weights")
                if parent != path and parent + "qkv_projection.weights.spec" in specs:
                    qkv, gate = (ParameterPath(f"{parent}{name}_projection.weights") for name in ("qkv", "gate"))
                    parts = [weight(part, saved_spec(part), leaf) for part in (qkv, gate)]
                    return load_as(leaf, stacked(parts, leaf.is_sharded))
                return load_as(leaf, weight(path, saved_spec(path), leaf))
            if isinstance(leaf, ShapeDtypeStruct | Array):
                value = parameter(path)
                assert value.shape == leaf.shape, f"Saved shape differs from model at {path}"
                # A declared dtype (norm scales: float32) is not weak; an unset one follows the saved array.
                if dtype is not None or not getattr(leaf, "weak_type", True):
                    value = value.astype(leaf.dtype)
                return jax.device_put(value, leaf.sharding)
            return leaf

        model = jax.tree_util.tree_map_with_path(
            restore, template, is_leaf=lambda node: isinstance(node, WeightMatrix)
        )
        if unused := arrays.keys() - consumed:
            raise ValueError(f"Unconsumed packed checkpoint tensors: {sorted(unused)}")
        if unused := specs.keys() - consumed_specs:
            raise ValueError(f"Unconsumed packed checkpoint specifications: {sorted(unused)}")
    assert isinstance(model, LanguageModel)
    return model
