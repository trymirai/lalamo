import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cattrs
import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, Float32

from lalamo.compressed.direction import DirectionMatrix, DirectionSpec
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
from lalamo.compressed.qtip_gaussian import COLUMN_CLASSES, STATE_BITS, QtipGaussianMatrix, QtipGaussianSpec
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.trellis import states_to_levels
from lalamo.compressed.utils.packing import unpack_uint8_to_uint
from lalamo.compressed.utils.post_gains import GainAxis
from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.modules.linear import Linear
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, Layout, ShapeDtypeMatrix, WeightMatrix

FOLDED_I4S4_QUANTIZATION = MLXSpec(bits=4, group_size=COLUMNS_PER_LADDER_INDEX, layout=Layout.OUTPUT_INPUT)


@dataclass(frozen=True)
class PackedQtipGaussianSpec:
    """A trellis leaf as packages save it: its row scale split into scales, gains and further per-row gains."""

    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 8]
    restart_columns: Literal[0, 64]
    scale_dtype: Literal["float16", "float32"] = "float16"
    pre_gain_count: int = 0
    post_gain_axes: tuple[GainAxis, ...] = ()


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


def folded_row_scale(path: str, scales: Array, axes: tuple[GainAxis, ...], *gains: Array) -> Float32[Array, " rows"]:
    """`scales` times every row gain in float32: the one row scale that replaces them. Column gains must be 1."""
    if not all(bool(jnp.all(gain == 1)) for axis, gain in zip(axes, gains, strict=True) if axis == GainAxis.COLUMN):
        raise ValueError(f"Column post-gains other than 1 do not fold into the row scales at {path}")
    return math.prod(
        (gain.astype(jnp.float32) for axis, gain in zip(axes, gains, strict=True) if axis == GainAxis.ROW),
        start=scales.astype(jnp.float32),
    )


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


def is_packed_checkpoint(config: JSON, metadata: dict[str, JSON], tensor_names: Iterable[str]) -> bool:
    # Lalamo's own saves also tag trellis leaves "QtipGaussianSpec", but keep a five-float codebook under each matrix.
    specs: list[dict[str, Any]] = [spec for spec in metadata.values() if isinstance(spec, dict)]
    parts = [part for spec in specs if spec.get("type") == "RowStackSpec" for _, part in spec["parts"]]
    return (
        native_config(config) != config
        or any(name.startswith("qtip_shared.") for name in tensor_names)
        or any(spec.get("type") in ("D4S4Spec", "I3S4Spec", "I4S4Spec", "SDirectionSpec") for spec in (*specs, *parts))
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
        template = config.init_from_directory(directory, EmptyInitializer(dtype, sharding_config))
        output_dims = {
            ParameterPath() / jax_path: linear.output_dims
            for jax_path, linear in jax.tree_util.tree_leaves_with_path(
                template, is_leaf=lambda node: isinstance(node, Linear)
            )
            if isinstance(linear, Linear)
        }
        shared: dict[str, Array] = {}
        # RoPE is recomputed from the config, as uzu does; older packages' saved tables match it to one float32 ulp.
        parameters = {name for name in arrays if name.startswith("decoder.transformer.ropes.")}
        specifications: set[str] = set()

        def parameter(name: str) -> Array:
            parameters.add(name)
            if not name.startswith("qtip_shared."):
                return arrays[name]
            if name not in shared:
                shared[name] = arrays[name]
            return shared[name]

        def parameter_tuple(path: ParameterPath, count: int) -> tuple[Array, ...]:
            return tuple(parameter(path / index) for index in range(count))

        def saved_spec(path: ParameterPath) -> dict[str, Any]:
            specifications.add(path / "spec")
            return json.loads(metadata[path / "spec"])

        def row_stack(parts: tuple[WeightMatrix, ...], is_sharded: bool) -> RowStackMatrix:
            spec = RowStackSpec(tuple((part.shape[0], part.spec) for part in parts))
            return RowStackMatrix(spec=spec, sharding_config=sharding_config, is_sharded=is_sharded, parts=parts)

        def stacked(hybrids: tuple[HybridMatrix, ...], is_sharded: bool) -> WeightMatrix:
            """Neighbours under one saved rotation become one hybrid; same-format leaves merge."""
            groups: list[tuple[IncoherenceSigns | KroneckerRotation, list[WeightMatrix]]] = []
            for part in hybrids:
                rotation = part.incoherence_signs
                assert rotation is not None
                if not groups or not (
                    jax.tree.structure(groups[-1][0]) == jax.tree.structure(rotation)
                    and all(map(jnp.array_equal, jax.tree.leaves(groups[-1][0]), jax.tree.leaves(rotation)))
                ):
                    groups.append((rotation, []))
                quantized = part.quantized
                for leaf in quantized.parts if isinstance(quantized, RowStackMatrix) else (quantized,):
                    leaves = groups[-1][1]
                    if leaves and (joined := merged(leaves[-1], leaf)) is not None:
                        leaves[-1] = joined
                    else:
                        leaves.append(leaf)
            fused = tuple(
                HybridMatrix.of(
                    leaves[0] if len(leaves) == 1 else row_stack(tuple(leaves), is_sharded),
                    rotation,
                    sharding_config,
                    is_sharded,
                )
                for rotation, leaves in groups
            )
            return fused[0] if len(fused) == 1 else row_stack(fused, is_sharded)

        def weight(path: ParameterPath, saved: dict[str, Any], template: ShapeDtypeMatrix) -> WeightMatrix:
            columns = template.shape[1]
            is_sharded = template.is_sharded
            saved = dict(saved)
            matrix: WeightMatrix
            match saved.pop("type"):
                case "FullPrecisionSpec":
                    spec = converter.structure(saved, FullPrecisionSpec)
                    matrix = spec.compress(
                        spec.layout.to_output_input(parameter(path / "weights")),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                    )
                case "QtipGaussianSpec":
                    table_name = saved.pop("table", f"qtip_shared.codebook_v{saved['vector_width']}")
                    layout = saved.pop("layout")
                    assert layout == Layout.OUTPUT_INPUT, f"Trellis leaves are stored output-input, got {layout}"
                    packed = converter.structure(saved, PackedQtipGaussianSpec)
                    gains = parameter(path / "gains")
                    # Every saved gain multiplies a whole row, so one float32 scale per row replaces them all.
                    scales = folded_row_scale(
                        path,
                        parameter(path / "scales"),
                        (GainAxis.ROW,) * (1 + packed.pre_gain_count) + packed.post_gain_axes,
                        gains,
                        *parameter_tuple(path / "pre_gains", packed.pre_gain_count),
                        *parameter_tuple(path / "post_gains", len(packed.post_gain_axes)),
                    )
                    leaf = QtipGaussianMatrix(
                        spec=QtipGaussianSpec(packed.vector_width, packed.transition_bits, packed.restart_columns),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        # The saved gains carried the matrix dtype.
                        dtype_=gains.dtype,
                        columns=columns,
                        codes=parameter(path / "codes"),
                        scales=scales,
                        codebook=codebook_from_table(parameter(table_name)),
                    )
                    rotation = KroneckerRotation(
                        signs=parameter(f"qtip_shared.signs_{columns}"), small_q=parameter(f"qtip_shared.q_{columns}")
                    )
                    matrix = HybridMatrix.of(leaf, rotation, sharding_config, is_sharded)
                case "RowStackSpec":
                    # Each part's saved spec is inline in the stack's; the parts have no spec entries of their own.
                    parts: list[HybridMatrix] = []
                    for index, (rows, part_spec) in enumerate(saved.pop("parts")):
                        part = weight(path / "parts" / index, part_spec, template)
                        assert isinstance(part, HybridMatrix)
                        assert part.shape[0] == rows
                        parts.append(part)
                    layout = saved.pop("layout")
                    assert layout == Layout.OUTPUT_INPUT and not saved, f"Unexpected row stack {saved} at {path}"
                    matrix = stacked(tuple(parts), is_sharded)
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
                case "I4S4Spec" if saved["layout"] == Layout.OUTPUT_INPUT:
                    # Level 2c - 15 times the group scale s = row scale * ladder value is (2s) * c - 15s: affine, once
                    # the nibbles are swapped to low-first. Gains fold into s in float32, exact for powers of two.
                    axes = tuple(map(GainAxis, saved.pop("post_gain_axes", ())))
                    row_scales = parameter(path / "row_scales")
                    groups = unpack_uint8_to_uint(parameter(path / "ladder_indices"), LADDER_INDEX_BITS)
                    gains = parameter_tuple(path / "post_gains", len(axes))
                    row_scale = folded_row_scale(path, row_scales, axes, *gains)
                    codes = parameter(path / "codes")
                    scales = row_scale[:, None] * jnp.take(parameter(path / "ladder"), groups).astype(jnp.float32)
                    quantized = FOLDED_I4S4_QUANTIZATION.from_packed_parameters(
                        # The I4 packer puts the even column in the high nibble; MLX reads the low one first.
                        packed_weights=(codes << 4) | (codes >> 4),
                        scales=(2 * scales).astype(dtype or row_scales.dtype),
                        biases=(-15 * scales).astype(dtype or row_scales.dtype),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                    )
                    signs = IncoherenceSigns(parameter(path / "input_hadamard_factors"), None)
                    matrix = HybridMatrix.of(quantized, signs, sharding_config, is_sharded)
                case "D4S4Spec" | "I3S4Spec" | "I4S4Spec" as kind_name:
                    assert "kind" not in saved
                    if saved.pop("post_gain_axes", ()):
                        raise ValueError(f"Lattice post-gains do not commute with the Hadamard rotation at {path}")
                    kind = LatticeKind(kind_name[:2].lower())
                    lattice = converter.structure({**saved, "kind": kind}, LatticeSpec)
                    if kind == LatticeKind.D4:
                        table = parameter(path / "table")
                    else:
                        table = odd_integer_table(lattice.code_bits)
                    sign_name = (
                        "output_hadamard_factors"
                        if lattice.layout == Layout.INPUT_OUTPUT
                        else "input_hadamard_factors"
                    )
                    leaf = LatticeMatrix(
                        spec=lattice,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=table,
                    )
                    signs = parameter(path / sign_name)
                    is_output = lattice.layout == Layout.INPUT_OUTPUT
                    rotation = IncoherenceSigns(
                        input_signs=None if is_output else signs, output_signs=signs if is_output else None
                    )
                    matrix = HybridMatrix.of(leaf, rotation, sharding_config, is_sharded)
                case "SDirectionSpec":
                    matrix = DirectionMatrix(
                        spec=converter.structure(saved, DirectionSpec),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        levels=parameter(path / "levels"),
                        unit_scale=parameter(path / "unit_scale"),
                        mean_norm=parameter(path / "mean_norm"),
                        tail=parameter(path / "tail"),
                    )
                case other:
                    raise ValueError(f"Unsupported packed checkpoint weight format {other!r} at {path}")
            return matrix if dtype is None else matrix.astype(dtype)

        def restore(jax_path: tuple[object, ...], leaf: object) -> object:
            path = ParameterPath() / jax_path
            if isinstance(leaf, ShapeDtypeMatrix):
                parent = path.removesuffix("qkvg_projection.weights")
                if path.endswith(".qkvg_projection.weights") and parent + "qkv_projection.weights.spec" in metadata:
                    assert path / "spec" not in metadata
                    qkv_path = ParameterPath(parent + "qkv_projection.weights")
                    gate_path = ParameterPath(parent + "gate_projection.weights")
                    qkv = weight(qkv_path, saved_spec(qkv_path), leaf)
                    gate = weight(gate_path, saved_spec(gate_path), leaf)
                    *qkv_rows, gate_rows = output_dims[ParameterPath(parent + "qkvg_projection")]
                    assert (qkv.shape[0], gate.shape[0]) == (sum(qkv_rows), gate_rows), f"Row split differs at {path}"
                    if isinstance(qkv, FullPrecisionMatrix) and isinstance(gate, FullPrecisionMatrix):
                        assert qkv.spec == gate.spec == FullPrecisionSpec()
                        assert qkv.dtype == gate.dtype
                        return load_as(
                            leaf,
                            replace(qkv, weights=jnp.concatenate((qkv.weights, gate.weights))),
                        )
                    assert isinstance(qkv, HybridMatrix) and isinstance(gate, HybridMatrix)
                    return load_as(leaf, stacked((qkv, gate), leaf.is_sharded))
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
        if unused := arrays.keys() - parameters:
            raise ValueError(f"Unconsumed packed checkpoint tensors: {sorted(unused)}")
        if unused := metadata.keys() - specifications:
            raise ValueError(f"Unconsumed packed checkpoint specifications: {sorted(unused)}")
    assert isinstance(model, LanguageModel)
    return model
