import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.compressed.hybrid import HybridMatrix
from lalamo.compressed.incoherence import (
    HadamardProcessor,
    IncoherenceProcessingMode,
    IncoherenceProcessor,
    KroneckerProcessor,
)
from lalamo.compressed.lattice import (
    COLUMNS_PER_LADDER_INDEX,
    D4_LADDER,
    D4_TABLE,
    LADDER_INDEX_BITS,
    LatticeMatrix,
    LatticeSpec,
)
from lalamo.compressed.mlx import MLXMatrix, MLXSpec
from lalamo.compressed.output_stack import OutputStackMatrix, OutputStackSpec
from lalamo.compressed.qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec, codebook_table
from lalamo.compressed.utils.packing import unpack_uint8_to_uint
from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.models.language_model import LanguageModel
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import Layout, ShapeDtypeMatrix, WeightMatrix


def merge_rows(top: WeightMatrix, bottom: WeightMatrix) -> WeightMatrix | None:
    if (top.spec, top.dtype) != (bottom.spec, bottom.dtype):
        return None
    if isinstance(top, MLXMatrix | QtipGaussianMatrix):
        return jax.tree.map(lambda *planes: jnp.concatenate(planes), top, bottom)
    return None


def native_config(value: JSON) -> JSON:
    # Older packages carry a null pard_token and save attention as separate qkv and gate projections.
    if isinstance(value, list):
        return [native_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = dict(value)
    pard_token = value.pop("pard_token", None)
    assert pard_token is None, "PARD checkpoints are not supported"
    if value.get("type") == "AttentionConfig" and "qkv_projection_config" in value:
        qkv = value.pop("qkv_projection_config")
        gate = value.pop("gate_projection_config")
        has_biases = value.pop("has_qkv_biases")
        assert gate == qkv and not has_biases, "Legacy qkv and gate projections share one bias-free config"
        value |= {"qkvg_projection_config": qkv, "has_qkvg_biases": False, "has_gate": True}
    return {name: native_config(item) for name, item in value.items()}


def import_packed_checkpoint(
    directory: Path | str, sharding_config: ShardingConfig, dtype: DTypeLike | None = None
) -> LanguageModel:
    directory = Path(directory)
    config = BaseModelConfig.from_json(native_config(json.loads((directory / "config.json").read_text())))
    with (directory / "model.safetensors").open("rb") as weights_file:
        metadata, arrays, _ = safe_read(weights_file)
        specs = {name: json.loads(value) for name, value in (metadata or {}).items()}
        template = config.init_from_directory(directory, EmptyInitializer(dtype, sharding_config))
        assert isinstance(template, LanguageModel)
        return load_packed_checkpoint(template, ExportResults(arrays, specs), dtype)


def load_packed_checkpoint(template: LanguageModel, exported: ExportResults, dtype: DTypeLike | None) -> LanguageModel:
    arrays, specs = exported
    sharding_config = template.sharding_config
    # RoPE is recomputed from the config, as uzu does; older packages' saved tables match it to one float32 ulp.
    consumed = {name for name in arrays if name.startswith("decoder.transformer.ropes.")}
    consumed_specs: set[str] = set()

    def parameter(name: str) -> Array:
        consumed.add(name)
        return arrays[name]

    def saved_spec(path: ParameterPath) -> dict[str, Any]:
        consumed_specs.add(path / "spec")
        spec = specs[path / "spec"]
        assert isinstance(spec, dict)
        return spec

    @cache
    def check_codebook(table_name: str, vector_width: int) -> None:
        assert np.allclose(parameter(table_name), codebook_table(vector_width), atol=1e-6), f"Unexpected {table_name}"

    def fold_post_gains(path: ParameterPath, scale: Array, axes: Iterable[str]) -> Array:
        scale = scale.astype(jnp.float32)
        for index, axis in enumerate(axes):
            gain = parameter(path / "post_gains" / index)
            match axis:
                case "row":
                    scale = scale * gain.astype(jnp.float32)
                case "column":
                    if not bool(jnp.all(gain == 1)):
                        raise ValueError(f"Column post-gains cannot fold into row scales at {path}")
                case _:
                    raise ValueError(f"Unknown post-gain axis {axis!r} at {path}")
        return scale

    def output_stack(parts: tuple[WeightMatrix, ...]) -> WeightMatrix:
        assert parts
        if len(parts) == 1:
            return parts[0]
        return OutputStackMatrix(OutputStackSpec(), sharding_config, parts[0].is_sharded, parts)

    def stack_projections(hybrids: Iterable[WeightMatrix]) -> WeightMatrix:
        # Neighbours under one saved rotation become one hybrid, in which neighbouring same-format leaves merge.
        groups: list[tuple[IncoherenceProcessor, list[WeightMatrix]]] = []
        for hybrid in hybrids:
            assert isinstance(hybrid, HybridMatrix) and hybrid.incoherence_signs is not None
            if not groups or not eqx.tree_equal(groups[-1][0], hybrid.incoherence_signs):
                groups.append((hybrid.incoherence_signs, []))
            leaves = groups[-1][1]
            if leaves and (joined := merge_rows(leaves[-1], hybrid.quantized)) is not None:
                leaves[-1] = joined
            else:
                leaves.append(hybrid.quantized)
        fused = tuple(HybridMatrix.of(output_stack(tuple(leaves)), rotation) for rotation, leaves in groups)
        return output_stack(fused)

    def load_weight(path: ParameterPath, saved: dict[str, Any], target: ShapeDtypeMatrix) -> WeightMatrix:
        columns = target.shape[1]
        is_sharded = target.is_sharded
        saved = dict(saved)
        matrix: WeightMatrix
        match saved.pop("type"):
            case "FullPrecisionSpec":
                matrix = target.load_exported(exported, prefix=path)
                consumed.add(path / "weights")
            case "QtipGaussianSpec":
                table_name = saved.pop("table", f"qtip_shared.codebook_v{saved['vector_width']}")
                layout, scale_dtype = saved.pop("layout"), saved.pop("scale_dtype", "float16")
                scales, gains = parameter(path / "scales"), parameter(path / "gains")
                assert layout == Layout.OUTPUT_INPUT, f"QTIP leaves are stored output-input at {path}"
                assert scales.dtype == scale_dtype, f"Saved scales differ from their spec at {path}"
                row_scale = scales.astype(jnp.float32) * gains.astype(jnp.float32)
                for index in range(saved.pop("pre_gain_count", 0)):
                    row_scale = row_scale * parameter(path / "pre_gains" / index).astype(jnp.float32)
                row_scale = fold_post_gains(path, row_scale, saved.pop("post_gain_axes", ()))
                # A table keeps one tape row per vocabulary entry: in an embedding slot those rows are its inputs.
                leaf = QtipGaussianMatrix(
                    spec=QtipGaussianSpec.from_json(
                        {"type": "QtipGaussianSpec", **saved, "layout": target.spec.layout}
                    ),
                    sharding_config=sharding_config,
                    is_sharded=is_sharded,
                    # The saved gains carry the matrix dtype.
                    dtype_=gains.dtype,
                    columns=columns,
                    codes=parameter(path / "codes"),
                    scales=row_scale,
                )
                check_codebook(table_name, leaf.spec.vector_width)
                mode = IncoherenceProcessingMode.INPUT
                if target.spec.layout == Layout.INPUT_OUTPUT:
                    mode = IncoherenceProcessingMode.OUTPUT
                rotation = KroneckerProcessor.of(
                    parameter(f"qtip_shared.signs_{columns}").astype(jnp.int32),
                    parameter(f"qtip_shared.q_{columns}"),
                    mode,
                )
                matrix = HybridMatrix.of(leaf, rotation)
            case "RowStackSpec":
                # Each part's saved spec is inline in the stack's; the parts have no spec entries of their own.
                rows, part_specs = zip(*saved.pop("parts"), strict=True)
                assert saved == {"layout": Layout.OUTPUT_INPUT}, f"Unexpected row stack {saved} at {path}"
                parts = [load_weight(path / "parts" / index, spec, target) for index, spec in enumerate(part_specs)]
                assert tuple(part.shape[0] for part in parts) == rows
                matrix = stack_projections(parts)
            case "HybridSpec":
                # Saved as HybridMatrix.export writes it, minus the spec entry of the inner matrix.
                inner_spec = {path / "quantized" / "spec": saved["quantization_spec"]}
                matrix = target.load_exported(ExportResults(arrays, {**specs, **inner_spec}), prefix=path)
                consumed.update(path / name for name in matrix.export().arrays)
            case "I4S4Spec":
                # Level 2c - 15 times the group scale s = row scale * ladder value is (2s) * c - 15s: affine, once
                # the nibbles are swapped to low-first. Gains fold into s in float32, exact for powers of two.
                row_scales = parameter(path / "row_scales")
                groups = unpack_uint8_to_uint(parameter(path / "ladder_indices"), LADDER_INDEX_BITS)
                ladder = jnp.take(parameter(path / "ladder"), groups).astype(jnp.float32)
                row_scale = fold_post_gains(path, row_scales, saved.pop("post_gain_axes", ()))
                scales = row_scale[:, None] * ladder
                assert saved == {"layout": Layout.OUTPUT_INPUT}, f"Unexpected I4S4 leaf {saved} at {path}"
                codes = parameter(path / "codes")
                quantized = MLXSpec(4, COLUMNS_PER_LADDER_INDEX, Layout.OUTPUT_INPUT).from_packed_parameters(
                    packed_weights=(codes << 4) | (codes >> 4),
                    scales=(2 * scales).astype(dtype or row_scales.dtype),
                    biases=(-15 * scales).astype(dtype or row_scales.dtype),
                    sharding_config=sharding_config,
                    is_sharded=is_sharded,
                )
                signs = HadamardProcessor.of(parameter(path / "input_hadamard_factors"), None)
                matrix = HybridMatrix.of(quantized, signs)
            case "D4S4Spec":
                if saved.pop("post_gain_axes", ()):
                    raise ValueError(f"Lattice post-gains do not commute with the Hadamard rotation at {path}")
                lattice = LatticeSpec.from_json({"type": "LatticeSpec", "kind": "d4", **saved})
                leaf = LatticeMatrix(
                    spec=lattice,
                    sharding_config=sharding_config,
                    is_sharded=is_sharded,
                    codes=parameter(path / "codes"),
                    row_scales=parameter(path / "row_scales"),
                    ladder_indices=parameter(path / "ladder_indices"),
                )
                assert np.array_equal(parameter(path / "table"), D4_TABLE), f"Unexpected D4 table at {path}"
                assert np.array_equal(parameter(path / "ladder"), D4_LADDER), f"Unexpected D4 ladder at {path}"
                if lattice.layout == Layout.INPUT_OUTPUT:
                    rotation = HadamardProcessor.of(None, parameter(path / "output_hadamard_factors"))
                else:
                    rotation = HadamardProcessor.of(parameter(path / "input_hadamard_factors"), None)
                matrix = HybridMatrix.of(leaf, rotation)
            case other:
                raise ValueError(f"Unsupported packed checkpoint weight format {other!r} at {path}")
        if dtype is not None:
            matrix = matrix.astype(dtype)
        return matrix

    def restore(jax_path: tuple[object, ...], leaf: object) -> object:
        path = ParameterPath() / jax_path
        if isinstance(leaf, ShapeDtypeStruct | Array):
            return load_as(leaf, parameter(path))
        if not isinstance(leaf, ShapeDtypeMatrix):
            return leaf
        # Older packages save the attention projection as a qkv leaf and a gate leaf.
        parent = path.removesuffix("qkvg_projection.weights")
        if parent + "qkv_projection.weights.spec" in specs:
            qkv, gate = (ParameterPath(f"{parent}{name}_projection.weights") for name in ("qkv", "gate"))
            parts = [load_weight(part, saved_spec(part), leaf) for part in (qkv, gate)]
            return load_as(leaf, stack_projections(parts))
        return load_as(leaf, load_weight(path, saved_spec(path), leaf))

    model = jax.tree_util.tree_map_with_path(restore, template, is_leaf=lambda node: isinstance(node, WeightMatrix))
    if unused := arrays.keys() - consumed:
        raise ValueError(f"Unconsumed packed checkpoint tensors: {sorted(unused)}")
    if unused := specs.keys() - consumed_specs:
        raise ValueError(f"Unconsumed packed checkpoint specifications: {sorted(unused)}")
    assert isinstance(model, LanguageModel)
    return model
