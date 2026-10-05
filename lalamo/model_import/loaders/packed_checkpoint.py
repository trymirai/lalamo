import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cattrs
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.compressed.direction import DirectionMatrix, DirectionSpec
from lalamo.compressed.lattice import LatticeKind, LatticeMatrix, LatticeSpec, odd_integer_table
from lalamo.compressed.qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
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
    # Lalamo's own saves also tag trellis leaves "QtipGaussianSpec", but keep their tables under each matrix.
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

        def row_stack(parts: tuple[QtipGaussianMatrix | LatticeMatrix, ...], is_sharded: bool) -> RowStackMatrix:
            spec = RowStackSpec(tuple((part.shape[0], part.spec) for part in parts))
            return RowStackMatrix(spec=spec, sharding_config=sharding_config, is_sharded=is_sharded, parts=parts)

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
                    spec = converter.structure(saved, QtipGaussianSpec)
                    matrix = QtipGaussianMatrix(
                        spec=spec,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=spec.msb_first_codes(parameter(path / "codes"), columns),
                        scales=parameter(path / "scales"),
                        gains=parameter(path / "gains"),
                        table=parameter(table_name),
                        signs=parameter(f"qtip_shared.signs_{columns}"),
                        small_q=parameter(f"qtip_shared.q_{columns}"),
                        pre_gains=parameter_tuple(path / "pre_gains", spec.pre_gain_count),
                        post_gains=parameter_tuple(path / "post_gains", len(spec.post_gain_axes)),
                    )
                case "RowStackSpec":
                    # Each part's saved spec is inline in the stack's; the parts have no spec entries of their own.
                    parts = []
                    for index, (rows, part_spec) in enumerate(saved.pop("parts")):
                        part = weight(path / "parts" / index, part_spec, template)
                        assert isinstance(part, QtipGaussianMatrix | LatticeMatrix)
                        assert part.shape[0] == rows
                        parts.append(part)
                    layout = saved.pop("layout")
                    assert layout == Layout.OUTPUT_INPUT and not saved, f"Unexpected row stack {saved} at {path}"
                    matrix = row_stack(tuple(parts), is_sharded)
                case "D4S4Spec" | "I3S4Spec" | "I4S4Spec" as kind_name:
                    assert "kind" not in saved
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
                    matrix = LatticeMatrix(
                        spec=lattice,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=table,
                        signs=parameter(path / sign_name),
                        post_gains=parameter_tuple(path / "post_gains", len(lattice.post_gain_axes)),
                    )
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
                    assert isinstance(qkv, QtipGaussianMatrix | LatticeMatrix)
                    assert isinstance(gate, QtipGaussianMatrix | LatticeMatrix)
                    return load_as(leaf, row_stack((qkv, gate), leaf.is_sharded))
                return load_as(leaf, weight(path, saved_spec(path), leaf))
            if isinstance(leaf, ShapeDtypeStruct | Array):
                value = parameter(path)
                assert value.shape == leaf.shape, f"Saved shape differs from model at {path}"
                if dtype is not None:
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
