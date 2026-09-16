import json
from pathlib import Path

import cattrs
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jaxtyping import Array

from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.s_surface import SSurfaceKind, SSurfaceMatrix, SSurfaceSpec
from lalamo.compressed.s_trellis import STrellisMatrix, STrellisSpec
from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.models.language_model import LanguageModel
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import Layout, ShapeDtypeMatrix, WeightMatrix


def _native_config(value: JSON) -> JSON:
    if isinstance(value, list):
        return [_native_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "AttentionConfig":
        assert not {"qkvg_projection_config", "has_qkvg_biases", "has_gate"} & value.keys()
        value = dict(value)
        value["qkvg_projection_config"] = value.pop("qkv_projection_config")
        assert value.pop("gate_projection_config") == value["qkvg_projection_config"]
        value["has_qkvg_biases"] = value.pop("has_qkv_biases")
        assert not value["has_qkvg_biases"]
        value["has_gate"] = True
    return {name: _native_config(item) for name, item in value.items()}


def load_s_checkpoint(directory: Path | str, sharding_config: ShardingConfig) -> LanguageModel:
    """Import the physical HYB036 S package, preserving its packed parameters."""
    directory = Path(directory)
    config = BaseModelConfig.from_json(_native_config(json.loads((directory / "config.json").read_text())))
    template = config.init_from_directory(directory, EmptyInitializer(None, sharding_config))
    assert isinstance(template, LanguageModel)
    converter = cattrs.Converter(forbid_extra_keys=True)

    with (directory / "model.safetensors").open("rb") as stream:
        metadata, arrays = safe_read(stream)
        assert metadata is not None
        parameters: dict[str, Array] = {}
        specifications: set[str] = set()

        def parameter(name: str) -> Array:
            # Shared tables and rotations are read once. The same map also
            # proves that import consumed every saved tensor.
            if name not in parameters:
                parameters[name] = arrays[name]
            return parameters[name]

        def weight(path: ParameterPath, template: ShapeDtypeMatrix) -> WeightMatrix:
            columns = template.shape[1]
            is_sharded = template.is_sharded
            saved = json.loads(metadata[path / "spec"])
            specifications.add(path / "spec")
            match saved.pop("type"):
                case "QtipGaussianSpec":
                    spec = converter.structure(saved, STrellisSpec)
                    return STrellisMatrix(
                        spec=spec,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        scales=parameter(path / "scales"),
                        gains=parameter(path / "gains"),
                        table=parameter(f"qtip_shared.codebook_v{spec.vector_width}"),
                        signs=parameter(f"qtip_shared.signs_{columns}"),
                        small_q=parameter(f"qtip_shared.q_{columns}"),
                    )
                case "D4S4Spec" | "I3S4Spec" as kind_name:
                    assert "kind" not in saved
                    kind = SSurfaceKind.D4 if kind_name == "D4S4Spec" else SSurfaceKind.I3
                    surface = converter.structure({**saved, "kind": kind}, SSurfaceSpec)
                    table = (
                        parameter(path / "table")
                        if kind == SSurfaceKind.D4
                        else jnp.arange(-7, 8, 2, dtype=jnp.int8)[:, None]
                    )
                    sign_name = (
                        "output_hadamard_factors"
                        if surface.layout == Layout.INPUT_OUTPUT
                        else "input_hadamard_factors"
                    )
                    return SSurfaceMatrix(
                        spec=surface,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=table,
                        signs=parameter(path / sign_name),
                    )
                case other:
                    raise ValueError(f"Unsupported S checkpoint weight format {other!r} at {path}")

        def restore(jax_path: tuple[object, ...], leaf: object) -> object:
            path = ParameterPath() / jax_path
            if isinstance(leaf, ShapeDtypeMatrix):
                if path.endswith(".qkvg_projection.weights"):
                    parent = path.removesuffix("qkvg_projection.weights")
                    qkv = weight(ParameterPath(parent + "qkv_projection.weights"), leaf)
                    gate = weight(ParameterPath(parent + "gate_projection.weights"), leaf)
                    assert isinstance(qkv, STrellisMatrix) and isinstance(gate, STrellisMatrix)
                    parts = (qkv, gate)
                    matrix = RowStackMatrix(
                        spec=RowStackSpec(tuple((part.shape[0], part.spec) for part in parts)),
                        sharding_config=sharding_config,
                        is_sharded=leaf.is_sharded,
                        parts=parts,
                    )
                    return load_as(leaf, matrix)
                return load_as(leaf, weight(path, leaf))
            if isinstance(leaf, ShapeDtypeStruct | Array):
                return load_as(leaf, parameter(path))
            return leaf

        model = jax.tree_util.tree_map_with_path(
            restore, template, is_leaf=lambda node: isinstance(node, WeightMatrix)
        )
        if unused := arrays.keys() - parameters.keys():
            raise ValueError(f"Unconsumed S checkpoint tensors: {sorted(unused)}")
        if unused := metadata.keys() - specifications:
            raise ValueError(f"Unconsumed S checkpoint specifications: {sorted(unused)}")
    assert isinstance(model, LanguageModel)
    return model
