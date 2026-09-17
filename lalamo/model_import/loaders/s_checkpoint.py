import json
from dataclasses import replace
from pathlib import Path

import cattrs
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike

from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.s_direction import SDirectionMatrix, SDirectionSpec
from lalamo.compressed.s_surface import SSurfaceKind, SSurfaceMatrix, SSurfaceSpec
from lalamo.compressed.s_trellis import STrellisMatrix, STrellisSpec
from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.modules.rope import SavedRoPEConfig, UnscaledRoPEConfig
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.registry_abc import make_registry_abc_converter
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import FullPrecisionMatrix, FullPrecisionSpec, Layout, ShapeDtypeMatrix, WeightMatrix


def _native_config(value: JSON) -> JSON:
    if isinstance(value, list):
        return [_native_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "pard_token" in value:
        value = dict(value)
        assert value.pop("pard_token") is None, "PARD checkpoints are not supported"
    if value.get("type") == "AttentionConfig" and "qkv_projection_config" in value:
        assert not {"qkvg_projection_config", "has_qkvg_biases", "has_gate"} & value.keys()
        value = dict(value)
        value["qkvg_projection_config"] = value.pop("qkv_projection_config")
        assert value.pop("gate_projection_config") == value["qkvg_projection_config"]
        value["has_qkvg_biases"] = value.pop("has_qkv_biases")
        assert not value["has_qkvg_biases"]
        value["has_gate"] = True
    return {name: _native_config(item) for name, item in value.items()}


def is_s_checkpoint(config: JSON, metadata: dict[str, JSON]) -> bool:
    return _native_config(config) != config or any(
        isinstance(spec, dict) and spec.get("type") in ("QtipGaussianSpec", "D4S4Spec", "I3S4Spec", "I4S4Spec")
        for spec in metadata.values()
    )


def load_s_checkpoint(
    directory: Path | str, sharding_config: ShardingConfig, dtype: DTypeLike | None = None
) -> LanguageModel:
    """Import original S exports without fitting their saved dense or packed weights."""
    directory = Path(directory)
    config = BaseModelConfig.from_json(_native_config(json.loads((directory / "config.json").read_text())))
    assert isinstance(config, LanguageModelConfig)
    converter = cattrs.Converter(forbid_extra_keys=True)

    with (directory / "model.safetensors").open("rb") as stream:
        metadata, arrays = safe_read(stream)
        assert metadata is not None
        transformer = config.decoder_config.transformer_config
        rope_configs = dict.fromkeys(layer.rope_config for layer in transformer.layer_configs if layer.rope_config)
        saved_ropes = {}
        for index, rope in enumerate(rope_configs):
            cosine = f"decoder.transformer.ropes.{index}.cosines"
            sine = f"decoder.transformer.ropes.{index}.sines"
            if cosine in arrays or sine in arrays:
                assert cosine in arrays and sine in arrays, "Saved RoPE requires both tables"
                assert isinstance(rope, UnscaledRoPEConfig | SavedRoPEConfig)
                saved_ropes[rope] = SavedRoPEConfig(rope.base, rope.max_sequence_length, rope.head_dim)
        if saved_ropes:
            layers = tuple(
                replace(layer, rope_config=saved_ropes.get(layer.rope_config, layer.rope_config))
                for layer in transformer.layer_configs
            )
            config = replace(
                config,
                decoder_config=replace(
                    config.decoder_config, transformer_config=replace(transformer, layer_configs=layers)
                ),
            )
        template = config.init_from_directory(directory, EmptyInitializer(dtype, sharding_config))
        shared: dict[str, Array] = {}
        parameters: set[str] = set()
        specifications: set[str] = set()

        def parameter(name: str) -> Array:
            parameters.add(name)
            if not name.startswith("qtip_shared."):
                return arrays[name]
            if name not in shared:
                shared[name] = arrays[name]
            return shared[name]

        def weight(path: ParameterPath, template: ShapeDtypeMatrix) -> WeightMatrix:
            columns = template.shape[1]
            is_sharded = template.is_sharded
            saved = json.loads(metadata[path / "spec"])
            specifications.add(path / "spec")
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
                    assert isinstance(table_name, str)
                    spec = converter.structure(saved, STrellisSpec)
                    matrix = STrellisMatrix(
                        spec=spec,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        scales=parameter(path / "scales"),
                        gains=parameter(path / "gains"),
                        table=parameter(table_name),
                        signs=parameter(f"qtip_shared.signs_{columns}"),
                        small_q=parameter(f"qtip_shared.q_{columns}"),
                        pre_gains=tuple(
                            parameter(path / f"pre_gains.{index}") for index in range(spec.pre_gain_count)
                        ),
                        post_gains=tuple(
                            parameter(path / f"post_gains.{index}") for index in range(len(spec.post_gain_axes))
                        ),
                    )
                case "RowStackSpec":
                    stack = make_registry_abc_converter(forbid_extra_keys=True).structure(
                        {"type": "RowStackSpec", **saved}, RowStackSpec
                    )
                    parts = []
                    for index in range(len(stack.parts)):
                        part = weight(path / "parts" / index, template)
                        assert isinstance(part, STrellisMatrix | SSurfaceMatrix)
                        parts.append(part)
                    matrix = RowStackMatrix(
                        spec=stack,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        parts=tuple(parts),
                    )
                case "D4S4Spec" | "I3S4Spec" | "I4S4Spec" as kind_name:
                    assert "kind" not in saved
                    kind = SSurfaceKind(kind_name[:2].lower())
                    surface = converter.structure({**saved, "kind": kind}, SSurfaceSpec)
                    states = 1 << surface.code_bits
                    table = (
                        parameter(path / "table")
                        if kind == SSurfaceKind.D4
                        else jnp.arange(1 - states, states, 2, dtype=jnp.int8)[:, None]
                    )
                    sign_name = (
                        "output_hadamard_factors"
                        if surface.layout == Layout.INPUT_OUTPUT
                        else "input_hadamard_factors"
                    )
                    matrix = SSurfaceMatrix(
                        spec=surface,
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=table,
                        signs=parameter(path / sign_name),
                        post_gains=tuple(
                            parameter(path / f"post_gains.{index}") for index in range(len(surface.post_gain_axes))
                        ),
                    )
                case "SDirectionSpec":
                    matrix = SDirectionMatrix(
                        spec=converter.structure(saved, SDirectionSpec),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        levels=parameter(path / "levels"),
                        unit_scale=parameter(path / "unit_scale"),
                        mean_norm=parameter(path / "mean_norm"),
                        tail=parameter(path / "tail"),
                    )
                case other:
                    raise ValueError(f"Unsupported S checkpoint weight format {other!r} at {path}")
            return matrix if dtype is None else matrix.astype(dtype)

        def restore(jax_path: tuple[object, ...], leaf: object) -> object:
            path = ParameterPath() / jax_path
            if isinstance(leaf, ShapeDtypeMatrix):
                parent = path.removesuffix("qkvg_projection.weights")
                if path.endswith(".qkvg_projection.weights") and parent + "qkv_projection.weights.spec" in metadata:
                    assert path / "spec" not in metadata
                    qkv = weight(ParameterPath(parent + "qkv_projection.weights"), leaf)
                    gate = weight(ParameterPath(parent + "gate_projection.weights"), leaf)
                    if isinstance(qkv, FullPrecisionMatrix) and isinstance(gate, FullPrecisionMatrix):
                        assert qkv.spec == gate.spec == FullPrecisionSpec()
                        assert qkv.dtype == gate.dtype
                        return load_as(
                            leaf,
                            replace(qkv, weights=jnp.concatenate((qkv.weights, gate.weights))),
                        )
                    assert isinstance(qkv, STrellisMatrix | SSurfaceMatrix)
                    assert isinstance(gate, STrellisMatrix | SSurfaceMatrix)
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
            raise ValueError(f"Unconsumed S checkpoint tensors: {sorted(unused)}")
        if unused := metadata.keys() - specifications:
            raise ValueError(f"Unconsumed S checkpoint specifications: {sorted(unused)}")
    assert isinstance(model, LanguageModel)
    return model
