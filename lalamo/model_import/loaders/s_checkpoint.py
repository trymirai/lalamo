import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.registry_abc import make_registry_abc_converter
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import Layout, ShapeDtypeMatrix, WeightMatrix


@dataclass(frozen=True)
class _SavedAttention:
    type: Literal["AttentionConfig"]
    qkv_projection_config: LinearConfig
    gate_projection_config: LinearConfig
    out_projection_config: LinearConfig
    query_norm_config: NormalizationConfig | None
    key_norm_config: NormalizationConfig | None
    num_heads: int
    num_groups: int
    head_dim: int
    is_causal: bool
    scale: float | None
    sliding_window_size: int | None
    logit_soft_cap: float | None
    has_sinks: bool
    has_qkv_biases: bool
    has_out_biases: bool
    normalize_values: bool
    is_kv_sharing: bool

    def to_native(self) -> AttentionConfig:
        assert self.qkv_projection_config == self.gate_projection_config
        assert not self.has_qkv_biases
        return AttentionConfig(
            qkvg_projection_config=self.qkv_projection_config,
            out_projection_config=self.out_projection_config,
            query_norm_config=self.query_norm_config,
            key_norm_config=self.key_norm_config,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            is_causal=self.is_causal,
            scale=self.scale,
            sliding_window_size=self.sliding_window_size,
            logit_soft_cap=self.logit_soft_cap,
            has_sinks=self.has_sinks,
            has_qkvg_biases=self.has_qkv_biases,
            has_out_biases=self.has_out_biases,
            has_gate=True,
            normalize_values=self.normalize_values,
            is_kv_sharing=self.is_kv_sharing,
        )


def _native_config(value: JSON) -> JSON:
    if isinstance(value, list):
        return [_native_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "AttentionConfig":
        saved = make_registry_abc_converter().structure(value, _SavedAttention)
        return saved.to_native().to_json()
    return {name: _native_config(item) for name, item in value.items()}


@dataclass(frozen=True)
class _SavedTrellis:
    type: Literal["QtipGaussianSpec"]
    layout: Layout
    vector_width: Literal[2, 4]
    transition_bits: Literal[4, 6, 8]
    restart_columns: Literal[0, 64]


@dataclass(frozen=True)
class _SavedSurface:
    type: Literal["D4S4Spec", "I3S4Spec"]
    layout: Layout


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
            match saved["type"]:
                case "QtipGaussianSpec":
                    trellis = converter.structure(saved, _SavedTrellis)
                    codes = parameter(path / "codes")
                    matrix = STrellisMatrix(
                        spec=STrellisSpec(
                            trellis.vector_width, trellis.transition_bits, trellis.restart_columns, trellis.layout
                        ),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=codes,
                        scales=parameter(path / "scales"),
                        gains=parameter(path / "gains"),
                        table=parameter(f"qtip_shared.codebook_v{trellis.vector_width}"),
                        signs=parameter(f"qtip_shared.signs_{columns}"),
                        small_q=parameter(f"qtip_shared.q_{columns}"),
                    )
                    return matrix.switch_sharding_config(sharding_config)
                case "D4S4Spec" | "I3S4Spec":
                    surface = converter.structure(saved, _SavedSurface)
                    kind = SSurfaceKind.D4 if surface.type == "D4S4Spec" else SSurfaceKind.I3
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
                        spec=SSurfaceSpec(kind, surface.layout),
                        sharding_config=sharding_config,
                        is_sharded=is_sharded,
                        codes=parameter(path / "codes"),
                        row_scales=parameter(path / "row_scales"),
                        ladder_indices=parameter(path / "ladder_indices"),
                        ladder=parameter(path / "ladder"),
                        table=table,
                        signs=parameter(path / sign_name),
                    ).switch_sharding_config(sharding_config)
                case other:
                    raise ValueError(f"Unsupported S checkpoint weight format {other!r} at {path}")

        def restore(jax_path: tuple[object, ...], leaf: object) -> object:
            path = ParameterPath() / jax_path
            if isinstance(leaf, ShapeDtypeMatrix):
                if path.endswith(".qkvg_projection.weights"):
                    parent = path.removesuffix("qkvg_projection.weights")
                    parts = (
                        weight(ParameterPath(parent + "qkv_projection.weights"), leaf),
                        weight(ParameterPath(parent + "gate_projection.weights"), leaf),
                    )
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
