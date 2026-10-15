"""Original D3 weights and the complete native matrix inventory for Mirai-M."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np

if TYPE_CHECKING:
    import jax
    from lalamo.modules.linear import Linear
    from lalamo.modules.normalization import Normalization
    from lalamo.modules.transformer_layer import TransformerLayer

    from convert_d3_lalamo import D3DecisionModel, D3DecisionModelConfig


SOURCE_REVISION = "521525effa79469d899a570726547dd7f00da3af"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class SourceTensor:
    filename: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class MatrixRecord:
    canonical_name: str
    shape: tuple[int, int]
    source_modules: tuple[str, ...]
    role: str
    quantized: bool

    @property
    def weights_count(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def canonical_weights_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def storage_shape(self) -> tuple[int, int]:
        if self.role == "embedding":
            return (self.shape[1], self.shape[0])
        return self.shape


class RawCheckpoint:
    """Read without retaining a second set of device arrays during fused loading."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.tensors: dict[str, SourceTensor] = {}
        for filename in sorted(set(weight_map.values()) | {"readout.safetensors"}):
            path = self.directory / filename
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if header_size > 16 * 1024 * 1024:
                    raise ValueError(f"Implausible safetensors header: {filename}")
                header = json.loads(stream.read(header_size))
            data_start = header_size + 8
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                name = "readout.weight" if filename == "readout.safetensors" else key
                if filename == "readout.safetensors" and key != "weight":
                    raise ValueError("Unexpected decision readout tensor")
                if name in self.tensors:
                    raise ValueError(f"Duplicate source tensor: {name}")
                if filename != "readout.safetensors" and weight_map.get(name) != filename:
                    raise ValueError(f"Source index mismatch: {name}")
                start, stop = entry["data_offsets"]
                if not 0 <= start <= stop <= path.stat().st_size - data_start:
                    raise ValueError(f"Invalid source tensor offsets: {name}")
                self.tensors[name] = SourceTensor(
                    filename, name, entry["dtype"], tuple(entry["shape"]), data_start, (start, stop)
                )
        if set(self.tensors) - {"readout.weight"} != set(weight_map):
            raise ValueError("Source tensor index does not cover its complete archive")

    def array(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        dtypes = {"BF16": np.dtype(ml_dtypes.bfloat16), "F32": np.dtype(np.float32), "F16": np.dtype(np.float16)}
        if info.dtype not in dtypes:
            raise ValueError(f"Expected original floating weights, got {name}: {info.dtype}")
        dtype = dtypes[info.dtype]
        if np.prod(info.shape, dtype=np.int64) * dtype.itemsize != info.data_offsets[1] - info.data_offsets[0]:
            raise ValueError(f"Source tensor shape/size mismatch: {name}")
        return np.memmap(
            self.directory / info.filename,
            dtype=dtype,
            mode="r",
            offset=info.data_start + info.data_offsets[0],
            shape=info.shape,
        )

    def model_weights(self) -> MappedModelWeights:
        return MappedModelWeights(self)

    def config(self) -> D3DecisionModelConfig:
        from convert_d3_lalamo import decision_config, read_contract

        return decision_config(self.directory, read_contract(self.directory))

    def load_teacher(self) -> D3DecisionModel:
        from convert_d3_lalamo import load_model

        weights = self.model_weights()
        model = load_model(self.config(), weights)
        if unused := set(weights) - weights.consumed:
            raise ValueError(f"Unconsumed original text tensors: {sorted(unused)}")
        return model

    def template(self) -> D3DecisionModel:
        import jax
        import jax.numpy as jnp
        from lalamo.initializer import EmptyInitializer
        from lalamo.utils.sharding import ShardingConfig

        sharding = ShardingConfig.replicated()
        with jax.set_mesh(sharding.mesh):
            return self.config().init(EmptyInitializer(jnp.bfloat16, sharding))

    def load_layer(
        self,
        index: int,
        template: D3DecisionModel | None = None,
    ) -> TransformerLayer:
        import jax
        from lalamo.model_import.loaders.huggingface import load_transformer_layer, resolve_decoder_load_layout

        template = self.template() if template is None else template
        if not 0 <= index < len(template.transformer.layers):
            raise ValueError("Layer index is outside the D3 backbone")
        weights = self.model_weights()
        layout = resolve_decoder_load_layout(weights)
        layer = template.transformer.layers[index]
        layer_path = layout.decoder_path / "layers" / index
        with jax.set_mesh(template.sharding_config.mesh):
            return load_transformer_layer(
                layer,
                weights,
                layer_path,
                layer_path,
                layout.mixer_key[type(layer.config.mixer_config)],
                layout.mlp_key,
                layout.pre_mixer_norm_key,
                layout.pre_mlp_norm_key,
                layout.up_proj_key,
                layout.gate_proj_key,
                layout.down_proj_key,
                layout.permute_conv,
            )

    def load_tail(
        self,
        template: D3DecisionModel | None = None,
    ) -> tuple[Normalization, Linear]:
        import jax
        from lalamo.model_import.loaders.huggingface import load_linear, load_rmsnorm, resolve_decoder_load_layout
        from lalamo.utils.parameter_path import ParameterPath

        template = self.template() if template is None else template
        weights = self.model_weights()
        layout = resolve_decoder_load_layout(weights)
        with jax.set_mesh(template.sharding_config.mesh):
            output_norm = load_rmsnorm(
                template.transformer.output_norm,
                weights,
                layout.decoder_path / layout.norm_key,
            )
            readout = load_linear(template.readout, weights, ParameterPath("readout"))
        return output_norm, readout

    def matrix_weights(self, record: MatrixRecord) -> np.ndarray:
        arrays = [self.array(name) for name in record.source_modules]
        if record.role == "embedding":
            result = arrays[0].T
        elif record.canonical_name.endswith("mixer.qkvg_projection.weights"):
            q_gate, k, v = arrays
            text_config = json.loads((self.directory / "config.json").read_text())["text_config"]
            heads = text_config["num_attention_heads"]
            q_gate = q_gate.reshape(heads, 2, -1, q_gate.shape[-1])
            q = q_gate[:, 0].reshape(-1, q_gate.shape[-1])
            gate = q_gate[:, 1].reshape(-1, q_gate.shape[-1])
            result = np.concatenate((q, k, v, gate), axis=0)
        elif len(arrays) == 1:
            result = arrays[0]
        else:
            result = np.concatenate(arrays, axis=0)
        if result.shape != record.canonical_weights_shape:
            raise ValueError(
                f"Fused matrix shape mismatch for {record.canonical_name}: "
                f"{result.shape} != {record.canonical_weights_shape}"
            )
        return result


class MappedModelWeights(Mapping[str, "jax.Array"]):
    def __init__(self, checkpoint: RawCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.names = {"model." + name: name for name in checkpoint.tensors if name.startswith("language_model.")}
        self.names["readout.weight"] = "readout.weight"
        self.consumed: set[str] = set()

    def __getitem__(self, key: str) -> jax.Array:
        import jax.numpy as jnp

        self.consumed.add(key)
        return jnp.asarray(self.checkpoint.array(self.names[key]))

    def __contains__(self, key: object) -> bool:
        return key in self.names

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


def source_modules_for_matrix(name: str) -> tuple[str, ...]:
    if name == "input_embedding":
        return ("language_model.embed_tokens.weight",)
    if name == "readout.weights":
        return ("readout.weight",)
    pieces = name.split(".")
    if len(pieces) != 6 or pieces[:2] != ["transformer", "layers"] or pieces[-1] != "weights":
        raise ValueError(f"Unrecognized D3 matrix path: {name}")
    layer = int(pieces[2])
    prefix = f"language_model.layers.{layer}."
    suffix = ".".join(pieces[3:5])
    suffixes = {
        "mlp.up_projection": ("mlp.up_proj.weight", "mlp.gate_proj.weight"),
        "mlp.down_projection": ("mlp.down_proj.weight",),
        "mixer.in_proj": (
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
        ),
        "mixer.out_proj": ("linear_attn.out_proj.weight",),
        "mixer.qkvg_projection": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        "mixer.out_projection": ("self_attn.o_proj.weight",),
    }
    return tuple(prefix + item for item in suffixes[suffix])


def matrix_inventory(checkpoint: RawCheckpoint) -> tuple[MatrixRecord, ...]:
    import jax
    import jax.numpy as jnp
    from lalamo.initializer import EmptyInitializer
    from lalamo.utils.parameter_path import ParameterPath
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.weight_matrix import WeightMatrix

    sharding = ShardingConfig.replicated()
    with jax.set_mesh(sharding.mesh):
        template = checkpoint.config().init(EmptyInitializer(jnp.bfloat16, sharding))
    records = []
    for path, matrix in jax.tree_util.tree_leaves_with_path(
        template, is_leaf=lambda node: isinstance(node, WeightMatrix)
    ):
        if not isinstance(matrix, WeightMatrix):
            continue
        name = str(ParameterPath() / path)
        role = "embedding" if name == "input_embedding" else "readout" if name == "readout.weights" else "trunk"
        modules = source_modules_for_matrix(name)
        if not set(modules) <= set(checkpoint.tensors):
            raise ValueError(f"Missing original weights for {name}")
        if len(matrix.shape) != 2:
            raise ValueError(f"Expected an unbatched D3 weight matrix: {name}")
        shape = (matrix.shape[0], matrix.shape[1])
        if role == "embedding":
            shape = (shape[1], shape[0])
        records.append(MatrixRecord(name, shape, modules, role, role != "readout"))
    records.sort(key=lambda record: record.canonical_name)
    if len(records) != 130 or sum(record.role == "trunk" for record in records) != 128:
        raise ValueError("Expected all 128 D3 backbone matrices, embedding, and decision readout")
    if len({name for record in records for name in record.source_modules}) != 250:
        raise ValueError("Expected 248 original backbone projections plus embedding and readout")
    return tuple(records)


def save_inventory(checkpoint: RawCheckpoint, output: Path) -> tuple[MatrixRecord, ...]:
    records = matrix_inventory(checkpoint)
    write_json(
        output,
        {
            "schema": "d3-mirai-m-inventory/1",
            "source_revision": SOURCE_REVISION,
            "source_manifest_sha256": sha256_file(checkpoint.directory / "MODEL_MANIFEST.json"),
            "module_inventory": [asdict(record) for record in records],
            "quantized_matrices": sum(record.quantized for record in records),
            "quantized_weights_count": sum(record.weights_count for record in records if record.quantized),
            "readout_preserved": True,
        },
    )
    return records
