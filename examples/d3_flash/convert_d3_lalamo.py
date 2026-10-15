"""Import the frozen D3 INT4 checkpoint with Lalamo and export its decision model.

The new D3DecisionModelConfig requires a corresponding Uzu engine adapter.
Its vocabulary embedding and 255-row FP32 readout remain separate.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, overload

import cattrs
import jax
import jax.numpy as jnp
import lalamo.safetensors as lalamo_safetensors
import numpy as np
from jaxtyping import Array
from lalamo.exportable import Exportable
from lalamo.initializer import EmptyInitializer, Initializer
from lalamo.model import BaseModel, BaseModelConfig
from lalamo.model_import.loaders.huggingface import (
    load_input_embedding_matrix,
    load_linear,
    load_rmsnorm,
    load_transformer_layer,
    resolve_decoder_load_layout,
)
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import Transformer, TransformerConfig
from lalamo.safetensors import safe_read
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import EmbeddingMatrix


@dataclass(frozen=True)
class DecisionContract:
    format_version: int
    format_id: str
    prompt: str
    base_model: str
    revision: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    attention_mode: Literal["noncausal_full_attention"]
    pooling: Literal["last"]
    max_length: int | None
    readout_dtype: Literal["float32"]


@dataclass(frozen=True)
class ShardIndex:
    weight_map: dict[str, str]


@dataclass(frozen=True)
class SourceQuantization:
    bits: int
    group_size: int
    module_bits: dict[str, int]
    recipe: str


@dataclass(frozen=True)
class SourceConfig:
    quantization_config: SourceQuantization


@dataclass(frozen=True)
class D3DecisionModelConfig(BaseModelConfig):
    transformer_config: TransformerConfig
    readout_config: LinearConfig
    decision: DecisionContract
    vocab_size: int
    activation_dtype: Literal["bfloat16"] = "bfloat16"

    def init(self, initializer: Initializer) -> "D3DecisionModel":
        model_dim = self.transformer_config.model_dim
        readout_initializer = EmptyInitializer(jnp.float32, initializer.sharding_config)
        return D3DecisionModel(
            config=self,
            sharding_config=initializer.sharding_config,
            input_embedding=initializer.embedding_matrix(self.vocab_size, model_dim),
            transformer=self.transformer_config.init(initializer),
            readout=self.readout_config.init(
                readout_initializer,
                input_dim=model_dim,
                output_dims=(len(self.decision.codes),),
                has_biases=False,
            ),
        )

    def init_from_directory(self, directory: Path, initializer: Initializer) -> "D3DecisionModel":
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        return self.init(initializer)


class D3DecisionModel(BaseModel[D3DecisionModelConfig]):
    input_embedding: EmbeddingMatrix
    transformer: Transformer
    readout: Linear


@dataclass(frozen=True)
class TensorReference:
    arrays: Mapping[str, Array]
    key: str


class CheckpointWeights(Mapping[str, Array]):
    def __init__(self, references: Mapping[str, TensorReference]) -> None:
        self.references = references
        self.consumed: set[str] = set()

    def __getitem__(self, key: str) -> Array:
        reference = self.references[key]
        self.consumed.add(key)
        return reference.arrays[reference.key]

    def __contains__(self, key: object) -> bool:
        return key in self.references

    def __iter__(self) -> Iterator[str]:
        return iter(self.references)

    def __len__(self) -> int:
        return len(self.references)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_contract(source: Path) -> DecisionContract:
    contract = cattrs.structure(json.loads((source / "decision_config.json").read_text()), DecisionContract)
    if len(contract.codes) != 255 or len(set(contract.codes)) != 255:
        raise ValueError("Expected the original 255 distinct D3 readout codes")
    if len(contract.token_ids) != len(contract.codes) or contract.temperature <= 0:
        raise ValueError("Invalid D3 code/token mapping or temperature")
    return contract


@overload
def hf_rmsnorm(config: NormalizationConfig) -> NormalizationConfig: ...


@overload
def hf_rmsnorm(config: None) -> None: ...


def hf_rmsnorm(config: NormalizationConfig | None) -> NormalizationConfig | None:
    if config is not None and config.scale_offset == 1.0:
        return replace(config, upcast_mode=UpcastMode.FULL_LAYER)
    return config


def decision_config(source: Path, contract: DecisionContract) -> D3DecisionModelConfig:
    foreign = HFQwen35Config.from_json(source / "config.json")
    if foreign.quantization is not None:
        raise ValueError("This adapter expects HF RMSNorm weights with their original +1 offset")
    decoder = foreign.to_decoder_config({})
    layers = []
    for layer in decoder.transformer_config.layer_configs:
        mixer = layer.mixer_config
        if isinstance(mixer, AttentionConfig):
            mixer = replace(
                mixer,
                is_causal=False,
                query_norm_config=hf_rmsnorm(mixer.query_norm_config),
                key_norm_config=hf_rmsnorm(mixer.key_norm_config),
            )
        layers.append(
            replace(
                layer,
                mixer_config=mixer,
                pre_mixer_norm_config=hf_rmsnorm(layer.pre_mixer_norm_config),
                pre_mlp_norm_config=hf_rmsnorm(layer.pre_mlp_norm_config),
                post_mixer_norm_config=hf_rmsnorm(layer.post_mixer_norm_config),
                post_mlp_norm_config=hf_rmsnorm(layer.post_mlp_norm_config),
            )
        )
    if len(layers) != 32 or sum(isinstance(layer.mixer_config, AttentionConfig) for layer in layers) != 8:
        raise ValueError("Expected the D3 Qwen3.5 backbone with 8 full-attention and 24 DeltaNet layers")
    return D3DecisionModelConfig(
        transformer_config=replace(
            decoder.transformer_config,
            layer_configs=tuple(layers),
            output_norm_config=replace(
                decoder.transformer_config.output_norm_config,
                upcast_mode=UpcastMode.FULL_LAYER,
            ),
        ),
        readout_config=LinearConfig(),
        decision=contract,
        vocab_size=decoder.vocab_size,
    )


def load_model(config: D3DecisionModelConfig, weights: Mapping[str, Array]) -> D3DecisionModel:
    sharding = ShardingConfig.replicated()
    with jax.set_mesh(sharding.mesh):
        template = config.init(EmptyInitializer(jnp.bfloat16, sharding))
        layout = resolve_decoder_load_layout(weights)
        embedding = load_input_embedding_matrix(template.input_embedding, weights, layout.embedding_path)
        layers = []
        for index, layer in enumerate(template.transformer.layers):
            print(f"Lalamo: loading layer {index + 1}/{len(template.transformer.layers)}", flush=True)
            layer_path = layout.decoder_path / "layers" / index
            layers.append(
                load_transformer_layer(
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
                ),
            )
        output_norm = load_rmsnorm(template.transformer.output_norm, weights, layout.decoder_path / layout.norm_key)
        transformer = load_as_at(
            lambda module: (module.layers, module.output_norm),
            template.transformer,
            (tuple(layers), output_norm),
        )
        readout = load_linear(template.readout, weights, ParameterPath("readout"))
        return load_as_at(
            lambda model: (model.input_embedding, model.transformer, model.readout),
            template,
            (embedding, transformer, readout),
        )


def convert(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_config = cattrs.structure(json.loads((source / "config.json").read_text()), SourceConfig)
    quantization = source_config.quantization_config
    if (quantization.bits, quantization.group_size, quantization.recipe) != (4, 64, "uniform"):
        raise ValueError("This conversion imports the existing uniform INT4/group-64 checkpoint")
    if len(quantization.module_bits) != 248 or set(quantization.module_bits.values()) != {4}:
        raise ValueError("Expected all 248 original text linear modules to be INT4")
    contract = read_contract(source)
    config = decision_config(source, contract)
    index = cattrs.structure(json.loads((source / "model.safetensors.index.json").read_text()), ShardIndex)
    source_manifest_sha = digest(source / "MODEL_MANIFEST.json")
    destination = output
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        output = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".d3-lalamo-", dir=output.parent)))
        references: dict[str, TensorReference] = {}
        excluded_vision = 0
        for filename in sorted(set(index.weight_map.values())):
            stream = stack.enter_context((source / filename).open("rb"))
            _, arrays = safe_read(stream)
            for key in arrays:
                if key.startswith("visual."):
                    excluded_vision += 1
                    continue
                if not key.startswith("language_model."):
                    raise ValueError(f"Unexpected checkpoint namespace: {key}")
                mapped = key.removesuffix(".qbiases") + ".biases" if key.endswith(".qbiases") else key
                mapped = "model." + mapped
                if mapped in references:
                    raise ValueError(f"Aliased checkpoint tensor: {mapped}")
                references[mapped] = TensorReference(arrays, key)
        head_stream = stack.enter_context((source / "readout.safetensors").open("rb"))
        _, head_arrays = safe_read(head_stream)
        if set(head_arrays) != {"weight"}:
            raise ValueError("Unexpected D3 readout checkpoint")
        references["readout.weight"] = TensorReference(head_arrays, "weight")
        weights = CheckpointWeights(references)
        model = load_model(config, weights)
        if unused := set(weights) - weights.consumed:
            raise ValueError(f"Unconsumed text tensors: {sorted(unused)}")

        exported = Exportable.export(model)
        head_keys = [key for key in exported.arrays if key.startswith("readout.")]
        if len(head_keys) != 1:
            raise ValueError(f"Unexpected exported readout arrays: {head_keys}")
        original_head = np.asarray(head_arrays["weight"])
        converted_head = np.asarray(exported.arrays[head_keys[0]])
        if converted_head.dtype != np.float32 or not np.array_equal(original_head, converted_head):
            raise ValueError("FP32 readout was changed during conversion")
        if converted_head.shape != (255, 4096):
            raise ValueError("D3 readout shape was changed")

        print("Lalamo: saving decision model", flush=True)
        model.save(output)
        copied = (
            "decision_config.json",
            "readout.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "d3_format.py",
            "d3_runtime.py",
            "LICENSE",
            "NOTICE",
        )
        for filename in copied:
            shutil.copy2(source / filename, output / filename)
        shutil.copy2(source / "config.json", output / "source_config.json")

        if lalamo_safetensors.__file__ is None:
            raise ValueError("Unable to locate the Lalamo checkout")
        lalamo_dir = Path(lalamo_safetensors.__file__).resolve().parent.parent
        lalamo_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lalamo_dir, text=True).strip()
        patch = subprocess.check_output(
            ["git", "diff", "--", "lalamo/model_import/loaders/huggingface.py"],
            cwd=lalamo_dir,
        )
        (output / "lalamo-loader.patch").write_bytes(patch)
        sizes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        record = {
            "created_at": datetime.now(UTC).isoformat(),
            "source": str(source),
            "source_model": "vllm-sr/d3-flash",
            "source_revision": "521525effa79469d899a570726547dd7f00da3af",
            "source_manifest_sha256": source_manifest_sha,
            "lalamo_revision": lalamo_revision,
            "lalamo_version": version("lalamo"),
            "lalamo_loader_patch_sha256": hashlib.sha256(patch).hexdigest(),
            "converter_sha256": digest(Path(__file__)),
            "scope": "text",
            "model_config_type": "D3DecisionModelConfig",
            "engine_adapter_required": True,
            "inference_validated": False,
            "quantization": {"bits": 4, "group_size": 64, "requantized": False},
            "key_mapping": ["language_model.* -> model.language_model.*", "*.qbiases -> *.biases"],
            "source_quantized_modules": len(quantization.module_bits),
            "source_text_tensors_consumed": len(weights.consumed),
            "excluded_vision_tensors": excluded_vision,
            "exported_arrays": len(exported.arrays),
            "exported_dtype_counts": dict(Counter(str(array.dtype) for array in exported.arrays.values())),
            "readout_bitwise_equal": True,
            "model_bytes": (output / "model.safetensors").stat().st_size,
            "total_bytes_before_manifest": sizes,
            "files": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
        }
        (output / "CONVERSION.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        if digest(source / "MODEL_MANIFEST.json") != source_manifest_sha:
            raise ValueError("The source manifest changed during conversion")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        output.rename(destination)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
