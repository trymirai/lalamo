"""Validate the native Lalamo D3 format conversion without loading a model.

This checks source identity, safetensors structure, tensor contracts, and
lossless preservation of non-matrix tensors. It does not assess quantization
quality, runtime behavior, or the Mirai-M recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, overload

import cattrs
import numpy as np

SCHEMA_VERSION = "d3-mirai-format-conversion/1"
MANIFEST_NAME = "CONVERSION.json"
SOURCE_MODEL = "vllm-sr/d3-flash"
SOURCE_REVISION = "521525effa79469d899a570726547dd7f00da3af"
EXPECTED_SOURCE_MANIFEST_SHA256 = "41f1572c1753f15b7e2857ff8f56804a49b14e4a90910fcd172b47ffdd22a9b3"
MODEL_CONFIG_TYPE = "D3DecisionModelConfig"
MAX_HEADER_BYTES = 100_000_000
EXPECTED_LAYERS = 32
EXPECTED_ATTENTION_LAYERS = 8
EXPECTED_PROJECTIONS = 248
EXPECTED_FUSED_MATRICES = 128
EXPECTED_QUANTIZED_MATRICES = 129
EXPECTED_NORMS = 105
EXPECTED_VOCAB = 248_320
EXPECTED_HIDDEN = 4_096
EXPECTED_CODES = 255
EXPECTED_OUTPUT_FILES = {
    "model.safetensors",
    "config.json",
    "source_config.json",
    "SOURCE_MODEL_MANIFEST.json",
    "decision_config.json",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "d3_format.py",
    "d3_runtime.py",
    "LICENSE",
    "NOTICE",
}
OUTPUT_SOURCE_COPIES = (
    ("MODEL_MANIFEST.json", "SOURCE_MODEL_MANIFEST.json"),
    ("config.json", "source_config.json"),
    ("decision_config.json", "decision_config.json"),
    ("readout.safetensors", "readout.safetensors"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("chat_template.jinja", "chat_template.jinja"),
    ("vocab.json", "vocab.json"),
    ("merges.txt", "merges.txt"),
    ("d3_format.py", "d3_format.py"),
    ("d3_runtime.py", "d3_runtime.py"),
    ("LICENSE", "LICENSE"),
    ("NOTICE", "NOTICE"),
)
REQUIRED_SOURCE_FILES = {
    "config.json",
    "model.safetensors.index.json",
    *(source for source, _ in OUTPUT_SOURCE_COPIES if source != "MODEL_MANIFEST.json"),
}

DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _safe_file(root: Path, value: Any, label: str) -> tuple[str, Path]:
    require(isinstance(value, str) and value, f"{label} must be a relative path")
    rel = PurePosixPath(value)
    require(
        not rel.is_absolute() and all(part not in {"", ".", ".."} for part in rel.parts),
        f"Unsafe {label}: {value!r}",
    )
    path = root.joinpath(*rel.parts)
    require(path.is_file() and not path.is_symlink(), f"Missing regular {label}: {value}")
    require(
        path.resolve().is_relative_to(root.resolve()),
        f"{label} escapes its root: {value}",
    )
    return rel.as_posix(), path


def _safe_source_file(root: Path, value: Any, label: str) -> tuple[str, Path]:
    """Resolve a source snapshot path; content hashes authenticate HF blob symlinks."""
    require(isinstance(value, str) and value, f"{label} must be a relative path")
    rel = PurePosixPath(value)
    require(
        not rel.is_absolute() and all(part not in {"", ".", ".."} for part in rel.parts),
        f"Unsafe {label}: {value!r}",
    )
    path = root.joinpath(*rel.parts)
    require(path.is_file(), f"Missing source file: {value}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Cannot resolve {label}: {value}") from exc
    require(resolved.is_file(), f"Source path is not a regular file: {value}")
    return rel.as_posix(), path


def _regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed in conversion bundle: {path}")
        if path.is_file():
            relative = path.relative_to(root)
            runtime_bytecode = (
                len(relative.parts) == 2
                and relative.parts[0] == "__pycache__"
                and relative.suffix == ".pyc"
                and any(relative.name.startswith(f"{module}.") for module in ("d3_format", "d3_runtime"))
            )
            if not runtime_bytecode:
                files.add(relative.as_posix())
    return files


def verify_file_hash_map(root: Path, hashes: Any, label: str, *, source_snapshot: bool = False) -> dict[str, str]:
    require(
        isinstance(hashes, dict) and hashes,
        f"{label} must be a non-empty path/hash object",
    )
    result: dict[str, str] = {}
    safe_path = _safe_source_file if source_snapshot else _safe_file
    for rel, expected in hashes.items():
        safe_rel, path = safe_path(root, rel, f"{label} file")
        require(
            isinstance(expected, str) and len(expected) == 64 and all(char in "0123456789abcdef" for char in expected),
            f"Invalid SHA-256 in {label}: {rel}",
        )
        actual = sha256_file(path)
        require(actual == expected, f"{label} hash mismatch: {safe_rel}")
        result[safe_rel] = actual
    return result


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


@dataclass
class SafeTensorFile:
    path: Path
    data_offset: int
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]

    @classmethod
    def open(cls, path: Path, *, allow_symlink: bool = False) -> SafeTensorFile:
        require(
            path.is_file() and (allow_symlink or not path.is_symlink()),
            f"Missing regular safetensors file: {path}",
        )
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            require(len(raw_length) == 8, f"Truncated safetensors header: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            require(
                0 < header_length <= MAX_HEADER_BYTES,
                f"Invalid safetensors header length: {path}",
            )
            raw_header = stream.read(header_length)
        require(len(raw_header) == header_length, f"Truncated safetensors header: {path}")
        try:
            header = json.loads(raw_header)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensors header: {path}: {exc}") from exc
        require(isinstance(header, dict), f"Safetensors header must be an object: {path}")
        metadata = header.pop("__metadata__", {})
        require(
            isinstance(metadata, dict)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()),
            f"Invalid safetensors metadata: {path}",
        )
        data_offset = 8 + header_length
        tensors: dict[str, TensorInfo] = {}
        for name, entry in header.items():
            require(isinstance(name, str) and name, f"Invalid tensor name in {path}")
            require(isinstance(entry, dict), f"Invalid tensor header entry: {name}")
            dtype, shape, offsets = (
                entry.get("dtype"),
                entry.get("shape"),
                entry.get("data_offsets"),
            )
            require(dtype in DTYPE_SIZE, f"Unsupported safetensors dtype {dtype!r}: {name}")
            require(
                isinstance(shape, list) and all(type(dim) is int and dim >= 0 for dim in shape),
                f"Invalid tensor shape: {name}",
            )
            require(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(offset) is int and offset >= 0 for offset in offsets),
                f"Invalid tensor offsets: {name}",
            )
            start, end = offsets
            require(
                end >= start and end - start == math.prod(shape) * DTYPE_SIZE[dtype],
                f"Invalid tensor byte size: {name}",
            )
            tensors[name] = TensorInfo(name, dtype, tuple(shape), start, end)
        cursor = 0
        for item in sorted(tensors.values(), key=lambda tensor: (tensor.start, tensor.end, tensor.name)):
            require(item.start == cursor, f"Safetensors gap/overlap before {item.name}")
            cursor = item.end
        require(
            data_offset + cursor == path.stat().st_size,
            f"Truncated/trailing safetensors payload: {path}",
        )
        return cls(path, data_offset, tensors, metadata)

    def iter_bytes(self, name: str, chunk_size: int = 8 * 1024 * 1024) -> Iterable[bytes]:
        info = self.tensors.get(name)
        require(info is not None, f"Missing tensor {name} in {self.path}")
        with self.path.open("rb") as stream:
            stream.seek(self.data_offset + info.start)
            remaining = info.end - info.start
            while remaining:
                chunk = stream.read(min(chunk_size, remaining))
                require(bool(chunk), f"Truncated tensor payload: {name}")
                remaining -= len(chunk)
                yield chunk

    def tensor_sha256(self, name: str) -> str:
        digest = hashlib.sha256()
        for chunk in self.iter_bytes(name):
            digest.update(chunk)
        return digest.hexdigest()


def _check_exact_files(root: Path, listed: set[str], manifest_name: str) -> None:
    actual = _regular_files(root)
    expected = listed | {manifest_name}
    require(
        actual == expected,
        f"Unexpected conversion file set: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def _verify_source_copies(output: Path, source_hashes: dict[str, str]) -> None:
    """Require every published original sidecar to preserve its pinned source bytes."""
    for source_name, output_name in OUTPUT_SOURCE_COPIES:
        require(source_name in source_hashes, f"Missing source sidecar hash: {source_name}")
        output_path = output / output_name
        require(output_path.is_file(), f"Missing preserved output sidecar: {output_name}")
        require(
            sha256_file(output_path) == source_hashes[source_name],
            f"Source sidecar changed: {output_name}",
        )


def _text_config(config: Any) -> dict[str, Any]:
    require(isinstance(config, dict), "Source config must be an object")
    if isinstance(config.get("text_config"), dict):
        text = config["text_config"]
    else:
        text = config
    require(
        text.get("model_type") in {"qwen3_5_text", "qwen3_5"},
        "Unexpected D3 Qwen3.5 source model type",
    )
    return text


def _matrix_inventory(
    text: dict[str, Any],
) -> dict[str, tuple[tuple[str, ...], tuple[int, int], str]]:
    hidden = text.get("hidden_size")
    intermediate = text.get("intermediate_size")
    layers = text.get("layer_types")
    vocab = text.get("vocab_size")
    require(
        all(type(value) is int and value > 0 for value in (hidden, intermediate, vocab)),
        "Invalid source model dimensions",
    )
    require(
        isinstance(layers, list) and layers and all(isinstance(item, str) for item in layers),
        "Invalid source layer_types",
    )
    heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads")
    head_dim = text.get("head_dim")
    key_heads = text.get("linear_num_key_heads")
    key_head_dim = text.get("linear_key_head_dim")
    value_heads = text.get("linear_num_value_heads")
    value_head_dim = text.get("linear_value_head_dim")
    require(
        all(
            type(value) is int and value > 0
            for value in (
                heads,
                kv_heads,
                head_dim,
                key_heads,
                key_head_dim,
                value_heads,
                value_head_dim,
            )
        ),
        "Invalid source attention dimensions",
    )
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    key_dim, value_dim = key_heads * key_head_dim, value_heads * value_head_dim
    result: dict[str, tuple[tuple[str, ...], tuple[int, int], str]] = {
        "input_embedding": (
            ("language_model.embed_tokens.weight",),
            (hidden, vocab),
            "embedding",
        ),
    }
    for index, layer_type in enumerate(layers):
        require(
            layer_type in {"linear_attention", "full_attention"},
            f"Unsupported source layer type: {layer_type}",
        )
        src = f"language_model.layers.{index}."
        dst = f"transformer.layers.{index}."
        result[dst + "mlp.up_projection.weights"] = (
            (src + "mlp.up_proj.weight", src + "mlp.gate_proj.weight"),
            (2 * intermediate, hidden),
            "trunk",
        )
        result[dst + "mlp.down_projection.weights"] = (
            (src + "mlp.down_proj.weight",),
            (hidden, intermediate),
            "trunk",
        )
        if layer_type == "full_attention":
            result[dst + "mixer.qkvg_projection.weights"] = (
                (
                    src + "self_attn.q_proj.weight",
                    src + "self_attn.k_proj.weight",
                    src + "self_attn.v_proj.weight",
                ),
                (2 * q_dim + 2 * kv_dim, hidden),
                "trunk",
            )
            result[dst + "mixer.out_projection.weights"] = (
                (src + "self_attn.o_proj.weight",),
                (hidden, q_dim),
                "trunk",
            )
        else:
            result[dst + "mixer.in_proj.weights"] = (
                (
                    src + "linear_attn.in_proj_qkv.weight",
                    src + "linear_attn.in_proj_z.weight",
                    src + "linear_attn.in_proj_b.weight",
                    src + "linear_attn.in_proj_a.weight",
                ),
                (2 * key_dim + 2 * value_dim + 2 * value_heads, hidden),
                "trunk",
            )
            result[dst + "mixer.out_proj.weights"] = (
                (src + "linear_attn.out_proj.weight",),
                (hidden, value_dim),
                "trunk",
            )
    result["readout.weights"] = (
        ("readout.weight",),
        (EXPECTED_CODES, hidden),
        "readout",
    )
    return result


def _source_projection_shapes(text: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    hidden = text["hidden_size"]
    intermediate = text["intermediate_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    value_heads = text["linear_num_value_heads"]
    result = {"language_model.embed_tokens.weight": (text["vocab_size"], hidden)}
    for index, layer_type in enumerate(text["layer_types"]):
        src = f"language_model.layers.{index}."
        result[src + "mlp.up_proj.weight"] = (intermediate, hidden)
        result[src + "mlp.gate_proj.weight"] = (intermediate, hidden)
        result[src + "mlp.down_proj.weight"] = (hidden, intermediate)
        if layer_type == "full_attention":
            result[src + "self_attn.q_proj.weight"] = (2 * q_dim, hidden)
            result[src + "self_attn.k_proj.weight"] = (kv_dim, hidden)
            result[src + "self_attn.v_proj.weight"] = (kv_dim, hidden)
            result[src + "self_attn.o_proj.weight"] = (hidden, q_dim)
        else:
            result[src + "linear_attn.in_proj_qkv.weight"] = (
                2 * key_dim + value_dim,
                hidden,
            )
            result[src + "linear_attn.in_proj_z.weight"] = (value_dim, hidden)
            result[src + "linear_attn.in_proj_b.weight"] = (value_heads, hidden)
            result[src + "linear_attn.in_proj_a.weight"] = (value_heads, hidden)
            result[src + "linear_attn.out_proj.weight"] = (hidden, value_dim)
    return result


def _expected_passthrough(
    text: dict[str, Any],
) -> dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]]:
    hidden = text["hidden_size"]
    layers = text["layer_types"]
    head_dim = text["head_dim"]
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    value_heads = text["linear_num_value_heads"]
    kernel = text["linear_conv_kernel_dim"]
    result: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    for index, layer_type in enumerate(layers):
        src = f"language_model.layers.{index}."
        dst = f"transformer.layers.{index}."
        result[src + "input_layernorm.weight"] = (
            dst + "pre_mixer_norm.scales",
            (hidden,),
            (hidden,),
        )
        result[src + "post_attention_layernorm.weight"] = (
            dst + "pre_mlp_norm.scales",
            (hidden,),
            (hidden,),
        )
        if layer_type == "full_attention":
            result[src + "self_attn.q_norm.weight"] = (
                dst + "mixer.query_norm.scales",
                (head_dim,),
                (head_dim,),
            )
            result[src + "self_attn.k_norm.weight"] = (
                dst + "mixer.key_norm.scales",
                (head_dim,),
                (head_dim,),
            )
        else:
            conv = 2 * key_dim + value_dim
            result[src + "linear_attn.conv1d.weight"] = (
                dst + "mixer.conv.weights",
                (conv, 1, kernel),
                (conv, kernel),
            )
            result[src + "linear_attn.A_log"] = (
                dst + "mixer.a_log",
                (value_heads,),
                (value_heads,),
            )
            result[src + "linear_attn.dt_bias"] = (
                dst + "mixer.dt_bias",
                (value_heads,),
                (value_heads,),
            )
            result[src + "linear_attn.norm.weight"] = (
                dst + "mixer.norm.scales",
                (text["linear_value_head_dim"],),
                (text["linear_value_head_dim"],),
            )
    result["language_model.norm.weight"] = (
        "transformer.output_norm.scales",
        (hidden,),
        (hidden,),
    )
    # Source readout is a separately stored full precision tensor.
    result["readout.weight"] = (
        "readout.weights.weights",
        (EXPECTED_CODES, hidden),
        (EXPECTED_CODES, hidden),
    )
    return result


def _source_index(source: Path, index_path: Path) -> tuple[dict[str, str], dict[str, SafeTensorFile]]:
    index = _json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    require(
        isinstance(weight_map, dict) and weight_map,
        "Invalid source HF safetensors index",
    )
    require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()),
        "Invalid source weight_map entries",
    )
    shard_names = sorted(set(weight_map.values()))
    shards: dict[str, SafeTensorFile] = {}
    actual: dict[str, str] = {}
    for shard_name in shard_names:
        _, shard_path = _safe_source_file(source, shard_name, "source shard")
        shard = SafeTensorFile.open(shard_path, allow_symlink=True)
        shards[shard_name] = shard
        for tensor_name in shard.tensors:
            require(tensor_name not in actual, f"Duplicate source tensor: {tensor_name}")
            actual[tensor_name] = shard_name
    require(actual == weight_map, "Source index does not match shard headers")
    return weight_map, shards


def _source_manifest_hash_maps(
    source_manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    identity = source_manifest.get("identity")
    nested_file_hashes = identity.get("files_sha256") if isinstance(identity, dict) else None
    require(
        isinstance(nested_file_hashes, dict),
        "Source MODEL_MANIFEST lacks identity.files_sha256",
    )
    top_level_file_hashes = source_manifest.get("files_sha256")
    require(
        isinstance(top_level_file_hashes, dict),
        "Source MODEL_MANIFEST lacks top-level files_sha256",
    )
    require(
        set(top_level_file_hashes) >= REQUIRED_SOURCE_FILES,
        "Source MODEL_MANIFEST top-level hashes omit a required source file",
    )
    for rel, expected in nested_file_hashes.items():
        require(
            isinstance(rel, str)
            and isinstance(expected, str)
            and len(expected) == 64
            and all(char in "0123456789abcdef" for char in expected),
            f"Invalid nested source MODEL_MANIFEST file hash: {rel}",
        )
    for rel, expected in top_level_file_hashes.items():
        require(
            isinstance(rel, str)
            and isinstance(expected, str)
            and len(expected) == 64
            and all(char in "0123456789abcdef" for char in expected),
            f"Invalid top-level source MODEL_MANIFEST file hash: {rel}",
        )
    return top_level_file_hashes, nested_file_hashes


def _verify_source_manifest_hashes(source_manifest: dict[str, Any], verified: dict[str, str]) -> set[str]:
    top_level_file_hashes, nested_file_hashes = _source_manifest_hash_maps(source_manifest)
    for rel, expected in nested_file_hashes.items():
        require(
            verified.get(rel) == expected,
            f"Source file hash disagrees with MODEL_MANIFEST identity: {rel}",
        )
    for rel, expected in top_level_file_hashes.items():
        if rel in verified:
            require(
                verified[rel] == expected,
                f"Source file hash disagrees with top-level MODEL_MANIFEST: {rel}",
            )
    return set(nested_file_hashes) | REQUIRED_SOURCE_FILES


def _validate_source_identity(
    source: Path, manifest: dict[str, Any]
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, SafeTensorFile],
    dict[str, Any],
    dict[str, Any],
]:
    require(manifest.get("source_model") == SOURCE_MODEL, "Unexpected source model id")
    require(manifest.get("source_revision") == SOURCE_REVISION, "Unexpected source revision")
    source_files = manifest.get("source_files")
    require(
        isinstance(source_files, dict),
        "CONVERSION.source_files must map source paths to hashes",
    )
    _, source_manifest_path = _safe_source_file(source, "MODEL_MANIFEST.json", "source model manifest")
    source_manifest_sha = sha256_file(source_manifest_path)
    require(
        source_manifest_sha == EXPECTED_SOURCE_MANIFEST_SHA256,
        "Original D3 source manifest is not the pinned manifest",
    )
    require(
        manifest.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
        "CONVERSION source manifest hash is not pinned",
    )
    source_manifest = _json(source_manifest_path)
    top_level_hashes, nested_hashes = _source_manifest_hash_maps(source_manifest)
    expected_source_hash_paths = set(nested_hashes) | REQUIRED_SOURCE_FILES | {"MODEL_MANIFEST.json"}
    require(
        set(source_files) == expected_source_hash_paths,
        "CONVERSION.source_files does not match the pinned source hash contract",
    )
    require(
        set(source_files) <= set(top_level_hashes) | {"MODEL_MANIFEST.json"},
        "CONVERSION.source_files contains a path absent from the pinned top-level manifest",
    )
    verified = verify_file_hash_map(source, source_files, "source_files", source_snapshot=True)
    required = REQUIRED_SOURCE_FILES | {"MODEL_MANIFEST.json"}
    require(
        required <= set(verified),
        f"Missing required source files: {sorted(required - set(verified))}",
    )
    require(
        verified["MODEL_MANIFEST.json"] == EXPECTED_SOURCE_MANIFEST_SHA256,
        "Source manifest hash does not match pinned source manifest",
    )
    require(
        source_manifest.get("repo_id") == SOURCE_MODEL,
        "Source MODEL_MANIFEST repo_id mismatch",
    )
    required_source_paths = (
        _verify_source_manifest_hashes(source_manifest, verified) | required | {"MODEL_MANIFEST.json"}
    )
    require(
        required_source_paths <= set(verified),
        "CONVERSION.source_files does not verify every pinned source manifest file",
    )
    weight_map, shards = _source_index(source, source / "model.safetensors.index.json")
    require(
        set(shards) <= set(verified),
        "CONVERSION.source_files omits one or more HF shards",
    )
    head = SafeTensorFile.open(source / "readout.safetensors", allow_symlink=True)
    require(set(head.tensors) == {"weight"}, "Unexpected source readout tensor schema")
    shards["readout.safetensors"] = head
    source_config = _json(source / "config.json")
    decision = _json(source / "decision_config.json")
    require(isinstance(decision, dict), "Source decision_config must be an object")
    require(
        decision.get("format_version") == 1 and decision.get("format_id") == "d3-code-readout-v1",
        "Unexpected D3 decision contract",
    )
    require(
        decision.get("prompt") == "d3" and decision.get("attention_mode") == "noncausal_full_attention",
        "Unexpected D3 decision prompt/attention contract",
    )
    require(
        decision.get("pooling") == "last" and decision.get("readout_dtype") == "float32",
        "Unexpected D3 pooling/readout contract",
    )
    codes, token_ids = decision.get("codes"), decision.get("token_ids")
    require(
        isinstance(codes, list)
        and len(codes) == EXPECTED_CODES
        and len(set(codes)) == EXPECTED_CODES
        and all(isinstance(code, str) for code in codes),
        "Source D3 code list must contain 255 distinct strings",
    )
    require(
        isinstance(token_ids, list)
        and len(token_ids) == EXPECTED_CODES
        and all(
            type(token_id) is int and 0 <= token_id < _text_config(source_config).get("vocab_size", 0)
            for token_id in token_ids
        ),
        "Source D3 token ids must be 255 integers within the vocabulary",
    )
    require(
        decision.get("base_model") == "Qwen/Qwen3.5-9B"
        and decision.get("revision") == "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "Unexpected D3 base-model identity",
    )
    identity = source_manifest.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("decision"), dict):
        require(
            identity["decision"] == decision,
            "Source MODEL_MANIFEST decision identity mismatch",
        )
    return verified, weight_map, shards, source_config, decision


def _tensor_dtype_for(shards: dict[str, SafeTensorFile], weight_map: dict[str, str], key: str) -> TensorInfo:
    require(key in weight_map, f"Missing source tensor {key}")
    return shards[weight_map[key]].tensors[key]


def _array_sha256(archive: SafeTensorFile, key: str) -> str:
    return archive.tensor_sha256(key)


def _expected_quantization_policy(manifest: dict[str, Any]) -> None:
    quant = manifest.get("quantization")
    require(isinstance(quant, dict), "CONVERSION.quantization must be an object")
    require(
        quant.get("bits") == 4 and quant.get("group_size") == 32,
        "Expected asymmetric INT4/group32 quantization",
    )
    require(quant.get("asymmetric") is True, "Expected asymmetric INT4")
    require(
        str(quant.get("scale_dtype", "")).lower() in {"bf16", "bfloat16"},
        "Expected BF16 quantization scales",
    )
    require(quant.get("zero_points") == "packed4", "Expected packed 4-bit zero points")
    require(quant.get("rht_block") == 32, "Expected RHT32")
    for flag in ("identity_factor_used", "fisher_optimized", "qad_applied"):
        require(
            quant.get(flag) is False,
            f"Format-only conversion must declare {flag}=false",
        )
    require(
        quant.get("linear_mode") == "input_output",
        "Linear RHT processing mode must be input_output",
    )
    require(
        quant.get("embedding_mode") == "output",
        "Embedding RHT processing mode must be output",
    )
    require(
        manifest.get("method") == "rht32_round_to_nearest",
        "Unexpected format-only quantization method",
    )
    require(
        manifest.get("official_mirai_m_recipe_complete") is False and manifest.get("inference_validated") is False,
        "Format-only manifest cannot claim full-M or runtime validation",
    )


def _parse_metadata(archive: SafeTensorFile, key: str) -> dict[str, Any]:
    raw = archive.metadata.get(key)
    require(isinstance(raw, str), f"Missing native matrix spec metadata: {key}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid matrix spec metadata {key}: {exc}") from exc
    require(isinstance(parsed, dict), f"Invalid matrix spec object: {key}")
    return parsed


def _spec_contract(archive: SafeTensorFile, name: str, mode: str, layout: str) -> None:
    outer = _parse_metadata(archive, name + ".spec")
    inner = _parse_metadata(archive, name + ".quantized.spec")
    expected_inner = {
        "type": "IntSpec",
        "bits": 4,
        "group_size": 32,
        "is_symmetric": False,
        "layout": layout,
    }
    require(inner == expected_inner, f"Wrong nested INT4 spec: {name}")
    expected_outer = {
        "type": "HybridSpec",
        "quantization_spec": expected_inner,
        "adapter_spec": None,
        "incoherence_block_size": 32,
        "incoherence_processing_mode": mode,
    }
    require(outer == expected_outer, f"Wrong Hybrid/RHT spec: {name}")


def _check_shape_dtype(archive: SafeTensorFile, key: str, shape: tuple[int, ...], dtype: str) -> TensorInfo:
    info = archive.tensors.get(key)
    require(info is not None, f"Missing native tensor: {key}")
    require(info.shape == shape, f"{key}: shape {info.shape} != expected {shape}")
    require(info.dtype == dtype, f"{key}: dtype {info.dtype} != expected {dtype}")
    return info


def _check_signs(archive: SafeTensorFile, key: str, size: int) -> None:
    info = _check_shape_dtype(archive, key, (size,), "I32")
    require(info.elements == size, f"Wrong sign count: {key}")
    for chunk in archive.iter_bytes(key):
        require(len(chunk) % 4 == 0, f"Misaligned sign data: {key}")
        values = np.frombuffer(chunk, dtype="<i4")
        require(bool(np.all((values == -1) | (values == 1))), f"Invalid RHT sign in {key}")


def _positive_finite_bf16(bits: np.ndarray) -> np.ndarray:
    return ((bits & 0x8000) == 0) & ((bits & 0x7FFF) != 0) & ((bits & 0x7F80) != 0x7F80)


def _check_bf16_positive(archive: SafeTensorFile, key: str, *, active_columns: int | None = None) -> None:
    info = archive.tensors[key]
    require(info.dtype == "BF16", f"Expected BF16 scales: {key}")
    if active_columns is None:
        for chunk in archive.iter_bytes(key):
            require(len(chunk) % 2 == 0, f"Misaligned BF16 scale data: {key}")
            bits = np.frombuffer(chunk, dtype="<u2")
            require(
                bool(np.all(_positive_finite_bf16(bits))),
                f"Non-positive/non-finite quantization scale: {key}",
            )
        return

    require(
        len(info.shape) == 2 and 0 <= active_columns <= info.shape[1],
        f"Invalid scale active columns: {key}",
    )
    rows, columns = info.shape
    row_bytes = columns * 2
    rows_per_chunk = max(1, (4 * 1024 * 1024) // max(row_bytes, 1))
    with archive.path.open("rb") as stream:
        for row_start in range(0, rows, rows_per_chunk):
            row_count = min(rows_per_chunk, rows - row_start)
            stream.seek(archive.data_offset + info.start + row_start * row_bytes)
            chunk = stream.read(row_count * row_bytes)
            require(
                len(chunk) == row_count * row_bytes,
                f"Truncated BF16 scale data: {key}",
            )
            bits = np.frombuffer(chunk, dtype="<u2").reshape(row_count, columns)
            require(
                bool(np.all(_positive_finite_bf16(bits[:, :active_columns]))),
                f"Non-positive/non-finite quantization scale: {key}",
            )
            require(
                not np.any(bits[:, active_columns:]),
                f"Nonzero BF16 scale padding: {key}",
            )


def _mapped_tensor(archive: SafeTensorFile, key: str, dtype: str) -> np.memmap:
    info = archive.tensors[key]
    return np.memmap(
        archive.path,
        mode="r",
        dtype=dtype,
        offset=archive.data_offset + info.start,
        shape=info.shape,
    )


def _check_zero_padding(archive: SafeTensorFile, key: str, actual_output: int, padded_output: int) -> None:
    info = archive.tensors[key]
    require(
        info.dtype == "U8" and len(info.shape) == 2,
        f"Invalid packed zero point array: {key}",
    )
    packed_columns = info.shape[1]
    require(packed_columns == padded_output // 2, f"Wrong packed zero point shape: {key}")
    active_bytes = (actual_output + 1) // 2
    packed = _mapped_tensor(archive, key, "u1")
    if active_bytes < packed_columns:
        require(
            not np.any(packed[:, active_bytes:]),
            f"Nonzero packed zero point padding: {key}",
        )
    if actual_output % 2:
        require(
            not np.any(packed[:, active_bytes - 1] & 0xF0),
            f"Nonzero packed zero point nibble padding: {key}",
        )


def _check_packed_group_tail(archive: SafeTensorFile, key: str, groups: int) -> None:
    info = archive.tensors[key]
    require(
        info.dtype == "U8" and len(info.shape) == 2,
        f"Invalid packed group array: {key}",
    )
    packed_columns = info.shape[1]
    active_bytes = (groups + 1) // 2
    require(packed_columns == active_bytes, f"Wrong packed group width: {key}")
    if groups % 2:
        packed = _mapped_tensor(archive, key, "u1")
        require(
            not np.any(packed[:, active_bytes - 1] & 0xF0),
            f"Nonzero packed group nibble padding: {key}",
        )


def _validate_quantized_matrix(
    archive: SafeTensorFile, name: str, shape: tuple[int, int], embedding: bool
) -> set[str]:
    output_dim, input_dim = shape
    require(
        output_dim > 0 and input_dim > 0 and input_dim % 32 == 0,
        f"Invalid group32 matrix dimensions: {name}",
    )
    mode, layout = ("output", "input_output") if embedding else ("input_output", "output_input")
    _spec_contract(archive, name, mode, layout)
    storage_rows, storage_cols = (input_dim, output_dim) if embedding else (output_dim, input_dim)
    weights_key = name + ".quantized.weights"
    scales_key = name + ".quantized.scales"
    zeros_key = name + ".quantized.zero_points"
    weights = _check_shape_dtype(archive, weights_key, (storage_rows, storage_cols // 2), "U8")
    scales_key_shape = (
        (storage_rows, storage_cols // 32) if embedding else (input_dim // 32, math.ceil(output_dim / 4) * 4)
    )
    scales = _check_shape_dtype(archive, scales_key, scales_key_shape, "BF16")
    zeros_shape = (
        (storage_rows, math.ceil((storage_cols // 32) / 2))
        if embedding
        else (input_dim // 32, math.ceil(output_dim / 4) * 2)
    )
    zeros = _check_shape_dtype(archive, zeros_key, zeros_shape, "U8")
    require(
        weights.elements * 2 == storage_rows * storage_cols,
        f"INT4 packed weights have unexpected group padding: {name}",
    )
    expected_scale_count = storage_rows * storage_cols // 32 if embedding else (input_dim // 32) * scales_key_shape[1]
    require(scales.elements == expected_scale_count, f"Unexpected scale group count: {name}")
    require(
        zeros.elements * 2 >= scales.elements,
        f"Packed zero point group coverage is short: {name}",
    )
    if embedding:
        _check_bf16_positive(archive, scales_key)
        _check_packed_group_tail(archive, zeros_key, storage_cols // 32)
    else:
        _check_bf16_positive(archive, scales_key, active_columns=output_dim)
        padded_output = math.ceil(output_dim / 4) * 4
        _check_zero_padding(archive, zeros_key, output_dim, padded_output)
        _check_signs(archive, name + ".incoherence_signs.input_signs", input_dim)
    _check_signs(archive, name + ".incoherence_signs.output_signs", output_dim)
    return {
        weights_key,
        scales_key,
        zeros_key,
        name + ".incoherence_signs.output_signs",
    } | (set() if embedding else {name + ".incoherence_signs.input_signs"})


def _bf16_as_f32(chunks: Iterable[bytes]) -> Iterable[bytes]:
    for chunk in chunks:
        require(len(chunk) % 2 == 0, "Misaligned BF16 tensor bytes")
        bits = np.frombuffer(chunk, dtype="<u2").astype("<u4") << 16
        yield bits.tobytes()


def _compare_passthrough(
    source_archive: SafeTensorFile,
    source_key: str,
    output: SafeTensorFile,
    output_key: str,
) -> None:
    src = source_archive.tensors[source_key]
    dst = output.tensors[output_key]
    require(dst.dtype == "F32", f"Preserved tensor must be F32: {output_key}")
    if source_key.endswith("conv1d.weight"):
        require(
            len(src.shape) == 3 and src.shape[1] == 1,
            f"Invalid source convolution shape: {source_key}",
        )
        require(
            (src.shape[0], src.shape[2]) == dst.shape,
            f"Convolution shape mismatch: {output_key}",
        )
    else:
        require(
            src.shape == dst.shape,
            f"Preserved source/output shape mismatch: {output_key}",
        )
    src_chunks: Iterable[bytes]
    if src.dtype == "BF16":
        src_chunks = _bf16_as_f32(source_archive.iter_bytes(source_key, chunk_size=4 * 1024 * 1024))
    elif src.dtype == "F32":
        src_chunks = source_archive.iter_bytes(source_key)
    else:
        raise ValueError(f"Unexpected source dtype for preserved tensor: {source_key} ({src.dtype})")
    source_iter, output_iter = iter(src_chunks), iter(output.iter_bytes(output_key))
    while True:
        try:
            expected = next(source_iter)
        except StopIteration:
            try:
                next(output_iter)
            except StopIteration:
                break
            raise ValueError(f"Output has extra bytes: {output_key}") from None
        try:
            actual = next(output_iter)
        except StopIteration as exc:
            raise ValueError(f"Output truncated while comparing preserved tensor: {output_key}") from exc
        require(
            actual == expected,
            f"Preserved tensor values differ after BF16/F32 conversion: {output_key}",
        )


def _verify_output_config(
    output: Path,
    decision: dict[str, Any],
    production: bool = True,
) -> tuple[dict[str, Any], int, int]:
    config = _json(output / "config.json")
    require(config.get("type") == MODEL_CONFIG_TYPE, "Unexpected native Lalamo config type")
    require(
        config.get("decision") == decision,
        "D3 decision contract changed during format conversion",
    )
    require(
        config.get("activation_dtype") == "bfloat16",
        "Expected BF16 model activation dtype",
    )
    transformer = config.get("transformer_config")
    require(isinstance(transformer, dict), "Missing transformer_config")
    layers = transformer.get("layer_configs")
    require(isinstance(layers, list), "Missing layer_configs")
    if production:
        require(config.get("vocab_size") == EXPECTED_VOCAB, "Wrong D3 vocabulary size")
        require(transformer.get("model_dim") == EXPECTED_HIDDEN, "Wrong D3 hidden dimension")
        require(len(layers) == EXPECTED_LAYERS, "Expected 32 D3 layers")
        require(
            len(decision.get("codes", [])) == EXPECTED_CODES,
            "Expected 255 D3 readout codes",
        )
    attention_count = 0
    norms: list[dict[str, Any]] = []
    output_norm = transformer.get("output_norm_config")
    if isinstance(output_norm, dict):
        norms.append(output_norm)
    for layer in layers:
        require(isinstance(layer, dict), "Invalid layer config")
        mixer = layer.get("mixer_config")
        require(isinstance(mixer, dict), "Missing mixer_config")
        if mixer.get("type") == "AttentionConfig":
            attention_count += 1
            require(mixer.get("is_causal") is False, "D3 attention must remain noncausal")
            for field in ("query_norm_config", "key_norm_config"):
                value = mixer.get(field)
                if isinstance(value, dict):
                    norms.append(value)
        else:
            require(mixer.get("type") == "DeltaNetConfig", "Unexpected D3 mixer type")
            value = mixer.get("norm_config")
            if isinstance(value, dict):
                norms.append(value)
        for field in (
            "pre_mixer_norm_config",
            "pre_mlp_norm_config",
            "post_mixer_norm_config",
            "post_mlp_norm_config",
        ):
            value = layer.get(field)
            if isinstance(value, dict):
                norms.append(value)
    ordinary = [norm for norm in norms if norm.get("scale_offset") == 1.0 and norm.get("upcast_mode") == "full_layer"]
    gated = [
        norm for norm in norms if norm.get("scale_offset") is None and norm.get("upcast_mode") == "only_normalization"
    ]
    require(
        len(ordinary) == 81 and len(gated) == 24 and len(norms) == EXPECTED_NORMS,
        "D3 norm precision/preservation config mismatch",
    )
    for norm in ordinary:
        require(
            norm.get("has_scale") is True and norm.get("has_biases") is False and norm.get("subtract_mean") is False,
            "Invalid ordinary RMSNorm contract",
        )
    for norm in gated:
        require(
            norm.get("has_scale") is True and norm.get("has_biases") is False and norm.get("subtract_mean") is False,
            "Invalid gated RMSNorm contract",
        )
    if production:
        require(
            attention_count == EXPECTED_ATTENTION_LAYERS,
            "Expected 8 full-attention and 24 DeltaNet layers",
        )
    return config, attention_count, len(norms)


def _verify_typed_output_config(source: Path, output: Path) -> None:
    """Independently rebuild HF config expectations and compare the typed export."""
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
    from lalamo.modules.linear import LinearConfig
    from lalamo.modules.token_mixers.attention import AttentionConfig
    from lalamo.modules.token_mixers.deltanet import DeltaNetConfig

    from convert_d3_lalamo import D3DecisionModelConfig, DecisionContract

    raw = _json(output / "config.json")
    config = D3DecisionModelConfig.from_json(raw)
    require(config.to_json() == raw, "Typed D3 config round-trip changed serialized fields")
    contract = cattrs.structure(_json(source / "decision_config.json"), DecisionContract)
    require(config.decision == contract, "Decision codes/token_ids/formatter contract changed")
    require(
        len(contract.codes) == len(set(contract.codes)) == len(contract.token_ids) == EXPECTED_CODES,
        "Wrong 255-code mapping",
    )
    require(
        contract.readout_dtype == "float32" and contract.pooling == "last",
        "Wrong decision pooling/readout contract",
    )
    require(
        contract.attention_mode == "noncausal_full_attention",
        "Wrong decision attention contract",
    )
    foreign = HFQwen35Config.from_json(source / "config.json")
    require(foreign.quantization is None, "HF RMSNorm raw +1 convention must remain in effect")
    _verify_transformer_config(config.transformer_config, foreign)
    full = [
        layer.mixer_config
        for layer in config.transformer_config.layer_configs
        if isinstance(layer.mixer_config, AttentionConfig)
    ]
    delta = [
        layer.mixer_config
        for layer in config.transformer_config.layer_configs
        if isinstance(layer.mixer_config, DeltaNetConfig)
    ]
    require(
        len(full) == EXPECTED_ATTENTION_LAYERS
        and len(delta) == EXPECTED_LAYERS - EXPECTED_ATTENTION_LAYERS
        and len(config.transformer_config.layer_configs) == EXPECTED_LAYERS,
        "Wrong full-attention/DeltaNet layer counts",
    )
    require(all(not layer.is_causal for layer in full), "Causal full-attention layer remains")
    require(
        config.vocab_size == foreign.vocab_size and config.activation_dtype == "bfloat16",
        "Wrong vocabulary/activation dtype",
    )
    require(config.readout_config == LinearConfig(), "Unexpected readout configuration")


def _expected_transformer_config(foreign: Any) -> Any:
    from lalamo.modules.normalization import NormalizationConfig, UpcastMode
    from lalamo.modules.token_mixers.attention import AttentionConfig

    original = foreign.to_decoder_config({}).transformer_config

    @overload
    def ordinary_norm(config: NormalizationConfig) -> NormalizationConfig: ...

    @overload
    def ordinary_norm(config: None) -> None: ...

    def ordinary_norm(
        config: NormalizationConfig | None,
    ) -> NormalizationConfig | None:
        if config is None:
            return None
        require(config.scale_offset == 1.0, "Expected raw HF ordinary RMSNorm +1 weights")
        return replace(config, upcast_mode=UpcastMode.FULL_LAYER)

    layers = []
    for layer in original.layer_configs:
        mixer = layer.mixer_config
        if isinstance(mixer, AttentionConfig):
            mixer = replace(
                mixer,
                is_causal=False,
                query_norm_config=ordinary_norm(mixer.query_norm_config),
                key_norm_config=ordinary_norm(mixer.key_norm_config),
            )
        layers.append(
            replace(
                layer,
                mixer_config=mixer,
                pre_mixer_norm_config=ordinary_norm(layer.pre_mixer_norm_config),
                pre_mlp_norm_config=ordinary_norm(layer.pre_mlp_norm_config),
                post_mixer_norm_config=ordinary_norm(layer.post_mixer_norm_config),
                post_mlp_norm_config=ordinary_norm(layer.post_mlp_norm_config),
            )
        )
    require(
        original.output_norm_config.scale_offset == 1.0,
        "Expected raw HF output RMSNorm +1 weights",
    )
    return replace(
        original,
        layer_configs=tuple(layers),
        output_norm_config=replace(original.output_norm_config, upcast_mode=UpcastMode.FULL_LAYER),
    )


def _verify_transformer_config(actual: Any, foreign: Any) -> None:
    from lalamo.modules.normalization import UpcastMode
    from lalamo.modules.token_mixers.attention import AttentionConfig
    from lalamo.modules.token_mixers.deltanet import DeltaNetConfig

    expected = _expected_transformer_config(foreign)
    require(
        actual == expected,
        "Transformer config differs from HF noncausal/FULL_LAYER ordinary RMS "
        "and ONLY_NORMALIZATION gated RMS contract",
    )
    ordinary = [actual.output_norm_config]
    gated = []
    for layer in actual.layer_configs:
        ordinary.extend(
            norm
            for norm in (
                layer.pre_mixer_norm_config,
                layer.pre_mlp_norm_config,
                layer.post_mixer_norm_config,
                layer.post_mlp_norm_config,
            )
            if norm is not None
        )
        if isinstance(layer.mixer_config, AttentionConfig):
            ordinary.extend(
                norm
                for norm in (
                    layer.mixer_config.query_norm_config,
                    layer.mixer_config.key_norm_config,
                )
                if norm is not None
            )
        elif isinstance(layer.mixer_config, DeltaNetConfig):
            gated.append(layer.mixer_config.norm_config)
    require(len(ordinary) == 81 and len(gated) == 24, "Unexpected ordinary/gated RMSNorm counts")
    require(
        all(norm.upcast_mode == UpcastMode.FULL_LAYER and norm.scale_offset == 1.0 for norm in ordinary),
        "HF ordinary RMSNorm precision or +1 changed",
    )
    require(
        all(norm.upcast_mode == UpcastMode.ONLY_NORMALIZATION and norm.scale_offset is None for norm in gated),
        "HF gated RMSNorm precision or direct scale changed",
    )


def verify_format(source: Path, output: Path, *, production: bool = True) -> dict[str, Any]:
    require(
        source.is_dir() and not source.is_symlink(),
        f"Missing source directory: {source}",
    )
    require(
        output.is_dir() and not output.is_symlink(),
        f"Missing output directory: {output}",
    )
    source, output = source.resolve(), output.resolve()
    manifest_path = output / MANIFEST_NAME
    require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"Missing {MANIFEST_NAME}",
    )
    manifest = _json(manifest_path)
    require(isinstance(manifest, dict), "CONVERSION.json must be an object")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "Unsupported format conversion schema",
    )

    output_hashes = verify_file_hash_map(output, manifest.get("files"), "files")
    require(
        set(output_hashes) == EXPECTED_OUTPUT_FILES,
        "CONVERSION.files must list exactly the published format bundle artifacts",
    )
    _check_exact_files(output, set(output_hashes), MANIFEST_NAME)
    source_hashes, weight_map, source_shards, source_config, decision = _validate_source_identity(source, manifest)
    source_text = _text_config(source_config)
    if production:
        require(
            source_text.get("num_hidden_layers") == EXPECTED_LAYERS,
            "Source D3 model must contain 32 layers",
        )
        require(
            source_text.get("hidden_size") == EXPECTED_HIDDEN and source_text.get("vocab_size") == EXPECTED_VOCAB,
            "Source D3 dimensions mismatch",
        )
        require(
            source_text.get("layer_types", []).count("full_attention") == EXPECTED_ATTENTION_LAYERS,
            "Source must contain 8 full-attention layers",
        )
        require(
            len(decision.get("codes", [])) == EXPECTED_CODES,
            "Source decision contract must contain 255 codes",
        )
    require(
        manifest.get("model_config_type") == MODEL_CONFIG_TYPE,
        "Wrong model_config_type declaration",
    )
    require(
        manifest.get("engine_adapter_required") is True,
        "D3 native config requires its engine adapter",
    )
    _expected_quantization_policy(manifest)

    inventory = _matrix_inventory(source_text)
    projections = {name: value for name, value in inventory.items() if value[2] == "trunk"}
    require(
        len(projections) == EXPECTED_FUSED_MATRICES
        and sum(len(value[0]) for value in projections.values()) == EXPECTED_PROJECTIONS,
        "Source matrix inventory mismatch",
    )
    source_projection_map = {name: list(value[0]) for name, value in projections.items()}
    require(
        manifest.get("source_projection_count") == EXPECTED_PROJECTIONS
        and manifest.get("fused_projection_count") == EXPECTED_FUSED_MATRICES,
        "Conversion projection counts do not match source architecture",
    )
    require(
        manifest.get("source_projection_map") == source_projection_map,
        "Conversion projection map does not match source architecture",
    )

    text_sources = {key for key in weight_map if key.startswith("language_model.")}
    vision_sources = {key for key in weight_map if key.startswith("visual.")}
    require(
        len(vision_sources) == 333 and len(text_sources) == 426,
        "Unexpected source HF text/vision tensor inventory",
    )
    input_key = "language_model.embed_tokens.weight"
    projection_keys = {key for value in projections.values() for key in value[0]}
    expected_nonmatrix = _expected_passthrough(source_text)
    expected_source_nonmatrix = set(expected_nonmatrix) - {"readout.weight"}
    require(
        text_sources == projection_keys | {input_key} | expected_source_nonmatrix,
        "Source text tensor coverage mismatch",
    )
    projection_shapes = _source_projection_shapes(source_text)
    require(
        set(projection_shapes) == projection_keys | {input_key},
        "Source projection shape inventory mismatch",
    )
    for tensor_name, expected_shape in projection_shapes.items():
        info = _tensor_dtype_for(source_shards, weight_map, tensor_name)
        require(info.shape == expected_shape, f"Source matrix shape mismatch: {tensor_name}")
        require(info.dtype == "BF16", f"Original source matrix must be BF16: {tensor_name}")
    passthrough_records = manifest.get("source_passthrough")
    require(isinstance(passthrough_records, list), "Missing source_passthrough records")
    passthrough_by_source: dict[str, dict[str, Any]] = {}
    passthrough_outputs: set[str] = set()
    for record in passthrough_records:
        require(isinstance(record, dict), "Invalid source_passthrough record")
        source_tensor = record.get("source_key")
        output_tensor = record.get("output_key")
        require(
            isinstance(source_tensor, str)
            and source_tensor not in passthrough_by_source
            and isinstance(output_tensor, str)
            and output_tensor not in passthrough_outputs,
            "Duplicate or invalid passthrough source/output",
        )
        passthrough_by_source[source_tensor] = record
        passthrough_outputs.add(output_tensor)
    require(
        set(passthrough_by_source) == set(expected_nonmatrix),
        "Passthrough map does not cover all preserved source tensors and readout",
    )
    readout = source_shards["readout.safetensors"]
    require(
        readout.tensors["weight"].shape == (EXPECTED_CODES, source_text["hidden_size"]),
        "Source readout shape mismatch",
    )
    require(readout.tensors["weight"].dtype == "F32", "Source decision readout must be F32")

    archive = SafeTensorFile.open(output / "model.safetensors")
    expected_arrays: set[str] = set()
    expected_metadata: set[str] = set()
    module_specs: list[str] = []
    quantized_modules = 0
    for canonical, (_, shape, role) in inventory.items():
        if role == "readout":
            spec_key = canonical + ".spec"
            expected_metadata.add(spec_key)
            spec = _parse_metadata(archive, spec_key)
            require(
                spec == {"type": "FullPrecisionSpec", "layout": "output_input"},
                "Readout must use FullPrecisionSpec",
            )
            info = _check_shape_dtype(archive, canonical + ".weights", shape, "F32")
            require(
                info.shape == (EXPECTED_CODES, EXPECTED_HIDDEN),
                "Readout array must be F32 [255,4096]",
            )
            expected_arrays.add(canonical + ".weights")
            module_specs.append(canonical)
            continue
        embedding = role == "embedding"
        _validate_quantized_matrix(archive, canonical, shape, embedding)
        quantized_modules += 1
        expected_metadata.update({canonical + ".spec", canonical + ".quantized.spec"})
        module_specs.append(canonical)
        expected_arrays.update(
            {
                canonical + ".quantized.weights",
                canonical + ".quantized.scales",
                canonical + ".quantized.zero_points",
                canonical + ".incoherence_signs.output_signs",
            }
        )
        if not embedding:
            expected_arrays.add(canonical + ".incoherence_signs.input_signs")
    require(
        quantized_modules == EXPECTED_QUANTIZED_MATRICES,
        "Expected 129 quantized matrices including embedding",
    )
    for source_tensor, (
        output_tensor,
        source_shape,
        output_shape,
    ) in expected_nonmatrix.items():
        record = passthrough_by_source[source_tensor]
        if source_tensor == "readout.weight":
            source_archive, source_key = readout, "weight"
        else:
            source_archive = source_shards[weight_map[source_tensor]]
            source_key = source_tensor
        require(
            source_archive.tensors[source_key].shape == source_shape,
            f"Source shape mismatch: {source_tensor}",
        )
        source_info = source_archive.tensors[source_key]
        expected_caster = "squeeze_axis_1" if source_tensor.endswith("conv1d.weight") else "identity"
        if source_info.dtype == "BF16":
            expected_caster += "+exact_bf16_to_fp32"
        require(
            record.get("source_shape") == list(source_info.shape)
            and record.get("source_dtype") == source_info.dtype
            and record.get("source_sha256") == source_archive.tensor_sha256(source_key),
            f"Wrong source passthrough proof record: {source_tensor}",
        )
        _check_shape_dtype(archive, output_tensor, output_shape, "F32")
        _compare_passthrough(source_archive, source_key, archive, output_tensor)
        record = passthrough_by_source[source_tensor]
        require(
            record.get("output_key") == output_tensor,
            f"Wrong passthrough output target: {source_tensor}",
        )
        require(
            record.get("shape") == list(output_shape)
            and record.get("dtype") == "F32"
            and record.get("caster") == expected_caster,
            f"Wrong passthrough format record: {source_tensor}",
        )
        require(
            record.get("sha256") == _array_sha256(archive, output_tensor),
            f"Passthrough tensor hash mismatch: {output_tensor}",
        )
        expected_arrays.add(output_tensor)
    require(
        set(archive.tensors) == expected_arrays,
        "Native tensor coverage mismatch: "
        f"missing={sorted(expected_arrays - set(archive.tensors))[:5]}, "
        f"extra={sorted(set(archive.tensors) - expected_arrays)[:5]}",
    )
    require(
        set(archive.metadata) == expected_metadata,
        "Unexpected native matrix metadata entries",
    )

    # Preserve source decision, tokenizer, and model-template sidecars byte-for-byte.
    _verify_source_copies(output, source_hashes)
    _, attention_count, norm_count = _verify_output_config(output, decision, production)
    if production:
        _verify_typed_output_config(source, output)

    require(
        manifest.get("source_text_tensors_consumed") == len(text_sources) + 1,
        "Wrong consumed source tensor count",
    )
    require(
        manifest.get("excluded_vision_tensors") == len(vision_sources),
        "Wrong excluded vision tensor count",
    )
    require(
        manifest.get("exported_arrays") == len(archive.tensors),
        "Wrong exported_arrays count",
    )

    return {
        "format_validation": {
            "status": "passed",
            "schema_version": SCHEMA_VERSION,
            "source_files_checked": len(source_hashes),
            "output_files_checked": len(output_hashes),
            "source_text_tensors": len(text_sources),
            "source_visual_tensors_excluded": len(vision_sources),
            "source_projection_weights": EXPECTED_PROJECTIONS,
            "fused_projection_matrices": EXPECTED_FUSED_MATRICES,
            "quantized_matrices": quantized_modules,
            "preserved_norms": norm_count,
            "attention_layers": attention_count,
            "native_tensors": len(archive.tensors),
            "native_metadata_entries": len(archive.metadata),
            "typed_config_contract": "passed" if production else "not_assessed",
        },
        "quality_evaluation": {"status": "not_assessed"},
        "runtime_validation": {"status": "not_assessed"},
        "full_m_recipe": {"status": "not_assessed"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned original D3 HF snapshot directory")
    parser.add_argument("output", type=Path, help="Lalamo format-conversion output directory")
    args = parser.parse_args()
    print(json.dumps(verify_format(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
