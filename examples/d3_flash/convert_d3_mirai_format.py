"""Stream original D3 BF16 into Lalamo INT4/RHT32 format without Fisher or QAD."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cattrs
import numpy as np
from frozendict import frozendict

from d3_mirai_m_common import (
    REPOSITORY_ROOT,
    SOURCE_REVISION,
    RawCheckpoint,
    sha256_file,
    write_json,
)

if TYPE_CHECKING:
    from jax import Array
    from lalamo.compressed.hybrid import HybridMatrix, HybridSpec
    from lalamo.modules.transformer_layer import TransformerLayer
    from lalamo.utils.sharding import ShardingConfig

SOURCE_MANIFEST_SHA256 = "41f1572c1753f15b7e2857ff8f56804a49b14e4a90910fcd172b47ffdd22a9b3"
BUFFER_BYTES = 16 * 1024 * 1024
ASSETS = (
    "chat_template.jinja",
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
CONCAT_KEYS = frozenset("input_embedding.quantized." + name for name in ("weights", "scales", "zero_points"))
REPLICATED_KEYS = frozenset({"input_embedding.incoherence_signs.output_signs"})
DTYPE_BYTES = {
    "U8": 1,
    "I8": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
}
TRACE_COUNTS: Counter[str] = Counter()


@dataclass(frozen=True)
class StoredTensor:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class TensorSpan:
    path: Path
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Fragment:
    path: str
    sha256: str


def tensor_header(path: Path) -> tuple[dict[str, str], dict[str, TensorSpan]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > BUFFER_BYTES or size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header size: {path}")
        raw = json.loads(stream.read(size))
    metadata = raw.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise ValueError("Safetensors metadata must contain string values")
    spans = {}
    previous = 0
    ordered = sorted(raw.items(), key=lambda item: item[1]["data_offsets"][0])
    for key, entry in ordered:
        info = cattrs.structure(entry, StoredTensor)
        start, stop = info.data_offsets
        if (
            info.dtype not in DTYPE_BYTES
            or any(d < 0 for d in info.shape)
            or start != previous
            or stop - start != math.prod(info.shape) * DTYPE_BYTES.get(info.dtype, 0)
        ):
            raise ValueError(f"Invalid tensor span: {key}")
        spans[key] = TensorSpan(path, 8 + size + start, stop - start, info.dtype, info.shape)
        previous = stop
    if 8 + size + previous != path.stat().st_size:
        raise ValueError(f"Safetensors payload size mismatch: {path}")
    return metadata, spans


def span_hash(span: TensorSpan) -> str:
    digest = hashlib.sha256()
    with span.path.open("rb") as stream:
        stream.seek(span.offset)
        remaining = span.size
        while remaining:
            data = stream.read(min(BUFFER_BYTES, remaining))
            if not data:
                raise ValueError("Truncated tensor payload")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def merge_fragments(output: Path, fragments: Sequence[Path]) -> dict[str, TensorSpan]:
    """Merge CPU byte spans; never materialize a complete model array or device model."""
    metadata: dict[str, str] = {}
    groups: dict[str, list[TensorSpan]] = {}
    for fragment in fragments:
        fragment_metadata, tensors = tensor_header(fragment)
        for key, value in fragment_metadata.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Conflicting metadata: {key}")
            metadata[key] = value
        for key, span in tensors.items():
            if key not in groups:
                groups[key] = [span]
            elif key in CONCAT_KEYS:
                first = groups[key][0]
                if span.dtype != first.dtype or span.shape[1:] != first.shape[1:]:
                    raise ValueError(f"Incompatible embedding chunk: {key}")
                groups[key].append(span)
            elif key in REPLICATED_KEYS:
                first = groups[key][0]
                if (span.dtype, span.shape, span_hash(span)) != (
                    first.dtype,
                    first.shape,
                    span_hash(first),
                ):
                    raise ValueError(f"Shared embedding RHT signs changed: {key}")
            else:
                raise ValueError(f"Duplicate tensor: {key}")
    header: dict[str, Any] = {"__metadata__": metadata} if metadata else {}
    cursor = 0
    order = sorted(groups, key=lambda k: (-DTYPE_BYTES[groups[k][0].dtype], k))
    for key in order:
        pieces = groups[key]
        first = pieces[0]
        shape = first.shape
        if key in CONCAT_KEYS:
            shape = (sum(piece.shape[0] for piece in pieces), *shape[1:])
        size = sum(piece.size for piece in pieces)
        if cursor % DTYPE_BYTES[first.dtype]:
            raise ValueError("Tensor alignment mismatch")
        header[key] = {
            "dtype": first.dtype,
            "shape": shape,
            "data_offsets": (cursor, cursor + size),
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", len(encoded)) + encoded)
        for key in order:
            for piece in groups[key]:
                with piece.path.open("rb") as source:
                    source.seek(piece.offset)
                    remaining = piece.size
                    while remaining:
                        data = source.read(min(BUFFER_BYTES, remaining))
                        if not data:
                            raise ValueError("Truncated fragment during merge")
                        destination.write(data)
                        remaining -= len(data)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(output)
    return tensor_header(output)[1]


def _file_stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def verify_source(
    source: Path,
    *,
    expected_manifest: str = SOURCE_MANIFEST_SHA256,
    cache_path: Path | None = None,
) -> dict[str, str]:
    manifest_path = source / "MODEL_MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("Pinned original D3 manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != "vllm-sr/d3-flash":
        raise ValueError("Unexpected source model")
    declared = dict(manifest["files_sha256"])
    if any(declared.get(name) != digest for name, digest in manifest["identity"]["files_sha256"].items()):
        raise ValueError("Top-level source hashes contradict decision identity hashes")
    required = {"config.json", "model.safetensors.index.json", *ASSETS}
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    required |= set(index.values())
    if not required <= set(declared):
        raise ValueError("Not every conversion source file has a pinned manifest hash")
    prior = json.loads(cache_path.read_text()) if cache_path is not None and cache_path.exists() else {}
    if prior.get("manifest_sha256") != expected_manifest:
        prior = {}
    checks = {}
    files = {}
    for name in sorted(required):
        if Path(name).name != name:
            raise ValueError("Source manifest contains a nonlocal file name")
        stamp = _file_stamp(source / name)
        cached = prior.get("files", {}).get(name, {})
        hit = cached.get("stamp") == stamp and (name not in declared or cached.get("sha256") == declared[name])
        digest = cached["sha256"] if hit else sha256_file(source / name)
        if name in declared and digest != declared[name]:
            raise ValueError(f"Original source file hash mismatch: {name}")
        files[name] = digest
        checks[name] = {"stamp": stamp, "sha256": digest, "cache_hit": hit}
    files["MODEL_MANIFEST.json"] = expected_manifest
    decision = json.loads((source / "decision_config.json").read_text())
    if decision != manifest["identity"]["decision"]:
        raise ValueError("Original decision contract does not match the source manifest")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if not set(index.values()) <= set(declared):
        raise ValueError("Not every indexed original shard has a declared hash")
    if cache_path is not None:
        write_json(cache_path, {"manifest_sha256": expected_manifest, "files": checks})
    return files


def quantization_spec(embedding: bool = False) -> HybridSpec:
    from lalamo.compressed.hybrid import HybridSpec, IncoherenceProcessingMode
    from lalamo.compressed.int import IntSpec
    from lalamo.weight_matrix import Layout

    return HybridSpec(
        IntSpec(
            bits=4,
            group_size=32,
            is_symmetric=False,
            layout=Layout.INPUT_OUTPUT if embedding else Layout.OUTPUT_INPUT,
        ),
        adapter_spec=None,
        incoherence_block_size=32,
        incoherence_processing_mode=IncoherenceProcessingMode.OUTPUT
        if embedding
        else IncoherenceProcessingMode.INPUT_OUTPUT,
    )


@cache
def compressor(
    embedding: bool = False,
) -> Callable[[Array, ShardingConfig], HybridMatrix]:
    import equinox as eqx
    import jax
    from lalamo.weight_matrix import CompressionImplementation

    spec = quantization_spec(embedding)

    @eqx.filter_jit
    def compress(weights: Array, sharding: ShardingConfig) -> HybridMatrix:
        TRACE_COUNTS[f"{embedding}/{weights.dtype}/{weights.shape}"] += 1
        return spec.compress(
            weights,
            key=jax.random.key(42),
            preconditioner=None,
            implementation=CompressionImplementation.INFERENCE,
            sharding_config=sharding,
        )

    return compress


def compress_bf16(weights: Array, sharding: ShardingConfig, *, embedding: bool = False) -> HybridMatrix:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from lalamo.compressed.int import IntMatrixForInference

    if weights.dtype != jnp.bfloat16:
        raise ValueError("Format conversion requires original BF16 matrix weights")
    sharding = replace(sharding, logical_to_physical=frozendict(sharding.logical_to_physical))
    with jax.set_mesh(sharding.mesh):
        result = compressor(embedding)(weights, sharding)
        jax.block_until_ready(eqx.filter(result, eqx.is_array))
    if not isinstance(result.quantized, IntMatrixForInference):
        raise TypeError("Expected native Lalamo INT4 inference parameters")
    if result.quantized.scales.dtype != jnp.bfloat16:
        raise ValueError("Lalamo INT4 scales are not BF16")
    return result


def compress_layer(layer: TransformerLayer) -> TransformerLayer:
    import equinox as eqx
    import jax
    from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrix

    matrices = jax.tree_util.tree_leaves_with_path(layer, is_leaf=lambda value: isinstance(value, WeightMatrix))
    found = 0
    for path, matrix in matrices:
        if not isinstance(matrix, WeightMatrix):
            continue
        if not isinstance(matrix, FullPrecisionMatrix):
            raise TypeError("Expected original unquantized text matrices")
        quantized = compress_bf16(matrix.decompress(), matrix.sharding_config)
        layer = eqx.tree_at(lambda module, p=path: _tree_get(module, p), layer, quantized)
        found += 1
    if found != 4:
        raise ValueError(f"Expected four fused matrices in a D3 layer, got {found}")
    return layer


def _tree_get(tree: Any, path: Sequence[Any]) -> Any:
    import jax.tree_util as jtu

    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            tree = getattr(tree, key.name)
        elif isinstance(key, jtu.SequenceKey):
            tree = tree[key.idx]
        else:
            raise TypeError("Unexpected D3 parameter path")
    return tree


def save_fragment(path: Path, value: Any, prefix: str) -> Fragment:
    from lalamo.exportable import Exportable
    from lalamo.safetensors import safe_write

    arrays, metadata = value.export() if isinstance(value, Exportable) else Exportable.export(value)
    arrays = {prefix + "." + key: array for key, array in arrays.items()}
    encoded = {prefix + "." + key: json.dumps(item) for key, item in metadata.items()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        safe_write(stream, arrays, encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return Fragment(path.name, sha256_file(path))


def memory_measurement() -> dict[str, Any]:
    import jax

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    statistics = jax.devices()[0].memory_stats()
    return {
        "rss_peak_bytes": rss,
        "device_memory_stats": statistics,
        "jit_traces": dict(TRACE_COUNTS),
    }


def runtime_identity() -> dict[str, Any]:
    from lalamo.compressed import hybrid

    # Lalamo's editable install is a namespace package, so its root has no
    # __file__. Locate the source tree through the concrete implementation.
    module_file = hybrid.__file__
    if module_file is None:
        raise RuntimeError("Cannot locate the native Lalamo Hybrid implementation")
    package = REPOSITORY_ROOT / "lalamo"
    if not Path(module_file).resolve().is_relative_to(package.resolve()):
        raise RuntimeError("Run conversion from the Lalamo source checkout")
    tree = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        tree.update(str(path.relative_to(package)).encode() + b"\0" + sha256_file(path).encode() + b"\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "lalamo"], cwd=REPOSITORY_ROOT)
    versions = {}
    for name in (
        "lalamo",
        "jax",
        "jaxlib",
        "metaljax",
        "equinox",
        "numpy",
        "cattrs",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            if name != "metaljax":
                raise
            versions[name] = None
    return {
        "lalamo_revision": revision,
        "lalamo_source_tree_sha256": tree.hexdigest(),
        "lalamo_dirty_diff_sha256": hashlib.sha256(dirty).hexdigest(),
        "decision_adapter_sha256": sha256_file(Path(__file__).with_name("convert_d3_lalamo.py")),
        "versions": versions,
        "METALJAX_MATMUL_PRECISION": os.environ.get("METALJAX_MATMUL_PRECISION"),
    }


def auxiliary_mapping(checkpoint: RawCheckpoint) -> dict[str, str]:
    mapped = {
        "language_model.norm.weight": "transformer.output_norm.scales",
        "readout.weight": "readout.weights.weights",
    }
    for index in range(32):
        source, target = f"language_model.layers.{index}", f"transformer.layers.{index}"
        mapped[source + ".input_layernorm.weight"] = target + ".pre_mixer_norm.scales"
        mapped[source + ".post_attention_layernorm.weight"] = target + ".pre_mlp_norm.scales"
        if source + ".linear_attn.norm.weight" in checkpoint.tensors:
            for src, dst in (
                ("norm.weight", "norm.scales"),
                ("conv1d.weight", "conv.weights"),
                ("dt_bias", "dt_bias"),
                ("A_log", "a_log"),
            ):
                mapped[source + ".linear_attn." + src] = target + ".mixer." + dst
        else:
            mapped[source + ".self_attn.q_norm.weight"] = target + ".mixer.query_norm.scales"
            mapped[source + ".self_attn.k_norm.weight"] = target + ".mixer.key_norm.scales"
    return mapped


def passthrough_records(checkpoint: RawCheckpoint, spans: dict[str, TensorSpan]) -> list[dict[str, Any]]:
    result = []
    for source_key, output_key in auxiliary_mapping(checkpoint).items():
        if output_key not in spans:
            raise ValueError(f"Missing source passthrough tensor: {output_key}")
        source = checkpoint.array(source_key)
        span = spans[output_key]
        expected = source
        caster = "identity"
        if source_key.endswith("conv1d.weight"):
            expected = np.squeeze(expected, axis=1)
            caster = "squeeze_axis_1"
        if span.dtype != "F32":
            raise ValueError(f"D3 auxiliary/head tensor must remain native FP32: {output_key}")
        if expected.dtype != np.float32:
            expected = expected.astype(np.float32)
            caster += "+exact_bf16_to_fp32"
        digest = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
        if span.shape != expected.shape or span_hash(span) != digest:
            raise ValueError(f"Passthrough differs from original source: {source_key}")
        result.append(
            {
                "source_key": source_key,
                "output_key": output_key,
                "source_shape": list(source.shape),
                "source_dtype": checkpoint.tensors[source_key].dtype,
                "source_sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
                "shape": list(span.shape),
                "dtype": span.dtype,
                "sha256": digest,
                "caster": caster,
            }
        )
    return result


def convert(
    source: Path,
    output: Path,
    *,
    embedding_chunk: int = 4096,
    report: Path | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from d3_mirai_m_common import matrix_inventory

    if embedding_chunk < 1:
        raise ValueError("Embedding vocabulary chunk must be positive")
    source, output = source.resolve(), output.resolve()
    if report is None:
        report = output.with_name(output.name + "-report")
    report = report.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from original source")
    if report == source or source in report.parents:
        raise ValueError("Report must be separate from original source")
    report.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=True)
    progress_path = staging / "PROGRESS.json"
    source_started = time.perf_counter()
    source_files = verify_source(source, cache_path=staging / "SOURCE_CHECKS.json")
    shutil.copy2(staging / "SOURCE_CHECKS.json", report / "source-checks.json")
    print(
        json.dumps(
            {
                "stage": "source_verified",
                "elapsed_seconds": time.perf_counter() - source_started,
            }
        ),
        flush=True,
    )
    checkpoint = RawCheckpoint(source)
    template = checkpoint.template()
    records = matrix_inventory(checkpoint)
    if jax.default_backend() == "metal":
        for record in records:
            width = record.shape[0] if record.role == "embedding" else record.shape[1]
            if record.quantized and width % 64:
                raise ValueError("Metal D3 conversion requires even group32 zero-point packing")
    identity = {
        "source_files": source_files,
        "converter_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(Path(__file__).with_name("d3_mirai_m_common.py")),
        "backend": jax.default_backend(),
        "embedding_chunk": embedding_chunk,
        "lalamo_version": version("lalamo"),
        "jax_version": version("jax"),
        "implementation": runtime_identity(),
    }
    progress: dict[str, Any] = {"identity": identity, "fragments": [], "stages": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Cannot resume conversion with changed source/code/backend/chunk identity")
        for item in progress["fragments"]:
            if sha256_file(staging / item["path"]) != item["sha256"]:
                raise ValueError("Conversion fragment hash mismatch on resume")
    finished = {item["name"] for item in progress["stages"]}

    def commit(name: str, fragment: Fragment, started: float) -> None:
        item = {
            "name": name,
            "elapsed_seconds": time.perf_counter() - started,
            **memory_measurement(),
        }
        progress["fragments"].append(asdict(fragment))
        progress["stages"].append(item)
        write_json(progress_path, progress)
        write_json(report / "progress.json", progress)
        print(json.dumps({"stage": "converted", **item}), flush=True)

    for index in range(32):
        name = f"layer-{index:02d}"
        if name in finished:
            continue
        started = time.perf_counter()
        layer = checkpoint.load_layer(index, template)
        layer = compress_layer(layer)
        fragment = save_fragment(staging / (name + ".safetensors"), layer, f"transformer.layers.{index}")
        del layer
        gc.collect()
        commit(name, fragment, started)
    embedding_source = checkpoint.array("language_model.embed_tokens.weight")
    if embedding_source.dtype != np.dtype(jnp.bfloat16):
        raise ValueError("Original embedding must be BF16")
    for start in range(0, template.config.vocab_size, embedding_chunk):
        stop = min(start + embedding_chunk, template.config.vocab_size)
        name = f"embedding-{start:06d}-{stop:06d}"
        if name in finished:
            continue
        started = time.perf_counter()
        weights = jax.device_put(
            jnp.asarray(embedding_source[start:stop].T),
            template.sharding_config.make_sharding((None, None)),
        )
        matrix = compress_bf16(weights, template.sharding_config, embedding=True)
        fragment = save_fragment(staging / (name + ".safetensors"), matrix, "input_embedding")
        del matrix, weights
        gc.collect()
        commit(name, fragment, started)
    del embedding_source
    if "tail" not in finished:
        started = time.perf_counter()
        norm, head = checkpoint.load_tail(template)
        norm_fragment = save_fragment(staging / "output-norm.safetensors", norm, "transformer.output_norm")
        head_fragment = save_fragment(staging / "native-readout.safetensors", head, "readout")
        ropes_fragment = save_fragment(
            staging / "ropes.safetensors",
            template.transformer.ropes,
            "transformer.ropes",
        )
        progress["fragments"].extend((asdict(norm_fragment), asdict(head_fragment)))
        del norm, head
        gc.collect()
        commit("tail", ropes_fragment, started)
    # Canonical order is essential for concatenated vocabulary chunks on resume.
    fragment_paths = [staging / item["path"] for item in sorted(progress["fragments"], key=lambda item: item["path"])]
    spans = merge_fragments(staging / "model.safetensors", fragment_paths)
    passthrough = passthrough_records(checkpoint, spans)
    projections = {record.canonical_name: list(record.source_modules) for record in records if record.role == "trunk"}
    consumed = (
        {name for names in projections.values() for name in names}
        | {r["source_key"] for r in passthrough}
        | {"language_model.embed_tokens.weight"}
    )
    text = {name for name in checkpoint.tensors if name.startswith("language_model.")} | {"readout.weight"}
    if consumed != text:
        raise ValueError(
            f"Source text coverage mismatch: missing={sorted(text - consumed)}, extra={sorted(consumed - text)}"
        )
    norms = [record for record in passthrough if record["output_key"].endswith(".scales")]
    if len(norms) != 105 or len(projections) != 128 or sum(map(len, projections.values())) != 248:
        raise ValueError("D3 normalization/projection inventory changed")
    write_json(staging / "config.json", template.config.to_json())
    for asset in ASSETS:
        shutil.copy2(source / asset, staging / asset)
    shutil.copy2(source / "config.json", staging / "source_config.json")
    shutil.copy2(source / "MODEL_MANIFEST.json", staging / "SOURCE_MODEL_MANIFEST.json")
    artifact_names = (
        "model.safetensors",
        "config.json",
        "source_config.json",
        "SOURCE_MODEL_MANIFEST.json",
        *ASSETS,
    )
    manifest = {
        "schema_version": "d3-mirai-format-conversion/1",
        "source_model": "vllm-sr/d3-flash",
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": source_files["MODEL_MANIFEST.json"],
        "source_files": source_files,
        "model_config_type": "D3DecisionModelConfig",
        "engine_adapter_required": True,
        "quantization": {
            "bits": 4,
            "group_size": 32,
            "asymmetric": True,
            "scale_dtype": "BF16",
            "zero_points": "packed4",
            "rht_block": 32,
            "linear_mode": "input_output",
            "embedding_mode": "output",
            "identity_factor_used": False,
            "fisher_optimized": False,
            "qad_applied": False,
        },
        "method": "rht32_round_to_nearest",
        "format": "Mirai-M-compatible",
        "official_mirai_m_recipe_complete": False,
        "inference_validated": False,
        "source_projection_count": 248,
        "fused_projection_count": 128,
        "source_projection_map": projections,
        "source_text_tensors_consumed": len(consumed),
        "excluded_vision_tensors": len(set(checkpoint.tensors) - text),
        "normalization_count": 105,
        "source_passthrough": passthrough,
        "exported_arrays": len(spans),
        "converter_sha256": identity["converter_sha256"],
        "runtime_identity": identity,
        "lalamo_revision": identity["implementation"]["lalamo_revision"],
        "measurements": progress["stages"],
        "files": {name: sha256_file(staging / name) for name in artifact_names},
    }
    write_json(staging / "CONVERSION.json", manifest)
    write_json(report / "CONVERSION.json", manifest)
    # Atomic directory publication, using links to our own staging files to
    # avoid another 4GB model copy. Original source files are never hard-linked.
    publishing = output.with_name(output.name + ".publishing")
    publishing.mkdir(exist_ok=True)
    for name in (*artifact_names, "CONVERSION.json"):
        target = publishing / name
        if target.exists():
            if sha256_file(target) != sha256_file(staging / name):
                raise ValueError("Interrupted publication conflicts with the completed bundle")
        else:
            os.link(staging / name, target)
    if {p.name for p in publishing.iterdir()} != {
        *artifact_names,
        "CONVERSION.json",
    }:
        raise ValueError("Unexpected files in conversion publication directory")
    publishing.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("metal", "cpu"), default="metal")
    parser.add_argument("--embedding-chunk", type=int, default=4096)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = args.backend
    os.environ["METALJAX_MATMUL_PRECISION"] = "highest"
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    import jax
    import lalamo  # noqa: F401

    if jax.default_backend() != args.backend:
        raise RuntimeError("Requested conversion backend was not selected")
    print(
        json.dumps(
            convert(
                args.source,
                args.output,
                embedding_chunk=args.embedding_chunk,
                report=args.report,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
