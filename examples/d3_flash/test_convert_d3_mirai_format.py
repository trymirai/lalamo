"""Small CPU contracts for literal, streaming Lalamo INT4/RHT32 conversion."""

from __future__ import annotations

import json
from dataclasses import replace

import lalamo

# isort: split

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lalamo.exportable import Exportable
from lalamo.initializer import RandomInitializer
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig
from lalamo.safetensors import safe_read
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionSpec, Layout, WeightMatrix

from convert_d3_lalamo import D3DecisionModelConfig, DecisionContract
from convert_d3_mirai_format import (
    ASSETS,
    TRACE_COUNTS,
    compress_bf16,
    compress_layer,
    compressor,
    merge_fragments,
    quantization_spec,
    runtime_identity,
    save_fragment,
    sha256_file,
    span_hash,
    tensor_header,
    verify_source,
)


def test_runtime_identity_accepts_namespace_package(monkeypatch):
    monkeypatch.setattr(lalamo, "__file__", None)
    identity = runtime_identity()
    assert len(identity["lalamo_revision"]) == 40
    for name in (
        "lalamo_source_tree_sha256",
        "lalamo_dirty_diff_sha256",
        "decision_adapter_sha256",
    ):
        assert len(identity[name]) == 64
        assert int(identity[name], 16) >= 0
    assert all(value for name, value in identity["versions"].items() if name != "metaljax")
    assert identity["versions"]["metaljax"] is None or isinstance(identity["versions"]["metaljax"], str)


@pytest.fixture(scope="module")
def model():
    sharding = ShardingConfig.replicated()
    with jax.set_mesh(sharding.mesh):
        norm = NormalizationConfig(
            epsilon=1e-6,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        mlp = DenseMLPConfig(LinearConfig(), SiLU(), False, False, None, None)
        gdn = DeltaNetConfig(
            LinearConfig(),
            SeparableCausalConvConfig(has_biases=False),
            LinearConfig(),
            replace(norm, scale_offset=None, upcast_mode=UpcastMode.ONLY_NORMALIZATION),
            num_heads=16,
            num_groups=8,
            head_dim=8,
            value_head_dim=4,
            kernel_size=3,
        )
        attention = AttentionConfig(
            LinearConfig(),
            LinearConfig(),
            norm,
            norm,
            num_heads=4,
            num_groups=2,
            head_dim=16,
            is_causal=False,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=False,
            has_out_biases=False,
            has_gate=True,
        )
        layers = tuple(TransformerLayerConfig(norm, mixer, None, norm, mlp, None) for mixer in (gdn, attention))
        decision = DecisionContract(
            1,
            "test",
            "test",
            "test",
            "test",
            tuple(str(i) for i in range(255)),
            tuple(range(255)),
            1.0,
            "noncausal_full_attention",
            "last",
            None,
            "float32",
        )
        config = D3DecisionModelConfig(TransformerConfig(layers, norm, 64, 64), LinearConfig(), decision, 65)
        result = config.init(RandomInitializer(jnp.bfloat16, sharding, key=jax.random.key(13)))
        head = FullPrecisionSpec(Layout.OUTPUT_INPUT).compress(
            jax.random.normal(jax.random.key(14), (255, 64), dtype=jnp.float32),
            sharding_config=sharding,
        )
        yield eqx.tree_at(lambda m: m.readout.weights, result, head)


def _bytes(array) -> bytes:
    return np.asarray(array).tobytes(order="C")


def test_cached_compressor_matches_literal_lalamo_bf16_oracle(model):
    weights = jax.random.normal(jax.random.key(4), (64, 128), dtype=jnp.bfloat16)
    expected = quantization_spec().compress(
        weights,
        key=jax.random.key(42),
        preconditioner=None,
        sharding_config=model.sharding_config,
    )
    actual = compress_bf16(weights, model.sharding_config)
    expected_arrays, expected_metadata = Exportable.export(expected)
    actual_arrays, actual_metadata = Exportable.export(actual)
    assert actual_metadata == expected_metadata
    assert actual_arrays.keys() == expected_arrays.keys()
    for key in actual_arrays:
        assert _bytes(actual_arrays[key]) == _bytes(expected_arrays[key]), key
    assert actual.quantized.scales.dtype == jnp.bfloat16
    assert compressor() is compressor()
    before = dict(TRACE_COUNTS)
    compress_bf16(weights * jnp.bfloat16(0.5), model.sharding_config)
    assert dict(TRACE_COUNTS) == before


def test_streamed_layers_embedding_and_tail_match_full_model_oracle(model, tmp_path):
    fragments = []
    converted = model
    for index, original in enumerate(model.transformer.layers):
        layer = compress_layer(original)
        converted = eqx.tree_at(lambda m, i=index: m.transformer.layers[i], converted, layer)
        path = tmp_path / f"layer-{index}.safetensors"
        save_fragment(path, layer, f"transformer.layers.{index}")
        fragments.append(path)
        for tree_path, original_matrix in jax.tree_util.tree_leaves_with_path(
            original, is_leaf=lambda value: isinstance(value, WeightMatrix)
        ):
            if not isinstance(original_matrix, WeightMatrix):
                continue
            from convert_d3_mirai_format import _tree_get

            reference = quantization_spec().compress(
                original_matrix.decompress(),
                key=jax.random.key(42),
                sharding_config=model.sharding_config,
            )
            actual = _tree_get(layer, tree_path)
            for key, array in Exportable.export(reference).arrays.items():
                assert _bytes(array) == _bytes(Exportable.export(actual).arrays[key])
    weights = model.input_embedding.decompress()
    full_embedding = quantization_spec(True).compress(
        weights, key=jax.random.key(42), sharding_config=model.sharding_config
    )
    converted = eqx.tree_at(lambda m: m.input_embedding, converted, full_embedding)
    for start in range(0, 65, 17):
        chunk = compress_bf16(weights[:, start : start + 17], model.sharding_config, embedding=True)
        path = tmp_path / f"embedding-{start:03d}.safetensors"
        save_fragment(path, chunk, "input_embedding")
        fragments.append(path)
    for name, value, prefix in (
        ("norm", model.transformer.output_norm, "transformer.output_norm"),
        ("head", model.readout, "readout"),
        ("ropes", model.transformer.ropes, "transformer.ropes"),
    ):
        path = tmp_path / f"{name}.safetensors"
        save_fragment(path, value, prefix)
        fragments.append(path)
    output = tmp_path / "model.safetensors"
    spans = merge_fragments(output, fragments)
    expected = Exportable.export(converted)
    assert set(spans) == set(expected.arrays)
    with output.open("rb") as stream:
        metadata, arrays = safe_read(stream)
        assert {key: json.loads(value) for key, value in metadata.items()} == expected.metadata
        for key, reference in expected.arrays.items():
            assert _bytes(arrays[key]) == _bytes(reference), key
            assert span_hash(spans[key]) == __import__("hashlib").sha256(_bytes(reference)).hexdigest()
    # Source model matrices, source embedding and the FP32 decision readout survive unchanged.
    assert model.input_embedding.spec == FullPrecisionSpec(Layout.INPUT_OUTPUT)
    assert model.readout.weights.dtype == jnp.float32
    assert _bytes(expected.arrays["readout.weights.weights"]) == _bytes(model.readout.weights.weights)


def test_merger_rejects_duplicate_payload_and_changed_shared_rht_signs(model, tmp_path):
    layer = compress_layer(model.transformer.layers[0])
    first = tmp_path / "layer.safetensors"
    save_fragment(first, layer, "transformer.layers.0")
    with pytest.raises(ValueError, match="Duplicate tensor"):
        merge_fragments(tmp_path / "invalid.safetensors", [first, first])
    embedding = compress_bf16(
        model.input_embedding.decompress()[:, :17],
        model.sharding_config,
        embedding=True,
    )
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_fragment(a, embedding, "input_embedding")
    modified = eqx.tree_at(
        lambda e: e.incoherence_signs.output_signs,
        embedding,
        -embedding.incoherence_signs.output_signs,
    )
    save_fragment(b, modified, "input_embedding")
    with pytest.raises(ValueError, match="signs changed"):
        merge_fragments(tmp_path / "bad-signs.safetensors", [a, b])


def test_rejects_fp32_quantization_source_and_truncated_fragment(model, tmp_path):
    with pytest.raises(ValueError, match="original BF16"):
        compress_bf16(jnp.zeros((32, 32), dtype=jnp.float32), model.sharding_config)
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="Truncated"):
        tensor_header(path)


def test_source_hash_cache_rechecks_modified_shard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        *ASSETS,
        "config.json",
        "model.safetensors.index.json",
        "shard.safetensors",
    ):
        (source / name).write_text("asset")
    decision = {"codes": ["A"], "temperature": 1.0}
    (source / "decision_config.json").write_text(json.dumps(decision))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
    manifest = {
        "repo_id": "vllm-sr/d3-flash",
        "files_sha256": {p.name: sha256_file(p) for p in source.iterdir()},
        "identity": {
            "files_sha256": {"shard.safetensors": sha256_file(source / "shard.safetensors")},
            "decision": decision,
        },
    }
    (source / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    digest = sha256_file(source / "MODEL_MANIFEST.json")
    cache_path = tmp_path / "checks.json"
    first = verify_source(source, expected_manifest=digest, cache_path=cache_path)
    assert verify_source(source, expected_manifest=digest, cache_path=cache_path) == first
    checks = json.loads(cache_path.read_text())["files"]
    assert checks["shard.safetensors"]["cache_hit"] is True
    (source / "shard.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="source file hash mismatch"):
        verify_source(source, expected_manifest=digest, cache_path=cache_path)


@pytest.mark.parametrize("name", ["config.json", "d3_format.py", "model.safetensors.index.json"])
def test_source_rejects_changed_config_formatter_or_index(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    for asset in (
        *ASSETS,
        "config.json",
        "model.safetensors.index.json",
        "shard.safetensors",
    ):
        (source / asset).write_text("asset")
    decision = {"codes": ["A"]}
    (source / "decision_config.json").write_text(json.dumps(decision))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
    files = {p.name: sha256_file(p) for p in source.iterdir()}
    manifest = {
        "repo_id": "vllm-sr/d3-flash",
        "files_sha256": files,
        "identity": {
            "files_sha256": {"shard.safetensors": files["shard.safetensors"]},
            "decision": decision,
        },
    }
    (source / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    digest = sha256_file(source / "MODEL_MANIFEST.json")
    with (source / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="source file hash mismatch"):
        verify_source(source, expected_manifest=digest)
