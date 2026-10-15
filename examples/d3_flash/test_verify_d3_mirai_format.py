import hashlib
import json
import struct
from dataclasses import asdict
from typing import Any

import lalamo  # noqa: F401
import pytest
from lalamo.model_import.model_configs.huggingface.qwen3_5 import (
    HFQwen35Config,
    Qwen35RopeParameters,
)

import convert_d3_lalamo
from convert_d3_lalamo import DecisionContract
from verify_d3_mirai_format import (
    EXPECTED_OUTPUT_FILES,
    OUTPUT_SOURCE_COPIES,
    REQUIRED_SOURCE_FILES,
    SafeTensorFile,
    _check_exact_files,
    _compare_passthrough,
    _expected_quantization_policy,
    _matrix_inventory,
    _source_index,
    _source_projection_shapes,
    _validate_quantized_matrix,
    _verify_source_copies,
    _verify_source_manifest_hashes,
    _verify_typed_output_config,
    verify_file_hash_map,
)


def write_safetensors(path, tensors, metadata=None):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    if metadata is not None:
        header["__metadata__"] = {key: json.dumps(value) for key, value in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def hybrid_metadata(name, *, embedding=False):
    layout = "input_output" if embedding else "output_input"
    mode = "output" if embedding else "input_output"
    int_spec = {
        "type": "IntSpec",
        "bits": 4,
        "group_size": 32,
        "is_symmetric": False,
        "layout": layout,
    }
    return {
        name + ".spec": {
            "type": "HybridSpec",
            "quantization_spec": int_spec,
            "adapter_spec": None,
            "incoherence_block_size": 32,
            "incoherence_processing_mode": mode,
        },
        name + ".quantized.spec": int_spec,
    }


def sign_data(size, value=1):
    return struct.pack("<i", value) * size


def bf16_data(size, bits=0x3F80):
    return struct.pack("<H", bits) * size


def make_text_matrix(path, *, output_dim=4, scale_bits=0x3F80, input_sign=1, zero_data=None):
    name = "transformer.layers.0.mlp.up_projection.weights"
    padded = (output_dim + 3) // 4 * 4
    scales_data = struct.pack("<H", scale_bits) * output_dim + bytes((padded - output_dim) * 2)
    zero_bytes = zero_data if zero_data is not None else bytes(padded // 2)
    tensors = {
        name + ".quantized.weights": ("U8", (output_dim, 16), bytes(output_dim * 16)),
        name + ".quantized.scales": ("BF16", (1, padded), scales_data),
        name + ".quantized.zero_points": ("U8", (1, padded // 2), zero_bytes),
        name + ".incoherence_signs.input_signs": (
            "I32",
            (32,),
            sign_data(32, input_sign),
        ),
        name + ".incoherence_signs.output_signs": (
            "I32",
            (output_dim,),
            sign_data(output_dim),
        ),
    }
    write_safetensors(path, tensors, hybrid_metadata(name))
    return SafeTensorFile.open(path), name


def test_safetensors_parser_rejects_payload_gap(tmp_path):
    path = tmp_path / "gap.safetensors"
    header = json.dumps({"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    with pytest.raises(ValueError, match="gap/overlap"):
        SafeTensorFile.open(path)


def test_text_hybrid_int4_group32_contract_passes(tmp_path):
    archive, name = make_text_matrix(tmp_path / "text.safetensors")
    checked = _validate_quantized_matrix(archive, name, (4, 32), embedding=False)
    assert len(checked) == 5


def test_text_contract_rejects_bad_rht_sign(tmp_path):
    archive, name = make_text_matrix(tmp_path / "bad-sign.safetensors", input_sign=0)
    with pytest.raises(ValueError, match="Invalid RHT sign"):
        _validate_quantized_matrix(archive, name, (4, 32), embedding=False)


@pytest.mark.parametrize("scale_bits", [0x0000, 0x8000, 0xBF80, 0x7F80, 0x7FC0])
def test_text_contract_rejects_nonpositive_or_nonfinite_bf16_scales(tmp_path, scale_bits):
    archive, name = make_text_matrix(tmp_path / "bad-scale.safetensors", scale_bits=scale_bits)
    with pytest.raises(ValueError, match="Non-positive/non-finite"):
        _validate_quantized_matrix(archive, name, (4, 32), embedding=False)


def test_text_contract_rejects_nonzero_packed_zero_padding(tmp_path):
    archive, name = make_text_matrix(
        tmp_path / "bad-padding.safetensors",
        output_dim=3,
        zero_data=bytes((0x11, 0x10)),
    )
    with pytest.raises(ValueError, match="nibble padding"):
        _validate_quantized_matrix(archive, name, (3, 32), embedding=False)


def test_embedding_uses_output_only_rht_and_input_output_layout(tmp_path):
    name = "input_embedding"
    tensors = {
        name + ".quantized.weights": ("U8", (64, 16), bytes(64 * 16)),
        name + ".quantized.scales": ("BF16", (64, 1), bf16_data(64)),
        name + ".quantized.zero_points": ("U8", (64, 1), bytes((0x01,)) * 64),
        name + ".incoherence_signs.output_signs": ("I32", (32,), sign_data(32)),
    }
    write_safetensors(
        tmp_path / "embedding.safetensors",
        tensors,
        hybrid_metadata(name, embedding=True),
    )
    archive = SafeTensorFile.open(tmp_path / "embedding.safetensors")
    checked = _validate_quantized_matrix(archive, name, (32, 64), embedding=True)
    assert len(checked) == 4
    assert name + ".incoherence_signs.input_signs" not in archive.tensors


def test_embedding_rejects_text_layout_spec(tmp_path):
    name = "input_embedding"
    tensors = {
        name + ".quantized.weights": ("U8", (64, 16), bytes(64 * 16)),
        name + ".quantized.scales": ("BF16", (64, 1), bf16_data(64)),
        name + ".quantized.zero_points": ("U8", (64, 1), bytes(64)),
        name + ".incoherence_signs.output_signs": ("I32", (32,), sign_data(32)),
    }
    write_safetensors(tmp_path / "wrong-layout.safetensors", tensors, hybrid_metadata(name))
    archive = SafeTensorFile.open(tmp_path / "wrong-layout.safetensors")
    with pytest.raises(ValueError, match="Wrong nested INT4 spec"):
        _validate_quantized_matrix(archive, name, (32, 64), embedding=True)


def test_source_projection_inventory_derives_128_style_fusions_without_model_init():
    text = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": 192,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 32,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
    }
    inventory = _matrix_inventory(text)
    assert len(inventory) == 10
    assert sum(len(source_keys) for source_keys, _, role in inventory.values() if role == "trunk") == 15
    assert inventory["transformer.layers.0.mixer.in_proj.weights"][1] == (132, 64)
    assert inventory["transformer.layers.1.mixer.qkvg_projection.weights"][1] == (
        192,
        64,
    )
    assert _source_projection_shapes(text)["language_model.layers.1.self_attn.q_proj.weight"] == (128, 64)


def test_bf16_source_passthrough_is_compared_after_exact_f32_cast(tmp_path):
    source_path = tmp_path / "source.safetensors"
    output_path = tmp_path / "output.safetensors"
    write_safetensors(source_path, {"norm": ("BF16", (2,), struct.pack("<HH", 0x3F80, 0xC020))})
    write_safetensors(output_path, {"norm.scales": ("F32", (2,), struct.pack("<ff", 1.0, -2.5))})
    _compare_passthrough(
        SafeTensorFile.open(source_path),
        "norm",
        SafeTensorFile.open(output_path),
        "norm.scales",
    )


def test_passthrough_value_mutation_is_detected(tmp_path):
    source_path = tmp_path / "source.safetensors"
    output_path = tmp_path / "output.safetensors"
    write_safetensors(source_path, {"norm": ("BF16", (2,), struct.pack("<HH", 0x3F80, 0xC020))})
    write_safetensors(output_path, {"norm.scales": ("F32", (2,), struct.pack("<ff", 1.0, -2.0))})
    with pytest.raises(ValueError, match="values differ"):
        _compare_passthrough(
            SafeTensorFile.open(source_path),
            "norm",
            SafeTensorFile.open(output_path),
            "norm.scales",
        )


def test_source_snapshot_hash_map_allows_only_explicit_hash_checked_symlink(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"pinned source bytes")
    (source / "weight.bin").symlink_to(blob)
    expected = hashlib.sha256(blob.read_bytes()).hexdigest()

    assert verify_file_hash_map(source, {"weight.bin": expected}, "source_files", source_snapshot=True) == {
        "weight.bin": expected
    }
    with pytest.raises(ValueError, match="regular"):
        verify_file_hash_map(source, {"weight.bin": expected}, "files")


def test_source_index_reads_hash_verified_snapshot_shard_symlinks(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    blob = tmp_path / "blob.safetensors"
    write_safetensors(blob, {"weight": ("U8", (2,), b"\x01\x02")})
    (source / "shard.safetensors").symlink_to(blob)
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))

    weight_map, shards = _source_index(source, index_path)
    assert weight_map == {"weight": "shard.safetensors"}
    assert shards["shard.safetensors"].tensors["weight"].shape == (2,)
    with pytest.raises(ValueError, match="regular safetensors"):
        SafeTensorFile.open(source / "shard.safetensors")


def test_source_manifest_checks_top_level_and_nested_file_hash_maps():
    source_hash = "a" * 64
    verified = dict.fromkeys(REQUIRED_SOURCE_FILES, source_hash)
    verified["index.json"] = source_hash
    manifest = {
        "files_sha256": dict.fromkeys(REQUIRED_SOURCE_FILES, source_hash),
        "identity": {"files_sha256": {"index.json": source_hash}},
    }
    assert _verify_source_manifest_hashes(manifest, verified) == REQUIRED_SOURCE_FILES | {"index.json"}

    manifest["files_sha256"]["config.json"] = "b" * 64
    with pytest.raises(ValueError, match="disagrees with top-level MODEL_MANIFEST"):
        _verify_source_manifest_hashes(manifest, verified)


def test_quantization_policy_matches_actual_format_conversion_manifest():
    manifest = {
        "method": "rht32_round_to_nearest",
        "official_mirai_m_recipe_complete": False,
        "inference_validated": False,
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
    }
    _expected_quantization_policy(manifest)

    for field, invalid in (
        ("rht_block", 64),
        ("zero_points", "packed8"),
        ("linear_mode", "output"),
        ("identity_factor_used", True),
    ):
        broken = json.loads(json.dumps(manifest))
        broken["quantization"][field] = invalid
        with pytest.raises(ValueError):
            _expected_quantization_policy(broken)


def test_chat_template_is_required_and_must_match_original_source_bytes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    source_hashes = {}
    for source_name, output_name in OUTPUT_SOURCE_COPIES:
        payload = f"pinned source content: {source_name}".encode()
        (source / source_name).write_bytes(payload)
        (output / output_name).write_bytes(payload)
        source_hashes[source_name] = hashlib.sha256(payload).hexdigest()

    assert "chat_template.jinja" in REQUIRED_SOURCE_FILES
    assert "chat_template.jinja" in EXPECTED_OUTPUT_FILES
    _verify_source_copies(output, source_hashes)

    (output / "chat_template.jinja").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Source sidecar changed: chat_template\.jinja"):
        _verify_source_copies(output, source_hashes)


def test_typed_config_uses_independent_hf_reconstruction(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    decision = DecisionContract(
        1,
        "test",
        "test",
        "test",
        "test",
        tuple(str(index) for index in range(255)),
        tuple(range(255)),
        1.0,
        "noncausal_full_attention",
        "last",
        None,
        "float32",
    )
    foreign = HFQwen35Config(
        model_type="qwen3_5",
        hidden_size=64,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        vocab_size=512,
        rope_parameters=Qwen35RopeParameters(rope_theta=10000.0, partial_rotary_factor=0.5),
        head_dim=16,
        attention_bias=False,
        layer_types=[
            kind
            for _ in range(8)
            for kind in (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        intermediate_size=128,
    )
    (source / "config.json").write_text(json.dumps(asdict(foreign)), encoding="utf-8")
    (source / "decision_config.json").write_text(json.dumps(asdict(decision)), encoding="utf-8")
    expected = convert_d3_lalamo.decision_config(source, decision)
    monkeypatch.setattr(
        convert_d3_lalamo,
        "decision_config",
        lambda *_args: pytest.fail("verifier must not trust converter config generation"),
    )
    (output / "config.json").write_text(json.dumps(expected.to_json()), encoding="utf-8")

    _verify_typed_output_config(source, output)

    def changed_config() -> dict[str, Any]:
        value = json.loads(json.dumps(expected.to_json()))
        assert isinstance(value, dict)
        return value

    changed_configs: list[dict[str, Any]] = []
    changed = changed_config()
    changed["transformer_config"]["hidden_dim"] += 1
    changed_configs.append(changed)
    changed = changed_config()
    changed["transformer_config"]["output_norm_config"]["upcast_mode"] = "only_normalization"
    changed_configs.append(changed)
    for norm_name in ("query_norm_config", "key_norm_config"):
        changed = changed_config()
        changed["transformer_config"]["layer_configs"][3]["mixer_config"][norm_name]["upcast_mode"] = (
            "only_normalization"
        )
        changed_configs.append(changed)
    changed = changed_config()
    changed["transformer_config"]["layer_configs"][0]["mixer_config"]["norm_config"]["upcast_mode"] = "full_layer"
    changed_configs.append(changed)

    for changed in changed_configs:
        (output / "config.json").write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="Transformer config differs from HF"):
            _verify_typed_output_config(source, output)


def test_file_inventory_ignores_only_published_script_bytecode_cache(tmp_path):
    bundle = tmp_path / "bundle"
    cache = bundle / "__pycache__"
    cache.mkdir(parents=True)
    (bundle / "CONVERSION.json").write_text("{}", encoding="utf-8")
    (bundle / "model.safetensors").write_bytes(b"archive")
    (cache / "d3_format.cpython-314.pyc").write_bytes(b"cache")
    (cache / "d3_runtime.cpython-314.pyc").write_bytes(b"cache")
    _check_exact_files(bundle, {"model.safetensors"}, "CONVERSION.json")

    (cache / "unrelated.cpython-314.pyc").write_bytes(b"extra")
    with pytest.raises(ValueError, match="Unexpected conversion file set"):
        _check_exact_files(bundle, {"model.safetensors"}, "CONVERSION.json")
