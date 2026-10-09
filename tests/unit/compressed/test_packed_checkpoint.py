import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.hybrid import (
    HybridSpec,
    IncoherenceKind,
    IncoherenceProcessingMode,
    IncoherenceSigns,
    KroneckerRotation,
)
from lalamo.compressed.int import IntSpec
from lalamo.compressed.lattice import LatticeKind, LatticeSpec
from lalamo.compressed.mlx import MLXSpec
from lalamo.compressed.qtip_gaussian import (
    STATE_BITS,
    QtipGaussianMatrix,
    QtipGaussianSpec,
    codebook_from_table,
    states_to_levels,
)
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.initializer import RandomInitializer
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.module import Keychain
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.safetensors import safe_write
from lalamo.utils.json import JSON
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
from lalamo.weight_matrix import Layout, WeightMatrix
from tests.helpers import build_tiny_attention_decoder_config, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

# Three lattice ladder bytes, rotated by kron(H_128, Q_3).
MODEL_DIM = 384
VOCABULARY = 32


def contains_rows(loaded: Array, saved: Array) -> bool:
    # Merged leaves concatenate rows, so a saved tensor is a window of the loaded one.
    # Plain tensors may load widened to their declared dtype (bf16 norm scales as float32), never rounded.
    array, saved_array = np.atleast_1d(np.asarray(loaded)), np.atleast_1d(np.asarray(saved))
    expected = saved_array.astype(array.dtype)
    if not np.array_equal(expected.astype(saved_array.dtype), saved_array):
        return False
    windows = (array[start : start + len(expected)] for start in range(len(array) - len(expected) + 1))
    return any(np.array_equal(window, expected) for window in windows)


def assert_loaded_every_saved_tensor(
    model: LanguageModel, saved: Mapping[str, Array], folds: Mapping[str, tuple[str, ...]]
) -> None:
    # `folds` maps each trellis leaf's saved scales to the saved row gains its one loaded scale multiplies.
    for leaf in jax.tree.leaves(model, is_leaf=lambda node: isinstance(node, QtipGaussianMatrix | KroneckerRotation)):
        if isinstance(leaf, QtipGaussianMatrix):
            width = leaf.spec.vector_width
            tables = [table for name, table in saved.items() if re.fullmatch(rf"qtip_shared\.\w+_v{width}", name)]
            assert any(np.array_equal(leaf.codebook, codebook_from_table(table)) for table in tables)
        if isinstance(leaf, KroneckerRotation):
            np.testing.assert_array_equal(leaf.signs, saved[f"qtip_shared.signs_{len(leaf.signs)}"])
            np.testing.assert_array_equal(leaf.small_q, saved[f"qtip_shared.q_{len(leaf.signs)}"])
    loaded = model.export().arrays
    folded = {name for stages in folds.values() for name in stages} - folds.keys()
    for saved_name, value in saved.items():
        if saved_name.startswith(("qtip_shared.", "decoder.transformer.ropes.")) or saved_name in folded:
            continue
        # Lattice sign vectors and the legacy qkv and gate leaves have their own names on disk.
        name = re.sub(r"\.(input|output)_hadamard_factors$", r".\1_signs", saved_name)
        name = re.sub(r"\.(qkv|gate)_projection\.", ".qkvg_projection.", name)
        matrix_path, _, leaf_name = re.sub(r"\.parts\.\d+(\.\w+)$", r"\1", name).rpartition(".")
        expected = value
        if saved_name in folds:
            expected = jnp.asarray(math.prod(saved[stage].astype(jnp.float32) for stage in folds[saved_name]))
        assert any(
            contains_rows(array, expected)
            for loaded_name, array in loaded.items()
            if loaded_name.startswith(matrix_path + ".") and loaded_name.endswith("." + leaf_name)
        ), saved_name


def weight_matrices(model: LanguageModel) -> dict[str, WeightMatrix]:
    leaves = jax.tree_util.tree_leaves_with_path(model, is_leaf=lambda x: isinstance(x, WeightMatrix))
    return {str(ParameterPath() / path): leaf for path, leaf in leaves if isinstance(leaf, WeightMatrix)}


def saved_i4s4(arrays: Mapping[str, Array], path: str, axes: Sequence[str], sharding_config: ShardingConfig) -> Array:
    # The producer's decode: even column in the high nibble, level 2c - 15, a ladder value per 64 columns.
    codes, ladder_indices = np.asarray(arrays[path + ".codes"]), np.asarray(arrays[path + ".ladder_indices"])
    levels = 2 * np.stack((codes >> 4, codes & 15), axis=-1).reshape(len(codes), -1).astype(np.float32) - 15
    groups = np.stack((ladder_indices & 15, ladder_indices >> 4), axis=-1).reshape(len(codes), -1)
    ladder = np.repeat(np.asarray(arrays[path + ".ladder"], np.float32)[groups], 64, axis=1)
    row_gains = [arrays[f"{path}.post_gains.{index}"] for index, axis in enumerate(axes) if axis == "row"]
    row_scales = math.prod(row_gains, start=np.asarray(arrays[path + ".row_scales"], np.float32))
    signs = IncoherenceSigns(jnp.asarray(arrays[path + ".input_hadamard_factors"]), None)
    return signs.unprocess_weights(jnp.asarray(levels * ladder * row_scales[:, None]), 32, sharding_config)


def test_i4s4_reference_matches_torch_decoded_rows() -> None:
    with np.load(Path(__file__).parent / "data" / "lattice_i4.npz") as data:
        arrays = {f"i4.{name}": data[name] for name in ("codes", "ladder_indices", "ladder")}
        arrays["i4.row_scales"] = data["row_scale_bits"].view(jnp.bfloat16)
        arrays["i4.input_hadamard_factors"] = data["signs"]
        reference = saved_i4s4(arrays, "i4", (), make_test_sharding_config())
        np.testing.assert_allclose(reference, data["expected"], atol=2e-7, rtol=1e-6)


def save_model(
    directory: Path, config: JSON, model: LanguageModel, arrays: Mapping[str, Array], metadata: Mapping[str, JSON]
) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    model.token_codec.tokenizer.save(str(directory / "tokenizer.json"))
    with (directory / "model.safetensors").open("wb") as stream:
        safe_write(stream, arrays, metadata={name: json.dumps(value) for name, value in metadata.items()})


def tiny_untied_model() -> LanguageModel:
    decoder = build_tiny_attention_decoder_config((None, None))
    layer_configs = tuple(
        replace(layer, mixer_config=replace(layer.mixer_config, has_gate=True))
        for layer in decoder.transformer_config.layer_configs
        if isinstance(layer.mixer_config, AttentionConfig)
    )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="{{ messages[0]['content'] }}",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        decoder_config=replace(
            decoder,
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            transformer_config=replace(decoder.transformer_config, model_dim=MODEL_DIM, layer_configs=layer_configs),
        ),
        generation_config=GenerationConfig(),
    )
    tokenizer = Tokenizer(WordLevel(vocab={f"token{i}": i for i in range(VOCABULARY)}, unk_token="token0"))
    return config.init(tokenizer, RandomInitializer(jnp.bfloat16, make_test_sharding_config(), key=jax.random.key(9)))


def test_packed_checkpoint_load_routes_every_saved_format(tmp_path: Path) -> None:
    first, second = "decoder.transformer.layers.0.", "decoder.transformer.layers.1."
    qkvg, out = first + "mixer.qkvg_projection.weights", first + "mixer.out_projection.weights"
    fused, up = second + "mixer.qkvg_projection.weights", second + "mlp.up_projection.weights"
    embedding, readout = "decoder.embedding.input_embedding", "decoder.embedding.output_embedding"
    model = tiny_untied_model()
    exported = model.export()
    packed = (qkvg, out, fused, up, embedding, readout)
    arrays = {name: value for name, value in exported.arrays.items() if name.rsplit(".", 1)[0] not in packed}
    metadata = {name: value for name, value in exported.metadata.items() if name.removesuffix(".spec") not in packed}
    folds: dict[str, tuple[str, ...]] = {}
    generator = np.random.default_rng(0)

    def random_signs() -> Array:
        return jnp.asarray(generator.choice([-1, 1], MODEL_DIM), jnp.int32)

    def uniform(shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return jnp.asarray(generator.uniform(0.5, 1.5, shape).astype(np.float32)).astype(dtype)

    def packed_bytes(shape: tuple[int, ...]) -> Array:
        return jnp.asarray(generator.integers(0, 256, shape, dtype=np.uint8))

    def trellis(path: str, rows: int, columns: int, spec: QtipGaussianSpec, scale_dtype: str) -> JSON:
        arrays[path + ".codes"] = packed_bytes((rows, spec.code_bytes(columns)))
        stages = {"scales": scale_dtype, "gains": "bfloat16", "pre_gains.0": "float32", "post_gains.0": "float32"}
        arrays.update({f"{path}.{name}": uniform((rows,), jnp.dtype(dtype)) for name, dtype in stages.items()})
        folds[path + ".scales"] = tuple(f"{path}.{name}" for name in stages)
        saved = {"layout": "output_input", **asdict(spec), "scale_dtype": scale_dtype, "pre_gain_count": 1}
        return {"type": "QtipGaussianSpec", **saved, "post_gain_axes": ["row"]}

    def lattice(path: str, rows: int, spec: LatticeSpec) -> JSON:
        arrays[path + ".codes"] = packed_bytes((rows, spec.code_bytes(MODEL_DIM)))
        arrays[path + ".row_scales"] = uniform((rows,), jnp.bfloat16)
        arrays[path + ".ladder_indices"] = packed_bytes((rows, MODEL_DIM // 128))
        arrays[path + ".ladder"] = uniform((16,), jnp.float16)
        signs = "output_hadamard_factors" if spec.layout == Layout.INPUT_OUTPUT else "input_hadamard_factors"
        arrays[f"{path}.{signs}"] = random_signs()
        if spec.kind == LatticeKind.D4:
            arrays[path + ".table"] = jnp.asarray(generator.integers(-8, 8, (256, 4), dtype=np.int8))
        return {"type": f"{spec.kind.upper()}S4Spec", "layout": spec.layout.value}

    def i4s4(path: str, rows: int, signs: Array) -> JSON:
        # Powers of two, so the affine fold rounds nothing.
        arrays[path + ".input_hadamard_factors"] = signs
        arrays[path + ".codes"] = packed_bytes((rows, MODEL_DIM // 2))
        arrays[path + ".ladder_indices"] = packed_bytes((rows, MODEL_DIM // 128))
        arrays[path + ".row_scales"] = jnp.asarray(np.exp2(generator.integers(-3, 3, rows)), jnp.bfloat16)
        arrays[path + ".ladder"] = jnp.asarray(np.exp2(generator.integers(-3, 3, 16)), jnp.float16)
        arrays[path + ".post_gains.0"] = jnp.asarray(np.exp2(generator.integers(-3, 3, rows)), jnp.float32)
        arrays[path + ".post_gains.1"] = jnp.ones(MODEL_DIM, jnp.float32)
        return {"type": "I4S4Spec", "layout": "output_input", "post_gain_axes": ["row", "column"]}

    v4, v2 = QtipGaussianSpec(4, 8, 64), QtipGaussianSpec(2, 6, 0)
    parts: list[JSON] = [
        [8, trellis(f"{qkvg}.parts.{index}", 8, MODEL_DIM, spec, "float32")] for index, spec in enumerate((v4, v4, v2))
    ]
    parts += [[4, i4s4(f"{qkvg}.parts.{index}", 4, random_signs())] for index in (3, 4)]
    metadata[qkvg + ".spec"] = {"type": "RowStackSpec", "parts": parts, "layout": "output_input"}
    metadata[out + ".spec"] = trellis(out, MODEL_DIM, 8, QtipGaussianSpec(2, 4, 0), "float16")
    # The second layer saves its attention projection as a legacy qkv leaf and a gate leaf.
    legacy = (second + "mixer.qkv_projection.weights", second + "mixer.gate_projection.weights")
    shared_signs = random_signs()
    metadata |= {path + ".spec": i4s4(path, rows, shared_signs) for path, rows in zip(legacy, (24, 8), strict=True)}
    metadata[up + ".spec"] = lattice(up, 32, LatticeSpec(LatticeKind.I3, Layout.OUTPUT_INPUT))
    metadata[embedding + ".spec"] = lattice(embedding, VOCABULARY, LatticeSpec(LatticeKind.D4, Layout.INPUT_OUTPUT))
    int4 = HybridSpec(IntSpec(4, 64, is_symmetric=True), None, 32, IncoherenceProcessingMode.INPUT)
    readout_weights = jax.random.normal(jax.random.key(0), (VOCABULARY, MODEL_DIM), jnp.bfloat16)
    readout_hybrid = int4.compress(readout_weights, key=jax.random.key(1), sharding_config=make_test_sharding_config())
    metadata[readout + ".spec"] = readout_hybrid.spec.to_json()
    arrays |= {f"{readout}.{name}": value for name, value in readout_hybrid.export().arrays.items()}
    for columns, order in ((MODEL_DIM, 3), (8, 1)):
        arrays[f"qtip_shared.signs_{columns}"] = jnp.asarray(generator.choice([-1.0, 1.0], columns), jnp.float32)
        arrays[f"qtip_shared.q_{columns}"] = jnp.asarray(
            np.linalg.qr(generator.normal(size=(order, order)))[0], jnp.float32
        )
    # Package codebooks hold scale * level + the offset of each column class for every state.
    levels = states_to_levels(jnp.arange(1 << STATE_BITS, dtype=jnp.uint32)).astype(jnp.float32)
    offsets = jnp.asarray((0.31, -0.17, 0.08, 0.44))
    for width in (2, 4):
        arrays[f"qtip_shared.codebook_v{width}"] = 0.05 * levels[:, :width] + offsets[:width]
    # Older packages save attention as separate qkv and gate projections, beside a null pard_token.
    config: Any = model.config.to_json()
    config["decoder_config"]["pard_token"] = None
    for layer in config["decoder_config"]["transformer_config"]["layer_configs"]:
        mixer = layer["mixer_config"]
        mixer["qkv_projection_config"] = mixer["gate_projection_config"] = mixer.pop("qkvg_projection_config")
        mixer["has_qkv_biases"] = mixer.pop("has_qkvg_biases")
        del mixer["has_gate"]
    save_model(tmp_path, config, model, arrays, metadata)

    restored = LanguageModel.load(tmp_path, make_test_sharding_config())

    matrices = weight_matrices(restored)
    sharding_config = make_test_sharding_config()
    # Same-format trellis neighbours merge under their shared Kronecker rotation; I4S4 parts of other signs do not.
    kronecker = HybridSpec(
        RowStackSpec(((16, v4), (8, v2))), None, None, IncoherenceProcessingMode.INPUT, IncoherenceKind.KRONECKER
    )
    folded_i4s4 = HybridSpec(MLXSpec(4, 64), None, incoherence_processing_mode=IncoherenceProcessingMode.INPUT)
    stack = matrices[qkvg]
    assert isinstance(stack, RowStackMatrix)
    assert stack.spec == RowStackSpec(((24, kronecker), (4, folded_i4s4), (4, folded_i4s4)))
    for index, part in zip((3, 4), stack.parts[1:], strict=True):
        expected = saved_i4s4(arrays, f"{qkvg}.parts.{index}", ("row", "column"), sharding_config)
        np.testing.assert_array_equal(part.astype(jnp.float32).decompress(), expected)
    # Legacy qkv and gate I4S4 leaves under one rotation merge into one affine int4 leaf.
    expected = jnp.concatenate([saved_i4s4(arrays, path, ("row", "column"), sharding_config) for path in legacy])
    assert matrices[fused].spec == folded_i4s4
    np.testing.assert_array_equal(matrices[fused].astype(jnp.float32).decompress(), expected)
    i4s4_prefixes = (f"{qkvg}.parts.3.", f"{qkvg}.parts.4.", *(path + "." for path in legacy))
    unfolded = {name: value for name, value in arrays.items() if not name.startswith(i4s4_prefixes)}
    assert_loaded_every_saved_tensor(restored, unfolded, folds)

    exported = restored.export()
    save_model(tmp_path / "exported", restored.config.to_json(), restored, exported.arrays, exported.metadata)
    reloaded = LanguageModel.load(tmp_path / "exported", sharding_config, jnp.bfloat16)
    assert jax.tree.structure(reloaded) == jax.tree.structure(restored)
    for reloaded_array, array in zip(jax.tree.leaves(reloaded), jax.tree.leaves(restored), strict=True):
        if isinstance(array, Array):
            np.testing.assert_array_equal(reloaded_array, array)

    batch_sharding = restored.sharding_config.resolve_sharding((LogicalAxis.BATCH, None))
    tokens = jax.device_put(jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32), batch_sharding)
    result = restored.decoder(
        tokens,
        jax.device_put(jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), tokens.shape), batch_sharding),
        state=restored.decoder.init_static_state(batch_size=2, capacity=4, dtype=jnp.bfloat16),
        keychain=Keychain.init(0, sharding_config=restored.sharding_config),
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
    )
    assert bool(jnp.all(jnp.isfinite(result.logits)))
