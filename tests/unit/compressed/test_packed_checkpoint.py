import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.hybrid import (
    HybridMatrix,
    HybridSpec,
    IncoherenceKind,
    IncoherenceProcessingMode,
    IncoherenceSigns,
    KroneckerRotation,
)
from lalamo.compressed.int import IntSpec
from lalamo.compressed.lattice import (
    COLUMNS_PER_LADDER_BYTE,
    LADDER_INDEX_BITS,
    LatticeKind,
    LatticeSpec,
)
from lalamo.compressed.qtip_gaussian import (
    STATE_BITS,
    QtipGaussianMatrix,
    QtipGaussianSpec,
    codebook_from_table,
    states_to_levels,
)
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.initializer import RandomInitializer
from lalamo.model_import.loaders.packed_checkpoint import load_packed_checkpoint
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
from lalamo.weight_matrix import FullPrecisionMatrix, Layout, WeightMatrix
from tests.helpers import build_tiny_attention_decoder_config, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

# A whole number of lattice ladder bytes.
MODEL_DIM = 3 * COLUMNS_PER_LADDER_BYTE
VOCABULARY = 32
# Package codebooks are float32 tables of CODEBOOK_SCALE * level + the offset of each column class.
CODEBOOK_SCALE = 0.05
CODEBOOK_OFFSETS = (0.31, -0.17, 0.08, 0.44)
QKVG = "decoder.transformer.layers.0.mixer.qkvg_projection.weights"


def contains_rows(loaded: Array, saved: Array) -> bool:
    # Merged trellis leaves concatenate rows, so a saved tensor is a window of the loaded one.
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
    """`folds` maps each trellis leaf's saved scales name to the saved tensors its one loaded scale multiplies."""
    consumed = {name for stages in folds.values() for name in stages}
    for leaf in jax.tree.leaves(model, is_leaf=lambda node: isinstance(node, QtipGaussianMatrix | KroneckerRotation)):
        if isinstance(leaf, QtipGaussianMatrix):
            width = leaf.spec.vector_width
            (table,) = [
                saved[name]
                for name in (f"qtip_shared.codebook_v{width}", f"qtip_shared.p8zm_v{width}")
                if name in saved
            ]
            np.testing.assert_array_equal(leaf.codebook, codebook_from_table(table))
        if isinstance(leaf, KroneckerRotation):
            columns = leaf.signs.shape[0]
            np.testing.assert_array_equal(leaf.signs, saved[f"qtip_shared.signs_{columns}"])
            np.testing.assert_array_equal(leaf.small_q, saved[f"qtip_shared.q_{columns}"])
    loaded = model.export().arrays
    for saved_name, value in saved.items():
        if (
            saved_name.startswith(("qtip_shared.", "decoder.transformer.ropes."))
            or saved_name in consumed - folds.keys()
        ):
            continue
        # Lattice sign vectors and the unfused layout's qkv and gate leaves have their own names on disk.
        name = re.sub(r"\.(input|output)_hadamard_factors$", r".\1_signs", saved_name)
        name = name.replace(".qkv_projection.weights.", ".qkvg_projection.weights.")
        name = name.replace(".gate_projection.weights.", ".qkvg_projection.weights.")
        matrix_path, _, leaf_name = name.rpartition(".")
        matrix_path = re.sub(r"\.parts\.\d+$", "", matrix_path)
        expected = jnp.asarray(
            math.prod(saved[stage].astype(jnp.float32) for stage in folds[saved_name])
            if saved_name in folds
            else value
        )
        assert any(
            contains_rows(array, expected)
            for loaded_name, array in loaded.items()
            if loaded_name.startswith(matrix_path + ".") and loaded_name.rpartition(".")[2] == leaf_name
        ), saved_name


def weight_matrices(model: LanguageModel) -> dict[str, WeightMatrix]:
    leaves = jax.tree_util.tree_leaves_with_path(model, is_leaf=lambda x: isinstance(x, WeightMatrix))
    return {str(ParameterPath() / path): leaf for path, leaf in leaves if isinstance(leaf, WeightMatrix)}


def i4s4(arrays: dict[str, Array], path: str, rows: int, seed: int = 0) -> dict[str, JSON]:
    """Saves an I4S4 leaf whose scales and row gains are powers of two, so its affine fold rounds nothing."""
    generator = np.random.default_rng(seed)
    arrays[path + ".input_hadamard_factors"] = jnp.asarray(generator.choice([-1, 1], MODEL_DIM), jnp.int32)
    arrays[path + ".codes"] = jnp.asarray(generator.integers(0, 256, (rows, MODEL_DIM // 2), np.uint8))
    arrays[path + ".ladder_indices"] = jnp.asarray(generator.integers(0, 256, (rows, MODEL_DIM // 128), np.uint8))
    arrays[path + ".row_scales"] = jnp.asarray(np.exp2(generator.integers(-3, 3, rows)), jnp.bfloat16)
    arrays[path + ".ladder"] = jnp.asarray(np.exp2(generator.integers(-3, 3, 1 << LADDER_INDEX_BITS)), jnp.float16)
    arrays[path + ".post_gains.0"] = jnp.asarray(np.exp2(generator.integers(-3, 3, rows)), jnp.float32)
    arrays[path + ".post_gains.1"] = jnp.ones(MODEL_DIM, jnp.float32)
    return {"type": "I4S4Spec", "layout": "output_input", "post_gain_axes": ["row", "column"]}


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
    directory: Path, model: LanguageModel, arrays: Mapping[str, Array], metadata: Mapping[str, JSON]
) -> None:
    (directory / "config.json").write_text(json.dumps(model.config.to_json()))
    model.token_codec.tokenizer.save(str(directory / "tokenizer.json"))
    with (directory / "model.safetensors").open("wb") as stream:
        safe_write(stream, arrays, metadata={name: json.dumps(value) for name, value in metadata.items()})


def load_with(directory: Path, replaced: str, arrays: dict[str, Array], metadata: dict[str, JSON]) -> LanguageModel:
    """Loads the tiny model with every tensor under `replaced` swapped for the given ones."""
    model = tiny_untied_model()
    exported = model.export()
    kept = {name: value for name, value in exported.arrays.items() if not name.startswith(replaced)}
    specs = {name: value for name, value in exported.metadata.items() if not name.startswith(replaced)}
    save_model(directory, model, kept | arrays, specs | metadata)
    return load_packed_checkpoint(directory, make_test_sharding_config())


def tiny_untied_model() -> LanguageModel:
    decoder = build_tiny_attention_decoder_config((None,))
    layer, *_ = decoder.transformer_config.layer_configs
    assert isinstance(layer.mixer_config, AttentionConfig)
    transformer = replace(
        decoder.transformer_config,
        model_dim=MODEL_DIM,
        layer_configs=(replace(layer, mixer_config=replace(layer.mixer_config, has_gate=True)),),
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
            transformer_config=transformer,
        ),
        generation_config=GenerationConfig(),
    )
    tokenizer = Tokenizer(WordLevel(vocab={f"token{i}": i for i in range(VOCABULARY)}, unk_token="token0"))
    return config.init(tokenizer, RandomInitializer(jnp.bfloat16, make_test_sharding_config(), key=jax.random.key(9)))


def assert_export_reloads(model: LanguageModel, directory: Path) -> None:
    """The exported layout loads back into the same matrices."""
    directory.mkdir()
    exported = model.export()
    save_model(directory, model, exported.arrays, exported.metadata)
    reloaded = LanguageModel.load(directory, make_test_sharding_config(), jnp.bfloat16)
    assert jax.tree.structure(reloaded) == jax.tree.structure(model)
    for reloaded_array, array in zip(jax.tree.leaves(reloaded), jax.tree.leaves(model), strict=True):
        if isinstance(array, Array):
            np.testing.assert_array_equal(reloaded_array, array)


def test_packed_checkpoint_load_routes_every_saved_format(tmp_path: Path) -> None:
    model = tiny_untied_model()
    generator = np.random.default_rng(0)
    layer = "decoder.transformer.layers.0."
    packed = (
        "decoder.embedding.input_embedding",
        "decoder.embedding.output_embedding",
        layer + "mixer.qkvg_projection.weights",
        layer + "mixer.out_projection.weights",
        layer + "mlp.up_projection.weights",
    )
    folds: dict[str, tuple[str, ...]] = {}
    exported = model.export()
    arrays = {name: value for name, value in exported.arrays.items() if name.rsplit(".", 1)[0] not in packed}
    metadata = {name: value for name, value in exported.metadata.items() if name.removesuffix(".spec") not in packed}

    def uniform(shape: tuple[int, ...], dtype: DTypeLike) -> Array:
        return jnp.asarray(generator.uniform(0.5, 1.5, shape).astype(np.float32)).astype(dtype)

    def packed_bytes(shape: tuple[int, ...]) -> Array:
        return jnp.asarray(generator.integers(0, 256, shape, dtype=np.uint8))

    def trellis(path: str, rows: int, columns: int, spec: QtipGaussianSpec, scale_dtype: str) -> dict[str, JSON]:
        arrays[path + ".codes"] = packed_bytes((rows, spec.code_bytes(columns)))
        arrays[path + ".scales"] = uniform((rows,), jnp.dtype(scale_dtype))
        arrays[path + ".gains"] = uniform((rows,), jnp.bfloat16)
        arrays[path + ".post_gains.0"] = uniform((rows,), jnp.float32)
        folds[path + ".scales"] = tuple(path + suffix for suffix in (".scales", ".gains", ".post_gains.0"))
        return {
            "type": "QtipGaussianSpec",
            "layout": "output_input",
            **asdict(spec),
            "scale_dtype": scale_dtype,
            "post_gain_axes": ["row"],
        }

    def lattice(path: str, rows: int, spec: LatticeSpec) -> dict[str, JSON]:
        arrays[path + ".codes"] = packed_bytes((rows, spec.code_bytes(MODEL_DIM)))
        arrays[path + ".row_scales"] = uniform((rows,), jnp.bfloat16)
        arrays[path + ".ladder_indices"] = packed_bytes((rows, MODEL_DIM // COLUMNS_PER_LADDER_BYTE))
        arrays[path + ".ladder"] = uniform((1 << LADDER_INDEX_BITS,), jnp.float16)
        signs = "output_hadamard_factors" if spec.layout == Layout.INPUT_OUTPUT else "input_hadamard_factors"
        arrays[f"{path}.{signs}"] = jnp.asarray(generator.choice([-1, 1], MODEL_DIM).astype(np.int32))
        if spec.kind == LatticeKind.D4:
            arrays[path + ".table"] = jnp.asarray(generator.integers(-8, 8, (256, 4), dtype=np.int8))
        return {"type": f"{spec.kind.upper()}S4Spec", "layout": spec.layout.value}

    readout = "decoder.embedding.output_embedding"
    metadata[readout + ".spec"] = lattice(readout, VOCABULARY, LatticeSpec(LatticeKind.I3, Layout.OUTPUT_INPUT))
    # Two trellis parts of one format and codebook merge into one leaf; the other format stays its own leaf.
    stack_specs = (
        (8, QtipGaussianSpec(4, 8, 64)),
        (8, QtipGaussianSpec(4, 8, 64)),
        (16, QtipGaussianSpec(2, 6, 0)),
    )
    qkvg = layer + "mixer.qkvg_projection.weights"
    metadata[qkvg + ".spec"] = {
        "type": "RowStackSpec",
        "parts": [
            [rows, trellis(f"{qkvg}.parts.{index}", rows, MODEL_DIM, spec, "float32")]
            for index, (rows, spec) in enumerate(stack_specs)
        ],
        "layout": "output_input",
    }
    out = layer + "mixer.out_projection.weights"
    metadata[out + ".spec"] = trellis(out, MODEL_DIM, 8, QtipGaussianSpec(2, 4, 0), "float16")
    up = layer + "mlp.up_projection.weights"
    metadata[up + ".spec"] = i4s4(arrays, up, 32)
    input_embedding = "decoder.embedding.input_embedding"
    metadata[input_embedding + ".spec"] = lattice(
        input_embedding, VOCABULARY, LatticeSpec(LatticeKind.D4, Layout.INPUT_OUTPUT)
    )
    for columns, order in ((MODEL_DIM, 3), (8, 1)):
        arrays[f"qtip_shared.signs_{columns}"] = jnp.asarray(generator.choice([-1.0, 1.0], columns).astype(np.float32))
        arrays[f"qtip_shared.q_{columns}"] = jnp.linalg.qr(
            jnp.asarray(generator.normal(size=(order, order)), jnp.float32)
        )[0]
    levels = states_to_levels(jnp.arange(1 << STATE_BITS, dtype=jnp.uint32))
    for width in (2, 4):
        offsets = jnp.asarray(CODEBOOK_OFFSETS[:width])
        arrays[f"qtip_shared.codebook_v{width}"] = CODEBOOK_SCALE * levels[:, :width].astype(jnp.float32) + offsets
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_json()))
    model.token_codec.tokenizer.save(str(tmp_path / "tokenizer.json"))
    with (tmp_path / "model.safetensors").open("wb") as stream:
        safe_write(stream, arrays, metadata={name: json.dumps(value) for name, value in metadata.items()})

    restored = LanguageModel.load(tmp_path, make_test_sharding_config())

    matrices = weight_matrices(restored)
    assert {name: type(matrix) for name, matrix in matrices.items()} == {
        input_embedding: HybridMatrix,
        readout: HybridMatrix,
        qkvg: HybridMatrix,
        out: HybridMatrix,
        up: HybridMatrix,
        layer + "mlp.down_projection.weights": FullPrecisionMatrix,
    }
    merged_specs = ((16, stack_specs[0][1]), stack_specs[2])
    assert matrices[qkvg].spec == HybridSpec(
        RowStackSpec(merged_specs), None, None, IncoherenceProcessingMode.INPUT, IncoherenceKind.KRONECKER
    )
    folded = matrices[up].astype(jnp.float32).decompress()
    np.testing.assert_array_equal(folded, saved_i4s4(arrays, up, ("row", "column"), make_test_sharding_config()))
    assert_loaded_every_saved_tensor(restored, {name: arrays[name] for name in arrays if up not in name}, folds)
    assert_export_reloads(restored, tmp_path / "exported")
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


@pytest.mark.parametrize("qkv_seed", [0, 1, None])
def test_qkv_and_i4s4_gate_merge_only_when_both_fold_under_shared_signs(tmp_path: Path, qkv_seed: int | None) -> None:
    """The gate's seed is 0: a qkv of seed 0 shares its signs, seed 1 does not, and None is a trellis qkv."""
    qkv, gate = QKVG.replace("qkvg", "qkv"), QKVG.replace("qkvg", "gate")
    arrays: dict[str, Array] = {}
    metadata: dict[str, JSON] = {gate + ".spec": i4s4(arrays, gate, 8)}
    if qkv_seed is None:
        spec = QtipGaussianSpec(4, 8, 64)
        arrays[qkv + ".codes"] = jnp.zeros((24, spec.code_bytes(MODEL_DIM)), jnp.uint8)
        arrays |= {qkv + ".scales": jnp.ones(24, jnp.float16), qkv + ".gains": jnp.ones(24, jnp.bfloat16)}
        arrays |= {f"qtip_shared.signs_{MODEL_DIM}": jnp.ones(MODEL_DIM), f"qtip_shared.q_{MODEL_DIM}": jnp.eye(3)}
        arrays["qtip_shared.codebook_v4"] = jnp.zeros((1 << STATE_BITS, 4))
        metadata[qkv + ".spec"] = {"type": "QtipGaussianSpec", "layout": "output_input", **asdict(spec)}
    else:
        metadata[qkv + ".spec"] = i4s4(arrays, qkv, 24, qkv_seed)
    model = load_with(tmp_path, QKVG, arrays, metadata)
    matrix = weight_matrices(model)[QKVG]
    if qkv_seed == 0:
        assert isinstance(matrix, HybridMatrix)
        config = make_test_sharding_config()
        expected = jnp.concatenate([saved_i4s4(arrays, name, ("row", "column"), config) for name in (qkv, gate)])
        np.testing.assert_array_equal(matrix.astype(jnp.float32).decompress(), expected)
    else:
        assert isinstance(matrix, RowStackMatrix)
        assert [*map(type, matrix.parts)] == [HybridMatrix, HybridMatrix]
        assert [part.spec.incoherence_kind for part in matrix.parts if isinstance(part, HybridMatrix)] == [
            IncoherenceKind.HADAMARD if qkv_seed else IncoherenceKind.KRONECKER,
            IncoherenceKind.HADAMARD,
        ]
    assert_export_reloads(model, tmp_path / "exported")


def test_hybrid_readout_loads_as_exported(tmp_path: Path) -> None:
    readout = "decoder.embedding.output_embedding"
    spec = HybridSpec(
        IntSpec(bits=4, group_size=64, is_symmetric=True, layout=Layout.OUTPUT_INPUT),
        None,
        32,
        IncoherenceProcessingMode.INPUT,
    )
    weights = jax.random.normal(jax.random.key(0), (VOCABULARY, MODEL_DIM), jnp.bfloat16)
    hybrid = spec.compress(weights, key=jax.random.key(1), sharding_config=make_test_sharding_config())
    exported = hybrid.export()
    arrays = {f"{readout}.{name}": value for name, value in exported.arrays.items()}
    model = load_with(tmp_path, readout, arrays, {readout + ".spec": exported.metadata["spec"]})
    np.testing.assert_array_equal(weight_matrices(model)[readout].decompress(), hybrid.decompress())
