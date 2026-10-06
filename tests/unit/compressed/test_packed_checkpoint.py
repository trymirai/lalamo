import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.direction import BLOCK_BYTES, BLOCK_COLUMNS, LEVEL_BITS, DirectionMatrix
from lalamo.compressed.lattice import (
    COLUMNS_PER_LADDER_BYTE,
    LADDER_INDEX_BITS,
    LatticeKind,
    LatticeMatrix,
    LatticeSpec,
)
from lalamo.compressed.qtip_gaussian import STATE_BITS, QtipGaussianMatrix, QtipGaussianSpec
from lalamo.compressed.row_stack import RowStackMatrix, RowStackSpec
from lalamo.compressed.trellis import states_to_levels
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
from lalamo.utils.sharding import LogicalAxis
from lalamo.weight_matrix import FullPrecisionMatrix, Layout, WeightMatrix
from tests.helpers import build_tiny_attention_decoder_config, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

# One direction block plus a dense tail, and a whole number of lattice ladder bytes.
MODEL_DIM = BLOCK_COLUMNS + COLUMNS_PER_LADDER_BYTE
VOCABULARY = 32
# Package codebooks are float32 tables of CODEBOOK_SCALE * level + the offset of each column class.
CODEBOOK_SCALE = 0.05
CODEBOOK_OFFSETS = (0.31, -0.17, 0.08, 0.44)


def assert_loaded_every_saved_tensor(
    model: LanguageModel, saved: Mapping[str, Array], folds: Mapping[str, tuple[str, ...]]
) -> None:
    """`folds` maps each trellis leaf's saved scales name to the saved tensors its one loaded scale multiplies."""
    consumed = {name for stages in folds.values() for name in stages}
    for leaf in jax.tree.leaves(model, is_leaf=lambda node: isinstance(node, QtipGaussianMatrix)):
        if isinstance(leaf, QtipGaussianMatrix):
            _, columns = leaf.shape
            width = leaf.spec.vector_width
            expected_codebook = (CODEBOOK_SCALE, *CODEBOOK_OFFSETS[:width] * (4 // width))
            np.testing.assert_allclose(leaf.codebook, expected_codebook, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(leaf.signs, saved[f"qtip_shared.signs_{columns}"])
            np.testing.assert_array_equal(leaf.small_q, saved[f"qtip_shared.q_{columns}"])
    loaded = model.export().arrays
    for saved_name, value in saved.items():
        if saved_name.startswith(("qtip_shared.", "decoder.transformer.ropes.")):
            continue
        # Lattice sign vectors and the unfused layout's qkv and gate leaves have their own names on disk.
        name = re.sub(r"\.(input|output)_hadamard_factors$", ".signs", saved_name)
        name = name.replace(".qkv_projection.weights.", ".qkvg_projection.weights.parts.0.")
        name = name.replace(".gate_projection.weights.", ".qkvg_projection.weights.parts.1.")
        if saved_name in folds:
            expected = math.prod(saved[stage].astype(jnp.float32) for stage in folds[saved_name])
            np.testing.assert_array_equal(loaded[name], expected, err_msg=name)
        elif saved_name not in consumed:
            assert loaded[name].dtype == value.dtype, name
            np.testing.assert_array_equal(loaded[name], value, err_msg=name)


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
    arrays[readout + ".codes"] = packed_bytes((VOCABULARY, BLOCK_BYTES))
    arrays[readout + ".levels"] = jnp.linspace(-1.5, 1.5, 1 << LEVEL_BITS, dtype=jnp.float32)
    arrays[readout + ".unit_scale"] = jnp.float32(0.7)
    arrays[readout + ".mean_norm"] = jnp.float32(3.0)
    arrays[readout + ".tail"] = uniform((VOCABULARY, MODEL_DIM - BLOCK_COLUMNS), jnp.bfloat16)
    metadata[readout + ".spec"] = {"type": "SDirectionSpec", "layout": "output_input"}
    stack_specs = (
        (24, QtipGaussianSpec(4, 8, 64)),
        (8, LatticeSpec(LatticeKind.I3, Layout.OUTPUT_INPUT)),
    )
    qkvg = layer + "mixer.qkvg_projection.weights"
    metadata[qkvg + ".spec"] = {
        "type": "RowStackSpec",
        "parts": [
            [24, trellis(qkvg + ".parts.0", 24, MODEL_DIM, stack_specs[0][1], "float32")],
            [8, lattice(qkvg + ".parts.1", 8, stack_specs[1][1])],
        ],
        "layout": "output_input",
    }
    out = layer + "mixer.out_projection.weights"
    metadata[out + ".spec"] = trellis(out, MODEL_DIM, 8, QtipGaussianSpec(2, 4, 0), "float16")
    up = layer + "mlp.up_projection.weights"
    metadata[up + ".spec"] = lattice(up, 32, LatticeSpec(LatticeKind.I4, Layout.OUTPUT_INPUT))
    input_embedding = "decoder.embedding.input_embedding"
    metadata[input_embedding + ".spec"] = lattice(
        input_embedding, VOCABULARY, LatticeSpec(LatticeKind.D4, Layout.INPUT_OUTPUT)
    )
    for columns, order in ((MODEL_DIM, 9), (8, 1)):
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

    matrices = {
        str(ParameterPath() / path): leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(restored, is_leaf=lambda x: isinstance(x, WeightMatrix))
        if isinstance(leaf, WeightMatrix)
    }
    assert {name: type(matrix) for name, matrix in matrices.items()} == {
        input_embedding: LatticeMatrix,
        readout: DirectionMatrix,
        qkvg: RowStackMatrix,
        out: QtipGaussianMatrix,
        up: LatticeMatrix,
        layer + "mlp.down_projection.weights": FullPrecisionMatrix,
    }
    assert matrices[qkvg].spec == RowStackSpec(stack_specs)
    assert_loaded_every_saved_tensor(restored, arrays, folds)
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
