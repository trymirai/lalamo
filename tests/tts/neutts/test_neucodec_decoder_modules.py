from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.model_import.loaders.neutts_codec_loaders import load_neucodec_audio_decoder
from lalamo.modules.audio.neutts.audio_decoding import NeuCodecAudioDecoder, NeuCodecAudioDecoderConfig
from lalamo.modules.audio.neutts.codec_modules import (
    NeuCodecFSQConfig,
    NeuCodecISTFTHeadConfig,
    NeuCodecLayerNormConfig,
    NeuCodecResidualFSQConfig,
    NeuCodecResnetBlockConfig,
    NeuCodecTransformerBlockConfig,
    NeuCodecVocosBackboneConfig,
    NeuCodecVocosDecoderConfig,
)


def _state_array(state_dict: Mapping[str, Any], key: str) -> jnp.ndarray:
    return jnp.asarray(state_dict[key].detach().cpu().numpy())


def _linear_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {
        "weights": _state_array(state_dict, f"{prefix}.weight"),
        "biases": _state_array(state_dict, f"{prefix}.bias"),
    }


def _linear_no_bias_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {
        "weights": _state_array(state_dict, f"{prefix}.weight"),
    }


def _norm_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {
        "weights": _state_array(state_dict, f"{prefix}.weight"),
        "biases": _state_array(state_dict, f"{prefix}.bias"),
    }


def _rms_norm_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {
        "weights": _state_array(state_dict, f"{prefix}.weight"),
    }


def _prefixed(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _resnet_block_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "norm1": _norm_weights(state_dict, f"{prefix}.norm1"),
        "conv1": _linear_weights(state_dict, f"{prefix}.conv1"),
        "norm2": _norm_weights(state_dict, f"{prefix}.norm2"),
        "conv2": _linear_weights(state_dict, f"{prefix}.conv2"),
    }


def _transformer_block_weights(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "att_norm": _rms_norm_weights(state_dict, f"{prefix}.att_norm"),
        "ffn_norm": _rms_norm_weights(state_dict, f"{prefix}.ffn_norm"),
        "att": {
            "c_attn": _linear_no_bias_weights(state_dict, f"{prefix}.att.c_attn"),
            "c_proj": _linear_no_bias_weights(state_dict, f"{prefix}.att.c_proj"),
        },
        "mlp": {
            "fc1": _linear_no_bias_weights(state_dict, f"{prefix}.mlp.fc1"),
            "fc2": _linear_no_bias_weights(state_dict, f"{prefix}.mlp.fc2"),
        },
    }


def _vocos_backbone_weights(state_dict: Mapping[str, Any], *, depth: int, prefix: str) -> dict[str, Any]:
    return {
        "embed": _linear_weights(state_dict, _prefixed(prefix, "embed")),
        "prior_net": [
            _resnet_block_weights(state_dict, _prefixed(prefix, f"prior_net.{index}")) for index in range(2)
        ],
        "transformers": [
            _transformer_block_weights(state_dict, _prefixed(prefix, f"transformers.{index}"))
            for index in range(depth)
        ],
        "post_net": [_resnet_block_weights(state_dict, _prefixed(prefix, f"post_net.{index}")) for index in range(2)],
        "final_layer_norm": _norm_weights(state_dict, _prefixed(prefix, "final_layer_norm")),
    }


def _tiny_audio_decoder() -> NeuCodecAudioDecoder:
    return NeuCodecAudioDecoderConfig(
        precision=jnp.float32,
        levels=(4, 4),
        quantizer_output_dim=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        rotary_dim=8,
        hop_length=4,
    ).empty()


def _parameter_value(index: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + jnp.asarray(index + 1, dtype=jnp.float32)


def _expected_weight_norm_fusion(weights_g: jnp.ndarray, weights_v: jnp.ndarray) -> jnp.ndarray:
    reduction_axes = tuple(range(1, weights_v.ndim))
    norms = jnp.linalg.norm(weights_v, axis=reduction_axes, keepdims=True)
    reshaped_g = weights_g.reshape((weights_g.shape[0],) + (1,) * (weights_v.ndim - 1))
    return weights_v * reshaped_g / norms


def _add_linear_checkpoint_weights(
    checkpoint: dict[str, jnp.ndarray],
    module_weights: Mapping[str, Any],
    *,
    prefix: str,
    index: int,
) -> int:
    weights = module_weights["weights"]
    assert isinstance(weights, jnp.ndarray)
    checkpoint[f"{prefix}.weight"] = _parameter_value(index, weights.shape)
    checkpoint[f"{prefix}.weight_g"] = -jnp.ones_like(checkpoint[f"{prefix}.weight"])
    checkpoint[f"{prefix}.weight_v"] = -jnp.ones_like(checkpoint[f"{prefix}.weight"])
    next_index = index + 1
    biases = module_weights.get("biases")
    if biases is not None:
        assert isinstance(biases, jnp.ndarray)
        checkpoint[f"{prefix}.bias"] = _parameter_value(next_index, biases.shape)
        next_index += 1
    return next_index


def _add_norm_checkpoint_weights(
    checkpoint: dict[str, jnp.ndarray],
    module_weights: Mapping[str, Any],
    *,
    prefix: str,
    index: int,
) -> int:
    weights = module_weights["weights"]
    assert isinstance(weights, jnp.ndarray)
    checkpoint[f"{prefix}.weight"] = _parameter_value(index, weights.shape)
    next_index = index + 1
    biases = module_weights.get("biases")
    if biases is not None:
        assert isinstance(biases, jnp.ndarray)
        checkpoint[f"{prefix}.bias"] = _parameter_value(next_index, biases.shape)
        next_index += 1
    return next_index


def _add_resnet_checkpoint_weights(
    checkpoint: dict[str, jnp.ndarray],
    block_weights: Mapping[str, Any],
    *,
    prefix: str,
    index: int,
) -> int:
    next_index = _add_norm_checkpoint_weights(
        checkpoint,
        block_weights["norm1"],
        prefix=f"{prefix}.norm1",
        index=index,
    )
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["conv1"],
        prefix=f"{prefix}.conv1",
        index=next_index,
    )
    next_index = _add_norm_checkpoint_weights(
        checkpoint,
        block_weights["norm2"],
        prefix=f"{prefix}.norm2",
        index=next_index,
    )
    return _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["conv2"],
        prefix=f"{prefix}.conv2",
        index=next_index,
    )


def _add_transformer_checkpoint_weights(
    checkpoint: dict[str, jnp.ndarray],
    block_weights: Mapping[str, Any],
    *,
    prefix: str,
    index: int,
) -> int:
    next_index = _add_norm_checkpoint_weights(
        checkpoint,
        block_weights["att_norm"],
        prefix=f"{prefix}.att_norm",
        index=index,
    )
    next_index = _add_norm_checkpoint_weights(
        checkpoint,
        block_weights["ffn_norm"],
        prefix=f"{prefix}.ffn_norm",
        index=next_index,
    )
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["att"]["c_attn"],
        prefix=f"{prefix}.att.c_attn",
        index=next_index,
    )
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["att"]["c_proj"],
        prefix=f"{prefix}.att.c_proj",
        index=next_index,
    )
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["mlp"]["fc1"],
        prefix=f"{prefix}.mlp.fc1",
        index=next_index,
    )
    return _add_linear_checkpoint_weights(
        checkpoint,
        block_weights["mlp"]["fc2"],
        prefix=f"{prefix}.mlp.fc2",
        index=next_index,
    )


def _synthetic_neucodec_checkpoint(decoder: NeuCodecAudioDecoder) -> dict[str, jnp.ndarray]:
    weights = decoder.export_weights()
    checkpoint: dict[str, jnp.ndarray] = {}
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        weights["quantizer"]["project_out"],
        prefix="generator.quantizer.project_out",
        index=0,
    )
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        weights["fc_post_a"],
        prefix="fc_post_a",
        index=next_index,
    )
    backbone_weights = weights["vocos_decoder"]["backbone"]
    next_index = _add_linear_checkpoint_weights(
        checkpoint,
        backbone_weights["embed"],
        prefix="generator.backbone.embed",
        index=next_index,
    )
    for block_index, block_weights in enumerate(backbone_weights["prior_net"]):
        next_index = _add_resnet_checkpoint_weights(
            checkpoint,
            block_weights,
            prefix=f"generator.backbone.prior_net.{block_index}",
            index=next_index,
        )
    for block_index, block_weights in enumerate(backbone_weights["transformers"]):
        next_index = _add_transformer_checkpoint_weights(
            checkpoint,
            block_weights,
            prefix=f"generator.backbone.transformers.{block_index}",
            index=next_index,
        )
    for block_index, block_weights in enumerate(backbone_weights["post_net"]):
        next_index = _add_resnet_checkpoint_weights(
            checkpoint,
            block_weights,
            prefix=f"generator.backbone.post_net.{block_index}",
            index=next_index,
        )
    next_index = _add_norm_checkpoint_weights(
        checkpoint,
        backbone_weights["final_layer_norm"],
        prefix="generator.backbone.final_layer_norm",
        index=next_index,
    )
    _add_linear_checkpoint_weights(
        checkpoint,
        weights["vocos_decoder"]["head"]["out"],
        prefix="generator.head.out",
        index=next_index,
    )
    checkpoint["generator.quantizer.project_in.weight"] = jnp.asarray([123.0], dtype=jnp.float32)
    checkpoint["generator.head.istft.window"] = jnp.asarray([456.0], dtype=jnp.float32)
    checkpoint["encoder.weight"] = jnp.asarray([789.0], dtype=jnp.float32)
    return checkpoint


def test_load_neucodec_audio_decoder_maps_neuphonic_checkpoint_keys() -> None:
    decoder = _tiny_audio_decoder()
    checkpoint = _synthetic_neucodec_checkpoint(decoder)

    loaded_decoder = load_neucodec_audio_decoder(decoder, checkpoint)
    loaded_weights = loaded_decoder.export_weights()

    np.testing.assert_allclose(
        np.asarray(loaded_weights["quantizer"]["project_out"]["weights"]),
        np.asarray(checkpoint["generator.quantizer.project_out.weight"]),
    )
    np.testing.assert_allclose(
        np.asarray(loaded_weights["fc_post_a"]["biases"]),
        np.asarray(checkpoint["fc_post_a.bias"]),
    )
    np.testing.assert_allclose(
        np.asarray(loaded_weights["vocos_decoder"]["backbone"]["embed"]["weights"]),
        np.asarray(checkpoint["generator.backbone.embed.weight"]),
    )
    np.testing.assert_allclose(
        np.asarray(loaded_weights["vocos_decoder"]["backbone"]["transformers"][0]["att"]["c_attn"]["weights"]),
        np.asarray(checkpoint["generator.backbone.transformers.0.att.c_attn.weight"]),
    )
    np.testing.assert_allclose(
        np.asarray(loaded_weights["vocos_decoder"]["head"]["out"]["biases"]),
        np.asarray(checkpoint["generator.head.out.bias"]),
    )


def test_load_neucodec_audio_decoder_fuses_weight_norm_fallback() -> None:
    decoder = _tiny_audio_decoder()
    checkpoint = _synthetic_neucodec_checkpoint(decoder)
    original_weight = checkpoint.pop("fc_post_a.weight")
    weights_v = jnp.arange(original_weight.size, dtype=jnp.float32).reshape(original_weight.shape) + 1
    weights_g = jnp.arange(original_weight.shape[0], dtype=jnp.float32).reshape((original_weight.shape[0], 1)) + 1
    checkpoint["fc_post_a.weight_v"] = weights_v
    checkpoint["fc_post_a.weight_g"] = weights_g

    loaded_decoder = load_neucodec_audio_decoder(decoder, checkpoint)

    np.testing.assert_allclose(
        np.asarray(loaded_decoder.fc_post_a.weights),
        np.asarray(_expected_weight_norm_fusion(weights_g, weights_v)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_neucodec_fsq_indices_to_codes_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    finite_scalar_quantization = pytest.importorskip("vector_quantize_pytorch.finite_scalar_quantization")
    indices = [[0, 1, 255, 4**7, 16383, 65535]]
    torch_fsq = finite_scalar_quantization.FSQ(levels=[4] * 8)
    lalamo_fsq = NeuCodecFSQConfig(levels=(4,) * 8, precision=jnp.float32).empty()

    expected_codes = torch_fsq.indices_to_codes(torch.tensor(indices, dtype=torch.long)).detach().cpu().numpy()
    actual_codes = np.asarray(lalamo_fsq.indices_to_codes(jnp.asarray(indices, dtype=jnp.int32)))

    np.testing.assert_allclose(actual_codes, expected_codes, rtol=0, atol=0)


def test_neucodec_residual_fsq_get_output_from_indices_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    vector_quantize_pytorch = pytest.importorskip("vector_quantize_pytorch")
    indices = np.asarray([[[12], [34], [56], [65535]]], dtype=np.int64)
    torch_residual_fsq = vector_quantize_pytorch.ResidualFSQ(dim=2048, levels=[4] * 8, num_quantizers=1)
    lalamo_residual_fsq = NeuCodecResidualFSQConfig(
        levels=(4,) * 8,
        num_quantizers=1,
        output_dim=2048,
        precision=jnp.float32,
    ).empty()
    project_out_weights = {
        "weights": jnp.asarray(torch_residual_fsq.project_out.weight.detach().cpu().numpy()),
        "biases": jnp.asarray(torch_residual_fsq.project_out.bias.detach().cpu().numpy()),
    }
    lalamo_residual_fsq = lalamo_residual_fsq.import_weights(
        {
            "fsq": lalamo_residual_fsq.fsq.export_weights(),
            "project_out": project_out_weights,
        },
    )

    with torch.no_grad():
        expected_output = torch_residual_fsq.get_output_from_indices(torch.as_tensor(indices, dtype=torch.long))
    actual_output = np.asarray(lalamo_residual_fsq.get_output_from_indices(jnp.asarray(indices, dtype=jnp.int32)))

    np.testing.assert_allclose(actual_output, expected_output.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)


def test_neucodec_layer_norm_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    torch_norm = torch.nn.LayerNorm(32, eps=1e-6, elementwise_affine=True).eval()
    lalamo_norm = NeuCodecLayerNormConfig(dim=32, precision=jnp.float32).empty()
    torch_input = torch.linspace(-1.0, 1.0, steps=2 * 5 * 32, dtype=torch.float32).reshape(2, 5, 32)
    state_dict = torch_norm.state_dict()
    lalamo_norm = lalamo_norm.import_weights(
        {
            "weights": jnp.asarray(state_dict["weight"].detach().cpu().numpy()),
            "biases": jnp.asarray(state_dict["bias"].detach().cpu().numpy()),
        },
    )

    with torch.no_grad():
        expected_output = torch_norm(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_norm(jnp.asarray(torch_input.detach().cpu().numpy())))

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)


def test_neucodec_resnet_block_matches_torch_vocos_reference() -> None:
    torch = pytest.importorskip("torch")
    codec_decoder_vocos = pytest.importorskip("neucodec.codec_decoder_vocos")
    torch.manual_seed(0)
    torch_block = codec_decoder_vocos.ResnetBlock(
        in_channels=32,
        out_channels=32,
        temb_channels=0,
        dropout=0.1,
    ).eval()
    lalamo_block = NeuCodecResnetBlockConfig(channels=32, precision=jnp.float32).empty()
    torch_input = torch.linspace(-1.0, 1.0, steps=1 * 32 * 7, dtype=torch.float32).reshape(1, 32, 7)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy()).transpose(0, 2, 1)

    state_dict = torch_block.state_dict()
    lalamo_block = lalamo_block.import_weights(
        {
            "norm1": {
                "weights": jnp.asarray(state_dict["norm1.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["norm1.bias"].detach().cpu().numpy()),
            },
            "conv1": {
                "weights": jnp.asarray(state_dict["conv1.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["conv1.bias"].detach().cpu().numpy()),
            },
            "norm2": {
                "weights": jnp.asarray(state_dict["norm2.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["norm2.bias"].detach().cpu().numpy()),
            },
            "conv2": {
                "weights": jnp.asarray(state_dict["conv2.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["conv2.bias"].detach().cpu().numpy()),
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_block(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_block(lalamo_input)).transpose(0, 2, 1)

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)


def test_neucodec_transformer_block_matches_torch_vocos_reference() -> None:
    torch = pytest.importorskip("torch")
    bs_roformer5 = pytest.importorskip("neucodec.bs_roformer5")
    torchtune_modules = pytest.importorskip("torchtune.modules")
    torch.manual_seed(0)
    rotary = torchtune_modules.RotaryPositionalEmbeddings(dim=16)
    torch_block = bs_roformer5.TransformerBlock(dim=64, n_heads=4, rotary_embed=rotary).eval()
    lalamo_block = NeuCodecTransformerBlockConfig(dim=64, num_heads=4, rotary_dim=16, precision=jnp.float32).empty()
    torch_input = torch.linspace(-0.5, 0.5, steps=1 * 5 * 64, dtype=torch.float32).reshape(1, 5, 64)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy())

    state_dict = torch_block.state_dict()
    lalamo_block = lalamo_block.import_weights(
        {
            "att_norm": {
                "weights": jnp.asarray(state_dict["att_norm.weight"].detach().cpu().numpy()),
            },
            "ffn_norm": {
                "weights": jnp.asarray(state_dict["ffn_norm.weight"].detach().cpu().numpy()),
            },
            "att": {
                "c_attn": {
                    "weights": jnp.asarray(state_dict["att.c_attn.weight"].detach().cpu().numpy()),
                },
                "c_proj": {
                    "weights": jnp.asarray(state_dict["att.c_proj.weight"].detach().cpu().numpy()),
                },
            },
            "mlp": {
                "fc1": {
                    "weights": jnp.asarray(state_dict["mlp.fc1.weight"].detach().cpu().numpy()),
                },
                "fc2": {
                    "weights": jnp.asarray(state_dict["mlp.fc2.weight"].detach().cpu().numpy()),
                },
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_block(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_block(lalamo_input))

    np.testing.assert_allclose(actual_output, expected_output, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "num_frames"),
    [
        (16, 4, 5),
        (1280, 320, 3),
    ],
)
def test_neucodec_istft_head_matches_torch_vocos_reference(
    n_fft: int,
    hop_length: int,
    num_frames: int,
) -> None:
    torch = pytest.importorskip("torch")
    codec_decoder_vocos = pytest.importorskip("neucodec.codec_decoder_vocos")
    torch.manual_seed(0)
    torch_head = codec_decoder_vocos.ISTFTHead(dim=16, n_fft=n_fft, hop_length=hop_length, padding="same").eval()
    lalamo_head = NeuCodecISTFTHeadConfig(
        dim=16,
        n_fft=n_fft,
        hop_length=hop_length,
        precision=jnp.float32,
    ).empty()
    torch_input = torch.linspace(-0.2, 0.2, steps=num_frames * 16, dtype=torch.float32).reshape(1, num_frames, 16)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy())

    state_dict = torch_head.state_dict()
    lalamo_head = lalamo_head.import_weights(
        {
            "out": {
                "weights": jnp.asarray(state_dict["out.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["out.bias"].detach().cpu().numpy()),
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_head(torch_input)[0].detach().cpu().numpy()
    actual_output = np.asarray(lalamo_head(lalamo_input))

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)


def test_neucodec_vocos_backbone_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    codec_decoder_vocos = pytest.importorskip("neucodec.codec_decoder_vocos")
    hidden_dim = 32
    depth = 1
    heads = 4
    pos_meb_dim = 8
    torch.manual_seed(0)
    torch_backbone = codec_decoder_vocos.VocosBackbone(
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        pos_meb_dim=pos_meb_dim,
    ).eval()
    lalamo_backbone = NeuCodecVocosBackboneConfig(
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        rotary_dim=pos_meb_dim,
        precision=jnp.float32,
    ).empty()
    torch_input = torch.linspace(-0.3, 0.3, steps=2 * 5 * hidden_dim, dtype=torch.float32).reshape(2, 5, hidden_dim)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy())
    lalamo_backbone = lalamo_backbone.import_weights(
        _vocos_backbone_weights(torch_backbone.state_dict(), depth=depth, prefix=""),
    )

    with torch.no_grad():
        expected_output = torch_backbone(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_backbone(lalamo_input))

    np.testing.assert_allclose(actual_output, expected_output, rtol=3e-4, atol=3e-4)


def test_neucodec_vocos_decoder_vq_false_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    codec_decoder_vocos = pytest.importorskip("neucodec.codec_decoder_vocos")
    hidden_dim = 32
    depth = 1
    heads = 4
    pos_meb_dim = 8
    hop_length = 4
    torch.manual_seed(0)
    torch_decoder = codec_decoder_vocos.CodecDecoderVocos(
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        pos_meb_dim=pos_meb_dim,
        hop_length=hop_length,
    ).eval()
    lalamo_decoder = NeuCodecVocosDecoderConfig(
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        rotary_dim=pos_meb_dim,
        hop_length=hop_length,
        precision=jnp.float32,
    ).empty()
    torch_input = torch.linspace(-0.2, 0.2, steps=1 * 4 * hidden_dim, dtype=torch.float32).reshape(1, 4, hidden_dim)
    state_dict = torch_decoder.state_dict()
    lalamo_decoder = lalamo_decoder.import_weights(
        {
            "backbone": _vocos_backbone_weights(state_dict, depth=depth, prefix="backbone"),
            "head": {
                "out": _linear_weights(state_dict, "head.out"),
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_decoder(torch_input, vq=False)[0].detach().cpu().numpy()
    actual_output = np.asarray(lalamo_decoder(jnp.asarray(torch_input.detach().cpu().numpy())))

    np.testing.assert_allclose(actual_output, expected_output, rtol=5e-4, atol=5e-4)
