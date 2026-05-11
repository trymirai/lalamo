from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath, cast_if_float
from lalamo.modules.audio.neutts.audio_decoding import NeuCodecAudioDecoder
from lalamo.modules.audio.neutts.codec_modules import (
    NeuCodecAttention,
    NeuCodecConv1d,
    NeuCodecGroupNorm,
    NeuCodecISTFTHead,
    NeuCodecLayerNorm,
    NeuCodecLinear,
    NeuCodecMLP,
    NeuCodecResidualFSQ,
    NeuCodecResnetBlock,
    NeuCodecRMSNorm,
    NeuCodecTransformerBlock,
    NeuCodecVocosBackbone,
    NeuCodecVocosDecoder,
)

from .common import load_parameters

__all__ = [
    "load_neucodec_audio_decoder",
    "load_neucodec_audio_decoder_from_huggingface",
    "load_neucodec_audio_decoder_from_pytorch_checkpoint",
]

NEUCODEC_REPO = "neuphonic/neucodec"
NEUCODEC_CHECKPOINT_FILENAME = "pytorch_model.bin"


def _fuse_weight_norm(weights_g: Array, weights_v: Array) -> Array:
    reduction_axes = tuple(range(1, weights_v.ndim))
    norms = jnp.linalg.norm(weights_v, axis=reduction_axes, keepdims=True)
    reshaped_g = weights_g.reshape((weights_g.shape[0],) + (1,) * (weights_v.ndim - 1))
    return weights_v * reshaped_g / norms


def _load_weight(weights_dict: Mapping[str, Array], path: ParameterPath) -> Array:
    weight_key = path / "weight"
    if weight_key in weights_dict:
        return weights_dict[weight_key]

    return _fuse_weight_norm(weights_dict[path / "weight_g"], weights_dict[path / "weight_v"])


def _load_linear(module: NeuCodecLinear, weights_dict: Mapping[str, Array], path: ParameterPath) -> NeuCodecLinear:
    weights = _load_weight(weights_dict, path)
    if module.biases is None:
        return load_parameters(lambda m: (m.weights,), module, (weights,))

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights, weights_dict[path / "bias"]),
    )


def _load_conv1d(module: NeuCodecConv1d, weights_dict: Mapping[str, Array], path: ParameterPath) -> NeuCodecConv1d:
    weights = _load_weight(weights_dict, path)
    if module.biases is None:
        return load_parameters(lambda m: (m.weights,), module, (weights,))

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights, weights_dict[path / "bias"]),
    )


def _load_group_norm(
    module: NeuCodecGroupNorm,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecGroupNorm:
    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights_dict[path / "weight"], weights_dict[path / "bias"]),
    )


def _load_layer_norm(
    module: NeuCodecLayerNorm,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecLayerNorm:
    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights_dict[path / "weight"], weights_dict[path / "bias"]),
    )


def _load_rms_norm(
    module: NeuCodecRMSNorm,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecRMSNorm:
    return load_parameters(
        lambda m: (m.weights,),
        module,
        (weights_dict[path / "weight"],),
    )


def _load_resnet_block(
    module: NeuCodecResnetBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecResnetBlock:
    norm1 = _load_group_norm(module.norm1, weights_dict, path / "norm1")
    conv1 = _load_conv1d(module.conv1, weights_dict, path / "conv1")
    norm2 = _load_group_norm(module.norm2, weights_dict, path / "norm2")
    conv2 = _load_conv1d(module.conv2, weights_dict, path / "conv2")
    return load_parameters(
        lambda m: (m.norm1, m.conv1, m.norm2, m.conv2),
        module,
        (norm1, conv1, norm2, conv2),
    )


def _load_attention(
    module: NeuCodecAttention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecAttention:
    c_attn = _load_linear(module.c_attn, weights_dict, path / "c_attn")
    c_proj = _load_linear(module.c_proj, weights_dict, path / "c_proj")
    return load_parameters(
        lambda m: (m.c_attn, m.c_proj),
        module,
        (c_attn, c_proj),
    )


def _load_mlp(module: NeuCodecMLP, weights_dict: Mapping[str, Array], path: ParameterPath) -> NeuCodecMLP:
    fc1 = _load_linear(module.fc1, weights_dict, path / "fc1")
    fc2 = _load_linear(module.fc2, weights_dict, path / "fc2")
    return load_parameters(
        lambda m: (m.fc1, m.fc2),
        module,
        (fc1, fc2),
    )


def _load_transformer_block(
    module: NeuCodecTransformerBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecTransformerBlock:
    att_norm = _load_rms_norm(module.att_norm, weights_dict, path / "att_norm")
    ffn_norm = _load_rms_norm(module.ffn_norm, weights_dict, path / "ffn_norm")
    att = _load_attention(module.att, weights_dict, path / "att")
    mlp = _load_mlp(module.mlp, weights_dict, path / "mlp")
    return load_parameters(
        lambda m: (m.att_norm, m.ffn_norm, m.att, m.mlp),
        module,
        (att_norm, ffn_norm, att, mlp),
    )


def _load_vocos_backbone(
    module: NeuCodecVocosBackbone,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecVocosBackbone:
    embed = _load_conv1d(module.embed, weights_dict, path / "embed")
    prior_net = tuple(
        _load_resnet_block(block, weights_dict, path / "prior_net" / index)
        for index, block in enumerate(module.prior_net)
    )
    transformers = tuple(
        _load_transformer_block(block, weights_dict, path / "transformers" / index)
        for index, block in enumerate(module.transformers)
    )
    post_net = tuple(
        _load_resnet_block(block, weights_dict, path / "post_net" / index)
        for index, block in enumerate(module.post_net)
    )
    final_layer_norm = _load_layer_norm(module.final_layer_norm, weights_dict, path / "final_layer_norm")
    return load_parameters(
        lambda m: (m.embed, m.prior_net, m.transformers, m.post_net, m.final_layer_norm),
        module,
        (embed, prior_net, transformers, post_net, final_layer_norm),
    )


def _load_istft_head(
    module: NeuCodecISTFTHead,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> NeuCodecISTFTHead:
    out = _load_linear(module.out, weights_dict, path / "out")
    return load_parameters(lambda m: (m.out,), module, (out,))


def _load_vocos_decoder(
    module: NeuCodecVocosDecoder,
    weights_dict: Mapping[str, Array],
) -> NeuCodecVocosDecoder:
    backbone = _load_vocos_backbone(module.backbone, weights_dict, ParameterPath("generator") / "backbone")
    head = _load_istft_head(module.head, weights_dict, ParameterPath("generator") / "head")
    return load_parameters(
        lambda m: (m.backbone, m.head),
        module,
        (backbone, head),
    )


def _load_residual_fsq(
    module: NeuCodecResidualFSQ,
    weights_dict: Mapping[str, Array],
) -> NeuCodecResidualFSQ:
    project_out = _load_linear(
        module.project_out,
        weights_dict,
        ParameterPath("generator") / "quantizer" / "project_out",
    )
    return load_parameters(lambda m: (m.project_out,), module, (project_out,))


def load_neucodec_audio_decoder(
    module: NeuCodecAudioDecoder,
    weights_dict: Mapping[str, Array],
) -> NeuCodecAudioDecoder:
    quantizer = _load_residual_fsq(module.quantizer, weights_dict)
    fc_post_a = _load_linear(module.fc_post_a, weights_dict, ParameterPath("fc_post_a"))
    vocos_decoder = _load_vocos_decoder(module.vocos_decoder, weights_dict)
    return load_parameters(
        lambda m: (m.quantizer, m.fc_post_a, m.vocos_decoder),
        module,
        (quantizer, fc_post_a, vocos_decoder),
    )


def _torch_checkpoint_to_jax_weights(torch_weights: Mapping[str, Any], precision: DTypeLike) -> dict[str, Array]:
    from lalamo.modules.torch_interop import torch_to_jax

    expected_prefixes = (
        "generator.quantizer.project_out.",
        "fc_post_a.",
        "generator.backbone.",
        "generator.head.out.",
    )
    return {
        str(key): cast_if_float(cast("Array", torch_to_jax(value)), precision)
        for key, value in torch_weights.items()
        if str(key).startswith(expected_prefixes)
    }


def load_neucodec_audio_decoder_from_pytorch_checkpoint(
    module: NeuCodecAudioDecoder,
    checkpoint_path: Path | str,
) -> NeuCodecAudioDecoder:
    import torch

    torch_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(torch_weights, Mapping):
        raise TypeError(f"Expected NeuCodec checkpoint to contain a mapping, got {type(torch_weights)}.")

    weights_dict = _torch_checkpoint_to_jax_weights(torch_weights, module.config.precision)
    return load_neucodec_audio_decoder(module, weights_dict)


def load_neucodec_audio_decoder_from_huggingface(
    module: NeuCodecAudioDecoder,
    *,
    repo_id: str = NEUCODEC_REPO,
    filename: str = NEUCODEC_CHECKPOINT_FILENAME,
    revision: str | None = None,
    local_files_only: bool = False,
) -> NeuCodecAudioDecoder:
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_files_only=local_files_only,
    )
    return load_neucodec_audio_decoder_from_pytorch_checkpoint(module, checkpoint_path)
