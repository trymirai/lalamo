"""Shared loaders for TTS audio modules.

Generic loaders for ConvNeXtBlock, UpsamplingBlock, DecoderBlock, and DACDecoder.
Model-specific loaders pass activation and residual-unit callbacks to
customize behavior per model family (similar to how LLM loaders parameterize
path names in load_huggingface_decoder).
"""

from collections.abc import Callable, Mapping

from jaxtyping import Array

from lalamo.common import ParameterPath
from lalamo.modules.audio.common_modules import (
    ConvNeXtBlock,
    DACDecoder,
    DecoderBlock,
    ResidualUnit,
    UpsamplingBlock,
)

from .common import load_parameters
from .nanocodec_loaders import load_causal_conv1d, load_causal_transpose_conv1d

__all__ = [
    "load_convnext_block",
    "load_dac_decoder",
    "load_decoder_block",
    "load_upsampling_block",
]

type _ModuleLoader[M] = Callable[[M, Mapping[str, Array], ParameterPath], M]


def load_convnext_block(
    module: ConvNeXtBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ConvNeXtBlock:
    """Load ConvNeXtBlock, fusing gamma (LayerScale) into pwconv2 if present."""
    depthwise_conv = load_causal_conv1d(module.depthwise_conv, weights_dict, path / "dwconv" / "conv")
    norm = load_parameters(
        lambda m: (m.scales, m.biases),
        module.norm,
        (weights_dict[path / "norm" / "weight"], weights_dict[path / "norm" / "bias"]),
    )
    pointwise_conv1 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.pointwise_conv1,
        (weights_dict[path / "pwconv1" / "weight"], weights_dict[path / "pwconv1" / "bias"]),
    )

    pwconv2_weight = weights_dict[path / "pwconv2" / "weight"]
    pwconv2_bias = weights_dict[path / "pwconv2" / "bias"]
    gamma_path = path / "gamma"
    if gamma_path in weights_dict:
        gamma = weights_dict[gamma_path]
        pwconv2_weight = pwconv2_weight * gamma[:, None]
        pwconv2_bias = pwconv2_bias * gamma
    pointwise_conv2 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.pointwise_conv2,
        (pwconv2_weight, pwconv2_bias),
    )

    return load_parameters(
        lambda m: (m.depthwise_conv, m.norm, m.pointwise_conv1, m.pointwise_conv2),
        module,
        (depthwise_conv, norm, pointwise_conv1, pointwise_conv2),
    )


def load_upsampling_block(
    module: UpsamplingBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> UpsamplingBlock:
    trans_conv = load_causal_transpose_conv1d(module.trans_conv, weights_dict, path / "0" / "conv")
    convnext = load_convnext_block(module.convnext, weights_dict, path / "1")
    return load_parameters(
        lambda m: (m.trans_conv, m.convnext),
        module,
        (trans_conv, convnext),
    )


def load_decoder_block(
    module: DecoderBlock,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    load_activation: _ModuleLoader,  # type: ignore[type-arg]
    load_residual: _ModuleLoader[ResidualUnit],
) -> DecoderBlock:
    snake = load_activation(module.snake, weights_dict, path / "block" / "0")
    trans_conv = load_causal_transpose_conv1d(module.trans_conv, weights_dict, path / "block" / "1" / "conv")
    residual_units = tuple(
        load_residual(unit, weights_dict, path / "block" / (idx + 2)) for idx, unit in enumerate(module.residual_units)
    )
    return load_parameters(
        lambda m: (m.snake, m.trans_conv, m.residual_units),
        module,
        (snake, trans_conv, residual_units),
    )


def load_dac_decoder(
    module: DACDecoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    load_activation: _ModuleLoader,  # type: ignore[type-arg]
    load_residual: _ModuleLoader[ResidualUnit],
) -> DACDecoder:
    first_conv = load_causal_conv1d(module.first_conv, weights_dict, path / "0" / "conv")
    num_blocks = len(module.decoder_blocks)
    decoder_blocks = tuple(
        load_decoder_block(
            block,
            weights_dict,
            path / (idx + 1),
            load_activation=load_activation,
            load_residual=load_residual,
        )
        for idx, block in enumerate(module.decoder_blocks)
    )
    final_snake = load_activation(module.final_snake, weights_dict, path / (num_blocks + 1))
    final_conv = load_causal_conv1d(module.final_conv, weights_dict, path / (num_blocks + 2) / "conv")
    return load_parameters(
        lambda m: (m.first_conv, m.decoder_blocks, m.final_snake, m.final_conv),
        module,
        (first_conv, decoder_blocks, final_snake, final_conv),
    )
