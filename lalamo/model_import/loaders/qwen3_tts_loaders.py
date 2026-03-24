from collections.abc import Mapping

from einops import rearrange
from jaxtyping import Array

from lalamo.common import ParameterPath
from lalamo.modules import FullPrecisionLinear, Transformer
from lalamo.modules.audio.common_modules import (
    ResidualUnit,
    SnakeBeta,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_audio_decoding import (
    Qwen3TTSAudioDecoder,
    Qwen3TTSPreTransformer,
    Qwen3TTSPreTransformerLayer,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_modules import (
    Qwen3TTSEuclideanCodebook,
    Qwen3TTSResidualVectorQuantization,
    Qwen3TTSResidualVectorQuantizer,
    Qwen3TTSSplitResidualVectorQuantizer,
    Qwen3TTSVectorQuantization,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_text_decoding import Qwen3TTSTextDecoder

from .audio_loaders import load_dac_decoder, load_upsampling_block
from .common import load_parameters
from .huggingface import load_attention, load_mlp, load_transformer_layer
from .nanocodec_loaders import load_causal_conv1d

__all__ = [
    "load_qwen3_tts_audio_decoder",
    "load_qwen3_tts_pre_transformer",
    "load_qwen3_tts_residual_unit",
    "load_qwen3_tts_snake_beta",
    "load_qwen3_tts_split_rvq",
    "load_qwen3_tts_text_decoder",
    "load_qwen3_tts_vector_quantization",
]


def load_qwen3_tts_snake_beta(
    module: SnakeBeta,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> SnakeBeta:
    return load_parameters(
        lambda m: (m.alpha, m.beta),
        module,
        (
            weights_dict[path / "alpha"],
            weights_dict[path / "beta"],
        ),
    )


def load_qwen3_tts_residual_unit(
    module: ResidualUnit,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> ResidualUnit:
    assert isinstance(module.act1, SnakeBeta)
    assert isinstance(module.act2, SnakeBeta)
    act1 = load_qwen3_tts_snake_beta(module.act1, weights_dict, path / "act1")
    conv1 = load_causal_conv1d(module.conv1, weights_dict, path / "conv1" / "conv")
    act2 = load_qwen3_tts_snake_beta(module.act2, weights_dict, path / "act2")
    conv2 = load_causal_conv1d(module.conv2, weights_dict, path / "conv2" / "conv")
    return load_parameters(
        lambda m: (m.act1, m.conv1, m.act2, m.conv2),
        module,
        (act1, conv1, act2, conv2),
    )


def load_qwen3_tts_euclidean_codebook(
    module: Qwen3TTSEuclideanCodebook,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSEuclideanCodebook:
    return load_parameters(
        lambda m: (m.cluster_usage, m.embedding_sum),
        module,
        (
            weights_dict[path / "cluster_usage"],
            weights_dict[path / "embedding_sum"],
        ),
    )


def load_qwen3_tts_vector_quantization(
    module: Qwen3TTSVectorQuantization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSVectorQuantization:
    codebook = load_qwen3_tts_euclidean_codebook(module.codebook, weights_dict, path / "_codebook")

    if module.project_out is not None:
        project_out = load_parameters(
            lambda m: (m.weights, m.biases),
            module.project_out,
            (
                weights_dict[path / "project_out" / "weight"],
                weights_dict[path / "project_out" / "bias"],
            ),
        )
    else:
        project_out = None

    return load_parameters(
        lambda m: (m.codebook, m.project_out),
        module,
        (codebook, project_out),
    )


def load_qwen3_tts_residual_vector_quantization(
    module: Qwen3TTSResidualVectorQuantization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSResidualVectorQuantization:
    layers = tuple(
        load_qwen3_tts_vector_quantization(
            layer,
            weights_dict,
            path / "layers" / idx,
        )
        for idx, layer in enumerate(module.layers)
    )

    return load_parameters(
        lambda m: (m.layers,),
        module,
        (layers,),
    )


def _load_rvq_output_projection_linear(
    module: FullPrecisionLinear,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> FullPrecisionLinear:
    weight = weights_dict[path / "weight"]
    weight = rearrange(weight, "out_ch in_ch 1 -> out_ch in_ch")
    if module.has_biases:
        bias = weights_dict[path / "bias"]
    else:
        bias = None
    return load_parameters(lambda m: (m.weights, m.biases), module, (weight, bias))


def load_qwen3_tts_residual_vector_quantizer(
    module: Qwen3TTSResidualVectorQuantizer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSResidualVectorQuantizer:
    rvq = load_qwen3_tts_residual_vector_quantization(module.rvq, weights_dict, path / "vq")

    if module.output_projection is None:
        output_projection = None
    else:
        output_projection = _load_rvq_output_projection_linear(
            module.output_projection,
            weights_dict,
            path / "output_proj",
        )

    return load_parameters(
        lambda m: (m.rvq, m.output_projection),
        module,
        (rvq, output_projection),
    )


def load_qwen3_tts_split_rvq(
    module: Qwen3TTSSplitResidualVectorQuantizer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSSplitResidualVectorQuantizer:
    rvq_first = load_qwen3_tts_residual_vector_quantizer(module.rvq_first, weights_dict, path / "rvq_first")
    rvq_rest = load_qwen3_tts_residual_vector_quantizer(module.rvq_rest, weights_dict, path / "rvq_rest")
    return load_parameters(
        lambda m: (m.rvq_first, m.rvq_rest),
        module,
        (rvq_first, rvq_rest),
    )


def load_qwen3_tts_pre_transformer_layer(
    module: Qwen3TTSPreTransformerLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSPreTransformerLayer:
    self_attn = load_attention(module.self_attn, weights_dict, path / "self_attn", reorder_q_proj_gate=False)
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp", "up_proj", "gate_proj", "down_proj")

    input_layernorm = load_parameters(
        lambda m: (m.scales,),
        module.input_layernorm,
        (weights_dict[path / "input_layernorm" / "weight"],),
    )

    post_attention_layernorm = load_parameters(
        lambda m: (m.scales,),
        module.post_attention_layernorm,
        (weights_dict[path / "post_attention_layernorm" / "weight"],),
    )

    self_attn_layer_scale = weights_dict[path / "self_attn_layer_scale" / "scale"]
    mlp_layer_scale = weights_dict[path / "mlp_layer_scale" / "scale"]

    return load_parameters(
        lambda m: (
            m.self_attn,
            m.mlp,
            m.input_layernorm,
            m.post_attention_layernorm,
            m.self_attn_layer_scale,
            m.mlp_layer_scale,
        ),
        module,
        (
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            self_attn_layer_scale,
            mlp_layer_scale,
        ),
    )


def load_qwen3_tts_pre_transformer(
    module: Qwen3TTSPreTransformer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Qwen3TTSPreTransformer:
    input_projection = load_parameters(
        lambda m: (m.weights, m.biases),
        module.input_projection,
        (
            weights_dict[path / "input_proj" / "weight"],
            weights_dict[path / "input_proj" / "bias"],
        ),
    )

    output_projection = load_parameters(
        lambda m: (m.weights, m.biases),
        module.output_projection,
        (
            weights_dict[path / "output_proj" / "weight"],
            weights_dict[path / "output_proj" / "bias"],
        ),
    )

    output_norm = load_parameters(
        lambda m: (m.scales,),
        module.output_norm,
        (weights_dict[path / "norm" / "weight"],),
    )

    layers = tuple(
        load_qwen3_tts_pre_transformer_layer(
            layer,
            weights_dict,
            path / "layers" / idx,
        )
        for idx, layer in enumerate(module.layers)
    )

    return load_parameters(
        lambda m: (m.input_projection, m.output_projection, m.output_norm, m.layers),
        module,
        (input_projection, output_projection, output_norm, layers),
    )


def _load_qwen3_tts_transformer(
    module: Transformer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Transformer:
    loaded_layers = tuple(
        load_transformer_layer(
            layer,
            weights_dict,
            path / "layers" / idx,
            path / "layers" / idx,
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
            "up_proj",
            "gate_proj",
            "down_proj",
            permute_conv=False,
            reorder_q_proj_gate=False,
        )
        for idx, layer in enumerate(module.layers)
    )

    output_norm = load_parameters(
        lambda m: (m.scales,),
        module.output_norm,
        (weights_dict[path / "norm" / "weight"],),
    )

    return load_parameters(
        lambda m: (m.layers, m.output_norm),
        module,
        (loaded_layers, output_norm),
    )


def load_qwen3_tts_text_decoder(
    module: Qwen3TTSTextDecoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath | None = None,
) -> Qwen3TTSTextDecoder:
    base_path = ParameterPath() if path is None else path

    codec_embedding = load_parameters(
        lambda m: (m.weights,),
        module.codec_embedding,
        (weights_dict[base_path / "talker" / "model" / "codec_embedding" / "weight"],),
    )
    text_embedding = load_parameters(
        lambda m: (m.weights,),
        module.text_embedding,
        (weights_dict[base_path / "talker" / "model" / "text_embedding" / "weight"],),
    )

    text_projection_fc1 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.text_projection_fc1,
        (
            weights_dict[base_path / "talker" / "text_projection" / "linear_fc1" / "weight"],
            weights_dict[base_path / "talker" / "text_projection" / "linear_fc1" / "bias"],
        ),
    )
    text_projection_fc2 = load_parameters(
        lambda m: (m.weights, m.biases),
        module.text_projection_fc2,
        (
            weights_dict[base_path / "talker" / "text_projection" / "linear_fc2" / "weight"],
            weights_dict[base_path / "talker" / "text_projection" / "linear_fc2" / "bias"],
        ),
    )

    talker_transformer = _load_qwen3_tts_transformer(
        module.talker_transformer,
        weights_dict,
        base_path / "talker" / "model",
    )
    codec_head = load_parameters(
        lambda m: (m.weights, m.biases),
        module.codec_head,
        (weights_dict[base_path / "talker" / "codec_head" / "weight"], None),
    )

    predictor_transformer = _load_qwen3_tts_transformer(
        module.predictor_transformer,
        weights_dict,
        base_path / "talker" / "code_predictor" / "model",
    )
    predictor_embeddings = tuple(
        load_parameters(
            lambda m: (m.weights,),
            embedding,
            (weights_dict[base_path / "talker" / "code_predictor" / "model" / "codec_embedding" / idx / "weight"],),
        )
        for idx, embedding in enumerate(module.predictor_embeddings)
    )
    predictor_heads = tuple(
        load_parameters(
            lambda m: (m.weights, m.biases),
            head,
            (
                weights_dict[base_path / "talker" / "code_predictor" / "lm_head" / idx / "weight"],
                None,
            ),
        )
        for idx, head in enumerate(module.predictor_heads)
    )

    if module.talker_to_predictor_projection is None:
        talker_to_predictor_projection = None
    else:
        projection_path = base_path / "talker" / "code_predictor" / "small_to_mtp_projection"
        if projection_path / "weight" in weights_dict:
            talker_to_predictor_projection = load_parameters(
                lambda m: (m.weights, m.biases),
                module.talker_to_predictor_projection,
                (
                    weights_dict[projection_path / "weight"],
                    weights_dict[projection_path / "bias"],
                ),
            )
        else:
            talker_to_predictor_projection = module.talker_to_predictor_projection

    return load_parameters(
        lambda m: (
            m.codec_embedding,
            m.text_embedding,
            m.text_projection_fc1,
            m.text_projection_fc2,
            m.talker_transformer,
            m.codec_head,
            m.predictor_transformer,
            m.predictor_embeddings,
            m.predictor_heads,
            m.talker_to_predictor_projection,
        ),
        module,
        (
            codec_embedding,
            text_embedding,
            text_projection_fc1,
            text_projection_fc2,
            talker_transformer,
            codec_head,
            predictor_transformer,
            predictor_embeddings,
            predictor_heads,
            talker_to_predictor_projection,
        ),
    )


def load_qwen3_tts_audio_decoder(
    module: Qwen3TTSAudioDecoder,
    weights_dict: Mapping[str, Array],
    path: ParameterPath | None = None,
) -> Qwen3TTSAudioDecoder:
    base_path = ParameterPath() if path is None else path

    quantizer = load_qwen3_tts_split_rvq(module.quantizer, weights_dict, base_path / "quantizer")
    pre_conv = load_causal_conv1d(module.pre_conv, weights_dict, base_path / "pre_conv" / "conv")
    pre_transformer = load_qwen3_tts_pre_transformer(
        module.pre_transformer,
        weights_dict,
        base_path / "pre_transformer",
    )

    upsample_blocks = tuple(
        load_upsampling_block(upsample_block, weights_dict, base_path / "upsample" / idx)
        for idx, upsample_block in enumerate(module.upsample_blocks)
    )

    dac_decoder = load_dac_decoder(
        module.dac_decoder,
        weights_dict,
        base_path / "decoder",
        load_activation=load_qwen3_tts_snake_beta,
        load_residual=load_qwen3_tts_residual_unit,
    )

    return load_parameters(
        lambda m: (
            m.quantizer,
            m.pre_conv,
            m.pre_transformer,
            m.upsample_blocks,
            m.dac_decoder,
        ),
        module,
        (
            quantizer,
            pre_conv,
            pre_transformer,
            upsample_blocks,
            dac_decoder,
        ),
    )
