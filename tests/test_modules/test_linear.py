from collections.abc import Callable
from itertools import product
from typing import NamedTuple

import jax
import pytest
import transformers
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRADecoderLayer, QLoRALlama
from fartsovka.models.qwen2 import Qwen2
from fartsovka.modules.linear import QLoRALinear
from tests.executorch_llama.source_transformation.lora import Int8DynActInt4WeightLinearLoRA
from tests.executorch_llama.source_transformation.lora import Int8DynActInt4WeightLinearLoRA as ETQLoRALinear
from tests.executorch_llama.transformer import Transformer as ETTransformer
from tests.executorch_llama.transformer import TransformerBlock as ETDecoderLayer

from .common import (
    LAYERS_TO_TEST,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)


def permute_queries_keys(x: Float[Array, " channels"], head_dim: int) -> Float[Array, " channels"]:
    num_rotors = head_dim // 2
    return rearrange(
        x,
        "(heads reim rotors) -> (heads rotors reim)",
        reim=2,
        rotors=num_rotors,
    )


def no_permutation(x: Float[Array, " channels"], num_heads: int) -> Float[Array, " channels"]:  # noqa: ARG001
    return x


class LinearDebugInfo(NamedTuple):
    fs_layer: QLoRALinear
    et_layers: list[ETQLoRALinear]
    permutations: list[Callable[[Float[Array, " channels"], int], Float[Array, " channels"]]]
    names: list[str]


def attention_qkvs(
    fs_decoder_layer: QLoRADecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_qkv = fs_decoder_layer.attention.qkv_projection
    et_qkvs = [
        et_decoder_layer.attention.wq,
        et_decoder_layer.attention.wk,
        et_decoder_layer.attention.wv,
    ]
    permutations = [
        permute_queries_keys,
        permute_queries_keys,
        no_permutation,
    ]
    names = ["query projection", "key projection", "value projection"]
    return LinearDebugInfo(fs_qkv, et_qkvs, permutations, names)  # type: ignore


def attention_out_proj(
    fs_decoder_layer: QLoRADecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_out_proj = fs_decoder_layer.attention.out_projection
    et_out_projs = [et_decoder_layer.attention.wo]
    permutations = [no_permutation]
    names = ["output projection"]
    assert isinstance(et_out_projs[0], ETQLoRALinear)
    return LinearDebugInfo(fs_out_proj, et_out_projs, permutations, names)  # type: ignore


def mlp_up_projs(
    fs_decoder_layer: QLoRADecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_mlp_up_proj = fs_decoder_layer.mlp.up_projection
    et_mlp_up_projs = [et_decoder_layer.feed_forward.w3, et_decoder_layer.feed_forward.w1]
    permutations = [
        no_permutation,
        no_permutation,
    ]
    names = ["up projection", "gate projection"]
    return LinearDebugInfo(fs_mlp_up_proj, et_mlp_up_projs, permutations, names)  # type: ignore


def mlp_down_proj(
    fs_decoder_layer: QLoRADecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_mlp_down_proj = fs_decoder_layer.mlp.down_projection
    et_mlp_down_projs = [et_decoder_layer.feed_forward.w2]
    permutations = [no_permutation]
    names = ["down projection"]
    return LinearDebugInfo(fs_mlp_down_proj, et_mlp_down_projs, permutations, names)  # type: ignore


type LinearExtractor = Callable[[QLoRADecoderLayer, ETDecoderLayer], LinearDebugInfo]

LINEAR_EXTRACTORS = [
    attention_qkvs,
    attention_out_proj,
    mlp_up_projs,
    mlp_down_proj,
]


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_linear(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index].mlp.down_proj
    fs_layer = fartsovka_llama.layers[layer_index].mlp.down_projection
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = hf_layer.in_features

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, (fs_output,) = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
    )


def test_linear_with_bias(
    huggingface_qwen25: transformers.Qwen2Model,
    fartsovka_qwen25: Qwen2,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_qwen25.model.layers[0].mlp.down_proj
    fs_layer = fartsovka_qwen25.layers[0].mlp.down_projection
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = hf_layer.in_features

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, (fs_output,) = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
    )


@pytest.mark.parametrize(
    ["layer_index", "linear_extractor"],
    list(product(LAYERS_TO_TEST, LINEAR_EXTRACTORS)),
)
def test_group_quantized_linear(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
    linear_extractor: LinearExtractor,
) -> None:
    head_dim = fartsovka_qlora_llama.layers[layer_index].attention.head_dim

    fs_layer, et_layers, permutations, names = linear_extractor(
        fartsovka_qlora_llama.layers[layer_index],
        executorch_llama.layers[layer_index],  # type: ignore
    )
    input_dim = fs_layer.input_dim

    fs_only_quantized = checkify_forward(super(QLoRALinear, fs_layer).__call__)

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    err, fs_outputs_only_quantized = fs_only_quantized(sample_input)
    err.throw()

    for fs_output, et_layer, permutation, name in zip(
        fs_outputs_only_quantized,
        et_layers,
        permutations,
        names,
        strict=True,
    ):
        permuted_fs_output = permutation(fs_output, head_dim)
        et_only_quantized = super(Int8DynActInt4WeightLinearLoRA, et_layer).forward  # type: ignore
        et_output_only_quantized = from_torch(et_only_quantized(sample_input_torch).squeeze(0))
        assert_close(
            result=permuted_fs_output,
            reference=et_output_only_quantized,
            rtol=QUANTIZED_RTOL,
            operation_name=name,
        )


@pytest.mark.parametrize(
    ["layer_index", "linear_extractor"],
    list(product(LAYERS_TO_TEST, LINEAR_EXTRACTORS)),
)
def test_qlora_linear(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
    linear_extractor: LinearExtractor,
) -> None:
    head_dim = fartsovka_qlora_llama.layers[layer_index].attention.head_dim

    fs_layer, et_layers, permutations, names = linear_extractor(
        fartsovka_qlora_llama.layers[layer_index],
        executorch_llama.layers[layer_index],  # type: ignore
    )
    input_dim = fs_layer.input_dim

    fs_forward = checkify_forward(fs_layer.__call__)

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    err, fs_outputs = fs_forward(sample_input)
    err.throw()

    for fs_output, et_layer, permutation, name in zip(fs_outputs, et_layers, permutations, names, strict=True):
        permuted_fs_output = permutation(fs_output, head_dim)
        et_output = from_torch(et_layer(sample_input_torch).squeeze(0))
        assert_close(
            result=permuted_fs_output,
            reference=et_output,
            rtol=QUANTIZED_RTOL,
            operation_name=name,
        )
