from collections.abc import Callable
from typing import NamedTuple

import jax
import pytest

# import transformers # No longer needed directly in this file
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

# Decoder, ETTransformer are no longer needed as direct imports for test signatures
from fartsovka.modules import DecoderLayer, QLoRALinear
from tests.executorch_llama.source_transformation.lora import Int8DynActInt4WeightLinearLoRA
from tests.executorch_llama.source_transformation.lora import Int8DynActInt4WeightLinearLoRA as ETQLoRALinear

# ETTransformer is no longer needed as direct import for test signatures
from tests.executorch_llama.transformer import TransformerBlock as ETDecoderLayer

from .common import (
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)

# Import LayerPair for type hinting the new fixture
from .conftest import LayerPair


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
    fs_decoder_layer: DecoderLayer,
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
    fs_decoder_layer: DecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_out_proj = fs_decoder_layer.attention.out_projection
    et_out_projs = [et_decoder_layer.attention.wo]
    permutations = [no_permutation]
    names = ["output projection"]
    assert isinstance(et_out_projs[0], ETQLoRALinear)
    return LinearDebugInfo(fs_out_proj, et_out_projs, permutations, names)  # type: ignore


def mlp_up_projs(
    fs_decoder_layer: DecoderLayer,
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
    fs_decoder_layer: DecoderLayer,
    et_decoder_layer: ETDecoderLayer,
) -> LinearDebugInfo:
    fs_mlp_down_proj = fs_decoder_layer.mlp.down_projection
    et_mlp_down_projs = [et_decoder_layer.feed_forward.w2]
    permutations = [no_permutation]
    names = ["down projection"]
    return LinearDebugInfo(fs_mlp_down_proj, et_mlp_down_projs, permutations, names)  # type: ignore


type LinearExtractor = Callable[[DecoderLayer, ETDecoderLayer], LinearDebugInfo]

LINEAR_EXTRACTORS = [
    attention_qkvs,
    attention_out_proj,
    mlp_up_projs,
    mlp_down_proj,
]


def test_linear(
    layers: LayerPair,
    rng_key: PRNGKeyArray,
) -> None:
    # layers.reference_layer is the HF/reference model's full DecoderLayer (e.g., LlamaDecoderLayer)
    # layers.fartsovka_layer is the Fartsovka DecoderLayer
    hf_layer = layers.reference_layer.mlp.down_proj  # type: ignore
    fs_layer = layers.fartsovka_layer.mlp.down_projection
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


def test_k_proj(
    layers: LayerPair,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = layers.reference_layer.self_attn.k_proj  # type: ignore
    fs_layer = layers.fartsovka_layer.attention.qkv_projection
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = hf_layer.in_features

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, (_, fs_output, _) = fs_layer_forward(sample_input)  # fs_output is the key projection
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
    )


def test_linear_with_bias(
    layers: LayerPair,
    # current_hf_model: transformers.PreTrainedModel, # Keep if skip logic is needed
    rng_key: PRNGKeyArray,
) -> None:
    # This test might require model-specific logic if bias handling differs significantly.
    # For example, Qwen2 has bias, Llama 3 does not in these layers.
    # The test will currently run for all models; consider skipping for non-Qwen if needed.
    # if "qwen" not in current_hf_model.config.model_type.lower(): # current_hf_model would be needed here
    #     pytest.skip("Skipping bias test for non-Qwen model")

    hf_layer = layers.reference_layer.mlp.down_proj  # type: ignore
    fs_layer = layers.fartsovka_layer.mlp.down_projection
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


@pytest.mark.parametrize("linear_extractor", LINEAR_EXTRACTORS)
def test_group_quantized_linear(
    qlora_layers: LayerPair,
    rng_key: PRNGKeyArray,
    linear_extractor: LinearExtractor,
) -> None:
    head_dim = qlora_layers.fartsovka_layer.attention.head_dim

    # qlora_layers.fartsovka_layer is a DecoderLayer from fartsovka_qlora_llama
    # qlora_layers.reference_layer is an ETDecoderLayer from executorch_llama
    fs_sub_linear_layer, et_sub_linear_layers, permutations, names = linear_extractor(
        qlora_layers.fartsovka_layer,
        qlora_layers.reference_layer,  # type: ignore
    )
    input_dim = fs_sub_linear_layer.input_dim  # fs_sub_linear_layer is a QLoRALinear

    fs_only_quantized = checkify_forward(super(QLoRALinear, fs_sub_linear_layer).__call__)

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    err, fs_outputs_only_quantized = fs_only_quantized(sample_input)
    err.throw()

    for fs_output, et_layer_ref, permutation, name in zip(
        fs_outputs_only_quantized,
        et_sub_linear_layers,  # This is a list of ETQLoRALinear
        permutations,
        names,
        strict=True,
    ):
        permuted_fs_output = permutation(fs_output, head_dim)
        # et_layer_ref is an ETQLoRALinear (Int8DynActInt4WeightLinearLoRA)
        et_only_quantized = super(Int8DynActInt4WeightLinearLoRA, et_layer_ref).forward
        et_output_only_quantized = from_torch(et_only_quantized(sample_input_torch).squeeze(0))
        assert_close(
            result=permuted_fs_output,
            reference=et_output_only_quantized,
            rtol=QUANTIZED_RTOL,
            operation_name=name,
        )


@pytest.mark.parametrize("linear_extractor", LINEAR_EXTRACTORS)
def test_qlora_linear(
    qlora_layers: LayerPair,
    rng_key: PRNGKeyArray,
    linear_extractor: LinearExtractor,
) -> None:
    head_dim = qlora_layers.fartsovka_layer.attention.head_dim

    # fs_sub_linear_layer is a QLoRALinear
    fs_sub_linear_layer, et_sub_linear_layers, permutations, names = linear_extractor(
        qlora_layers.fartsovka_layer,
        qlora_layers.reference_layer,  # type: ignore
    )
    input_dim = fs_sub_linear_layer.input_dim

    fs_forward = checkify_forward(fs_sub_linear_layer.__call__)  # Call the QLoRALinear layer

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    err, fs_outputs = fs_forward(sample_input)
    err.throw()

    # et_sub_linear_layers contains ETQLoRALinear instances
    for fs_output, et_layer_ref, permutation, name in zip(
        fs_outputs, et_sub_linear_layers, permutations, names, strict=True
    ):
        permuted_fs_output = permutation(fs_output, head_dim)
        et_output = from_torch(et_layer_ref(sample_input_torch).squeeze(0))  # Call the ETQLoRALinear layer
        assert_close(
            result=permuted_fs_output,
            reference=et_output,
            rtol=QUANTIZED_RTOL,
            operation_name=name,
        )
