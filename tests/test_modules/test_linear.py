import jax
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
from tests.executorch_llama.source_transformation.lora import Int8DynActInt4WeightLinearLoRA
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import QUANTIZED_ATOL, assert_close, checkify_forward, from_torch, to_torch


def test_linear(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_llama.model.layers[0].mlp.down_proj
    fs_layer = fartsovka_llama.layers[0].mlp.down_projection
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = hf_layer.in_features

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, (fs_output,) = fs_layer_forward(sample_input)
    err.throw()
    assert_close(hf_output, fs_output)


def test_group_quantized_linear(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
) -> None:
    fs_layer = fartsovka_qlora_llama.layers[0].mlp.down_projection
    et_layer = executorch_llama.layers[0].feed_forward.w2
    input_dim = et_layer.in_features

    fs_only_quantized = checkify_forward(fs_layer.quantized_linear)
    et_only_quantized = super(Int8DynActInt4WeightLinearLoRA, et_layer).forward  # type: ignore

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    et_output_only_quantized = from_torch(et_only_quantized(sample_input_torch).squeeze(0))
    err, (fs_output_only_quantized,) = fs_only_quantized(sample_input)
    err.throw()
    assert_close(fs_output_only_quantized, et_output_only_quantized, atol=QUANTIZED_ATOL)


def test_qlora_linear(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
) -> None:
    fs_layer = fartsovka_qlora_llama.layers[0].mlp.down_projection
    fs_layer_forward = checkify_forward(fs_layer)
    et_layer = executorch_llama.layers[0].feed_forward.w2
    input_dim = et_layer.in_features

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    et_output = from_torch(et_layer(sample_input_torch).squeeze(0))
    err, (fs_output,) = fs_layer_forward(sample_input)
    err.throw()
    assert_close(fs_output, et_output, atol=QUANTIZED_ATOL)
