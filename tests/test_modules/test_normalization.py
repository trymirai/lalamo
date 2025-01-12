import jax
import pytest
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import LAYERS_TO_TEST, QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_rms_norm(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index].input_layernorm
    fs_layer = fartsovka_llama.layers[layer_index].attention_norm
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        operation_name="rms_norm",
    )


def test_out_norm_executorch(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = executorch_llama.norm
    fs_layer = fartsovka_qlora_llama.output_norm
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        rtol=QUANTIZED_RTOL,
        operation_name="out_norm",
    )
