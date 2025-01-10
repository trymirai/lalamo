import jax
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import assert_close, checkify_forward, from_torch, to_torch


def test_mlp(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_llama.model.layers[0].mlp
    fs_layer = fartsovka_llama.layers[0].mlp
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input)
    err.throw()
    assert_close(hf_output, fs_output)


def test_qlora_mlp(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
) -> None:
    fs_layer = fartsovka_qlora_llama.layers[0].mlp
    fs_layer_forward = checkify_forward(fs_layer)
    et_layer = executorch_llama.layers[0].feed_forward
    input_dim = fs_layer.model_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    et_output = from_torch(et_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input)
    err.throw()
    assert_close(fs_output, et_output)
