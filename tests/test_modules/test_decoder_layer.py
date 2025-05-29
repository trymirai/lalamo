import jax
import pytest
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from fartsovka.modules import Decoder
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import (
    LAYERS_TO_TEST,
    MAX_TOKEN_INDEX,
    QUANTIZED_ATOL,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_decoder_layer(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: Decoder,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index]  # type: ignore
    assert isinstance(hf_layer, LlamaDecoderLayer)
    fs_layer = fartsovka_llama.layers[layer_index]
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.mlp.up_projection.input_dim
    sequence_length = 64

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    rng_key = jax.random.PRNGKey(0)
    position_ids = jax.random.randint(rng_key, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)  # type: ignore
    positional_embeddings = fartsovka_llama.global_rope(position_ids)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    hf_output = from_torch(
        hf_layer(sample_input_torch, attention_mask=torch_mask, position_embeddings=(cos, sin))[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=hf_output,
    )


# GEMMA_ATOL = 0.05  # TODO(norpadon): Figure out why decoder layer error is so large
# GEMMA_RTOL = 0.05


# @pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
# def test_gemma2_decoder_layer(
#     huggingface_gemma2: transformers.GemmaModel,
#     fartsovka_gemma2: Gemma2Decoder,
#     rng_key: PRNGKeyArray,
#     layer_index: int,
# ) -> None:
#     hf_layer = huggingface_gemma2.model.layers[layer_index]
#     fs_layer = fartsovka_gemma2.layers[layer_index]
#     fs_layer_forward = checkify_forward(fs_layer)

#     input_dim = fs_layer.mlp.up_projection.input_dim
#     sequence_length = 64

#     sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
#     sample_input_torch = to_torch(sample_input).unsqueeze(0)

#     # Get positional embeddings
#     position_ids = jnp.arange(sequence_length)
#     position_ids_torch = to_torch(position_ids).unsqueeze(0)
#     cos, sin = huggingface_gemma2.model.rotary_emb(sample_input_torch, position_ids_torch)
#     positional_embeddings = fartsovka_gemma2.rope(position_ids)

#     # Create causal mask
#     torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
#     torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
#     jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

#     # Run forward passes
#     hf_output = from_torch(
#         hf_layer(sample_input_torch, attention_mask=torch_mask, position_embeddings=(cos, sin))[0].squeeze(0),
#     )
#     err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings, mask=jax_mask)
#     err.throw()
#     assert_close(
#         result=fs_output.output,
#         reference=hf_output,
#         atol=GEMMA_ATOL,
#         rtol=GEMMA_RTOL,
#     )


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_qlora_decoder_layer(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    fs_layer = fartsovka_qlora_llama.layers[layer_index]
    fs_layer_forward = checkify_forward(fs_layer)
    et_layer = executorch_llama.layers[layer_index]

    input_dim = fs_layer.mlp.up_projection.input_dim
    sequence_length = 64

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    rng_key = jax.random.PRNGKey(0)
    position_ids = jax.random.randint(rng_key, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    freqs_cos, freqs_sin = executorch_llama.rope.get_freqs(None, sequence_length)
    positional_embeddings = fartsovka_qlora_llama.global_rope(position_ids)

    # Create causal mask
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    et_output = from_torch(et_layer(sample_input_torch, freqs_cos, freqs_sin).squeeze(0))
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=et_output,
        atol=QUANTIZED_ATOL,
        rtol=QUANTIZED_RTOL,
    )
