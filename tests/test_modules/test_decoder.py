from copy import deepcopy

import equinox as eqx
import pytest
import torch
import transformers
from jax import numpy as jnp

from fartsovka.modules import Decoder
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch

DECODER_ATOL = 3e-3
DECODER_RTOL = 0.05
QUANTIZED_DECODER_ATOL = 2.0  # LMAO


TOKENS = [
    128000,
    128000,
    128006,
    9125,
    128007,
    271,
    2675,
    527,
    264,
    11190,
    15592,
    18328,
    369,
    5944,
    10631,
    323,
    19075,
    128009,
    128006,
    882,
    128007,
    271,
    3923,
    649,
    499,
    1520,
    757,
    449,
    30,
    128009,
    128006,
    78191,
    128007,
]


NUM_LAYERS_IN_TRUNCATED_MODELS = [0, 1, 3, 7]


def test_llama(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: Decoder,
) -> None:
    fs_decoder = fartsovka_llama
    fs_decoder_forward = checkify_forward(fs_decoder)

    sequence_length = len(TOKENS)
    token_ids = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    torch_pre_softmax = huggingface_llama(
        token_ids_torch,
        attention_mask=torch_mask,
        position_ids=position_ids_torch,
    )[0]
    # Run forward passes
    hf_output = from_torch(torch_pre_softmax.squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=hf_output,
        atol=DECODER_ATOL,
        rtol=DECODER_RTOL,
    )


def test_qwen2(
    huggingface_qwen25: transformers.Qwen2Model,
    fartsovka_qwen25: Decoder,
) -> None:
    fs_decoder = fartsovka_qwen25
    fs_decoder_forward = checkify_forward(fs_decoder)

    sequence_length = len(TOKENS)
    token_ids = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    torch_pre_softmax = huggingface_qwen25(
        token_ids_torch,
        attention_mask=torch_mask,
        position_ids=position_ids_torch,
    )[0]
    # Run forward passes
    hf_output = from_torch(torch_pre_softmax.squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=hf_output,
        atol=DECODER_ATOL,
        rtol=DECODER_RTOL,
    )


def test_gemma2(
    huggingface_gemma2: transformers.Gemma2Model,
    fartsovka_gemma2: Decoder,
) -> None:
    fs_decoder = fartsovka_gemma2
    fs_decoder_forward = checkify_forward(fs_decoder)

    sequence_length = len(TOKENS)
    token_ids = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Create causal mask
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    torch_pre_softmax = huggingface_gemma2(
        token_ids_torch,
        position_ids=position_ids_torch,
    )[0]
    # Run forward passes
    hf_output = from_torch(torch_pre_softmax.squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=hf_output,
        atol=DECODER_ATOL,
        rtol=DECODER_RTOL,
    )


@pytest.mark.parametrize("num_layers_in_truncated_model", NUM_LAYERS_IN_TRUNCATED_MODELS)
def test_qlora_decoder_truncated(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    num_layers_in_truncated_model: int,
) -> None:
    fs_decoder_big = fartsovka_qlora_llama
    fs_decoder = eqx.tree_at(lambda d: d.layers, fs_decoder_big, fs_decoder_big.layers[:num_layers_in_truncated_model])
    fs_decoder_forward = checkify_forward(fs_decoder)

    et_decoder = deepcopy(executorch_llama)
    et_decoder.layers = et_decoder.layers[:num_layers_in_truncated_model]

    sequence_length = len(TOKENS)
    token_ids = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)

    # Create causal mask
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    et_output = from_torch(et_decoder(tokens=token_ids_torch).squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=et_output,
        atol=QUANTIZED_DECODER_ATOL,
        rtol=QUANTIZED_RTOL,
    )


def test_qlora_decoder(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
) -> None:
    fs_decoder = fartsovka_qlora_llama
    fs_decoder_forward = checkify_forward(fs_decoder)
    et_decoder = executorch_llama

    sequence_length = len(TOKENS)
    token_ids = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Create position IDs
    position_ids = jnp.arange(sequence_length)

    # Create causal mask
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    et_output = from_torch(et_decoder(tokens=token_ids_torch).squeeze(0))
    err, fs_output = fs_decoder_forward(token_ids, position_ids, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=et_output,
        atol=QUANTIZED_DECODER_ATOL,
        rtol=QUANTIZED_RTOL,
    )
