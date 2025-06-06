from copy import deepcopy

import equinox as eqx
import pytest
import torch
import transformers  # For hf_model type hint
from jax import numpy as jnp

# from jaxtyping import PRNGKeyArray # Only if rng_key fixture is used for random elements
from fartsovka.modules import Decoder  # For fartsovka_model type hint
from tests.executorch_llama.transformer import Transformer as ETTransformer

# MAX_TOKEN_INDEX is not used if we standardize position_ids to jnp.arange
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


def test_generic_decoder(
    hf_model: transformers.PreTrainedModel,  # Provided by model_test_pair
    fartsovka_model: Decoder,  # Provided by model_test_pair
    # rng_key: PRNGKeyArray, # Add if needed for e.g. random position_ids
) -> None:
    fs_decoder_forward = checkify_forward(fartsovka_model)

    sequence_length = len(TOKENS)
    token_ids_jax = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids_jax).unsqueeze(0)

    # Standardize position_ids to jnp.arange for this generic test.
    position_ids_jax = jnp.arange(sequence_length, dtype=jnp.int32)
    position_ids_torch = to_torch(position_ids_jax).unsqueeze(0)

    # Create causal mask - standard for auto-regressive models.
    # Some models (like older Gemma) might infer causality from position_ids alone
    # and not strictly require an attention_mask for causal language modeling.
    torch_attention_mask = torch.triu(
        torch.ones((1, sequence_length, sequence_length), dtype=hf_model.dtype) * torch.finfo(hf_model.dtype).min,
        diagonal=1,
    )  # type: ignore
    # Ensure the mask is expanded to match the expected [batch_size, num_heads, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
    # This varies by model. A common shape for causal mask is [1, 1, seq_len, seq_len]
    if hf_model.config.model_type in ["llama", "qwen2", "gemma2", "gemma"]:  # Common models
        final_torch_attention_mask = torch_attention_mask[:, None, :, :]  # Add head dim: [1, 1, seq_len, seq_len]
    else:  # Default or fallback if model needs different mask shape
        final_torch_attention_mask = torch_attention_mask

    jax_attention_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Call Hugging Face model
    # The exact arguments can vary. Most models accept input_ids, attention_mask, position_ids.
    try:
        # Common signature for many causal LM models
        hf_outputs = hf_model(
            input_ids=token_ids_torch,
            attention_mask=final_torch_attention_mask,
            position_ids=position_ids_torch,
            return_dict=True,
        )
    except TypeError:
        # Fallback for models that might not use 'attention_mask' or prefer different arg names
        # This might happen with some base models or older model versions
        try:
            hf_outputs = hf_model(
                input_ids=token_ids_torch,
                position_ids=position_ids_torch,
                return_dict=True,
            )
        except (
            TypeError
        ):  # Fallback to simplest possible call, assuming model infers mask or doesn't need position_ids
            hf_outputs = hf_model(input_ids=token_ids_torch, return_dict=True)

    # Assuming the primary output (last hidden state) is what we need
    if hasattr(hf_outputs, "last_hidden_state"):
        torch_pre_softmax = hf_outputs.last_hidden_state
    elif isinstance(hf_outputs, tuple) and len(hf_outputs) > 0 and isinstance(hf_outputs[0], torch.Tensor):
        torch_pre_softmax = hf_outputs[0]
    elif isinstance(hf_outputs, torch.Tensor):  # Direct tensor output
        torch_pre_softmax = hf_outputs
    else:
        raise ValueError(f"Unexpected Hugging Face model output type: {type(hf_outputs)}")

    hf_output_final = from_torch(torch_pre_softmax.squeeze(0))

    err, fs_output = fs_decoder_forward(token_ids_jax, position_ids_jax, mask=jax_attention_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=hf_output_final,
        atol=DECODER_ATOL,
        rtol=DECODER_RTOL,
        operation_name=f"generic_decoder_test_for_{hf_model.config.model_type}",
    )


# QLoRA tests remain specific as they use different fixtures and test truncation
@pytest.mark.parametrize("num_layers_in_truncated_model", NUM_LAYERS_IN_TRUNCATED_MODELS)
def test_qlora_decoder_truncated(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    num_layers_in_truncated_model: int,
) -> None:
    fs_decoder_big = fartsovka_qlora_llama
    if num_layers_in_truncated_model > len(fs_decoder_big.layers):  # type: ignore
        pytest.skip(f"Requested {num_layers_in_truncated_model} layers, model has {len(fs_decoder_big.layers)}")  # type: ignore

    fs_decoder = eqx.tree_at(lambda d: d.layers, fs_decoder_big, fs_decoder_big.layers[:num_layers_in_truncated_model])  # type: ignore
    fs_decoder_forward = checkify_forward(fs_decoder)

    et_decoder = deepcopy(executorch_llama)
    if num_layers_in_truncated_model > len(et_decoder.layers):  # type: ignore
        pytest.skip(f"Requested {num_layers_in_truncated_model} layers for ET, model has {len(et_decoder.layers)}")  # type: ignore
    et_decoder.layers = et_decoder.layers[:num_layers_in_truncated_model]  # type: ignore

    sequence_length = len(TOKENS)
    token_ids_jax = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids_jax).unsqueeze(0)

    position_ids_jax = jnp.arange(sequence_length, dtype=jnp.int32)
    jax_attention_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Executorch model's forward might be direct, not taking many args
    et_output_logits = et_decoder(tokens=token_ids_torch)
    et_output_tensor = et_output_logits[0] if isinstance(et_output_logits, tuple) else et_output_logits
    et_output_final = from_torch(et_output_tensor.squeeze(0))

    err, fs_output = fs_decoder_forward(token_ids_jax, position_ids_jax, mask=jax_attention_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=et_output_final,
        atol=QUANTIZED_DECODER_ATOL,
        rtol=QUANTIZED_RTOL,
        operation_name=f"qlora_decoder_truncated_{num_layers_in_truncated_model}_layers",
    )


def test_qlora_decoder(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
) -> None:
    fs_decoder_forward = checkify_forward(fartsovka_qlora_llama)
    et_decoder = executorch_llama

    sequence_length = len(TOKENS)
    token_ids_jax = jnp.array(TOKENS)
    token_ids_torch = to_torch(token_ids_jax).unsqueeze(0)

    position_ids_jax = jnp.arange(sequence_length, dtype=jnp.int32)
    jax_attention_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    et_output_logits = et_decoder(tokens=token_ids_torch)
    et_output_tensor = et_output_logits[0] if isinstance(et_output_logits, tuple) else et_output_logits
    et_output_final = from_torch(et_output_tensor.squeeze(0))

    err, fs_output = fs_decoder_forward(token_ids_jax, position_ids_jax, mask=jax_attention_mask)
    err.throw()
    assert_close(
        result=fs_output.output,
        reference=et_output_final,
        atol=QUANTIZED_DECODER_ATOL,
        rtol=QUANTIZED_RTOL,
        operation_name="qlora_decoder_full",
    )
