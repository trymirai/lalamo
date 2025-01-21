import jax
import pytest
import torch
import transformers
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.models.llama.modeling_llama import DynamicCache, LlamaAttention

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
from fartsovka.modules.kv_cache import KVCacheLayerSlice
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import (
    LAYERS_TO_TEST,
    QUANTIZED_ATOL,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_attention_no_mask_no_cache(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index].self_attn
    assert isinstance(hf_layer, LlamaAttention)
    fs_layer = fartsovka_llama.layers[layer_index].attention
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim
    sequence_length = 512

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)
    # We create a zero mask to ensure that the attention is not affected by the mask
    torch_zero_mask = torch.zeros((1, 1, sequence_length, sequence_length))
    positional_embeddings = fartsovka_llama.rope(position_ids)

    # Run forward passes
    hf_output = from_torch(
        hf_layer(sample_input_torch, position_embeddings=(cos, sin), attention_mask=torch_zero_mask)[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings)
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_attention_with_mask(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index].self_attn
    assert isinstance(hf_layer, LlamaAttention)
    fs_layer = fartsovka_llama.layers[layer_index].attention
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim
    sequence_length = 512

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)
    positional_embeddings = fartsovka_llama.rope(position_ids)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    hf_output = from_torch(
        hf_layer(sample_input_torch, attention_mask=torch_mask, position_embeddings=(cos, sin))[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_attention_with_mask_and_kv_cache(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    hf_layer = huggingface_llama.model.layers[layer_index].self_attn
    assert isinstance(hf_layer, LlamaAttention)
    fs_layer = fartsovka_llama.layers[layer_index].attention
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.model_dim
    head_dim = fs_layer.head_dim
    num_groups = fs_layer.num_groups
    sequence_length = 32
    prefix_length = 256

    input_key, keys_key, values_key = jax.random.split(rng_key, 3)

    sample_input = jax.random.normal(input_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Create KV cache
    kv_cache = KVCacheLayerSlice(
        keys=jax.random.normal(keys_key, (prefix_length, num_groups, head_dim)),
        values=jax.random.normal(values_key, (prefix_length, num_groups, head_dim)),
    )

    torch_keys = to_torch(kv_cache.keys).permute(1, 0, 2).unsqueeze(0)
    torch_values = to_torch(kv_cache.values).permute(1, 0, 2).unsqueeze(0)
    torch_cache = DynamicCache(len(huggingface_llama.model.layers))
    torch_cache.update(torch_keys, torch_values, layer_index, dict())
    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)
    cos, sin = huggingface_llama.model.rotary_emb(sample_input_torch, position_ids_torch)
    positional_embeddings = fartsovka_llama.rope(position_ids)

    # Create causal mask
    torch_mask = torch.triu(torch.ones((sequence_length, sequence_length)) * float("-inf"), diagonal=1)
    torch_mask = torch.concatenate([torch.zeros(sequence_length, prefix_length), torch_mask], dim=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)

    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))
    jax_mask = jnp.concatenate([jnp.ones((sequence_length, prefix_length), dtype=bool), jax_mask], axis=1)

    # Run forward passes
    hf_output = from_torch(
        hf_layer(
            sample_input_torch,
            attention_mask=torch_mask,
            position_embeddings=(cos, sin),
            past_key_value=torch_cache,
        )[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        positional_embeddings=positional_embeddings,
        mask=jax_mask,
        kv_cache=kv_cache,
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


@pytest.mark.parametrize("layer_index", LAYERS_TO_TEST)
def test_qlora_attention(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
    layer_index: int,
) -> None:
    fs_layer = fartsovka_qlora_llama.layers[layer_index].attention
    fs_layer_forward = checkify_forward(fs_layer)
    et_layer = executorch_llama.layers[layer_index].attention

    input_dim = fs_layer.model_dim
    sequence_length = 10

    sample_input = jax.random.normal(rng_key, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    # Get positional embeddings
    position_ids = jnp.arange(sequence_length)
    positional_embeddings = fartsovka_qlora_llama.rope(position_ids)

    et_freqs_cos, et_freqs_sin = executorch_llama.rope.get_freqs(None, sequence_length)
    freqs_cos = to_torch(positional_embeddings.cosines[:, : fs_layer.head_dim // 2])
    assert et_freqs_cos.shape == freqs_cos.shape
    freqs_sin = to_torch(positional_embeddings.sines[:, : fs_layer.head_dim // 2])
    assert et_freqs_sin.shape == freqs_sin.shape

    # Create causal mask
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    # Run forward passes
    et_output = from_torch(et_layer(sample_input_torch, freqs_cos, freqs_sin).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input, positional_embeddings=positional_embeddings, mask=jax_mask)
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=et_output,
        atol=QUANTIZED_ATOL,
        rtol=QUANTIZED_RTOL,
    )
