import jax
import pytest
import torch
import transformers  # For hf_model type hint
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention

# Decoder and ETTransformer are needed for specific QLoRA test fixtures and parent model type hints
from fartsovka.modules import Decoder, KVCacheLayerSlice
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import (
    MAX_TOKEN_INDEX,
    QUANTIZED_ATOL,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)

# Import LayerPair for type hinting the new fixture
from .conftest import LayerPair


def test_attention_no_mask_no_cache(
    layers: LayerPair,  # Fixture iterated_layers_data yields LayerPair
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    # layers.reference_layer is the HF/reference model's full layer (e.g., LlamaDecoderLayer)
    hf_attention_layer = layers.reference_layer.self_attn  # type: ignore
    if hf_model.config.model_type == "llama":
        assert isinstance(hf_attention_layer, LlamaAttention)

    fs_attention_layer = layers.fartsovka_layer.attention
    fs_layer_forward = checkify_forward(fs_attention_layer)

    input_dim = fs_attention_layer.model_dim
    sequence_length = 512

    # Split rng_key for distinct random operations
    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)

    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    # Attempt to get RoPE from hf_model.model.rotary_emb
    hf_rotary_emb_module = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rotary_emb_module is None:
        pytest.skip(f"Could not find rotary_emb for HF model {hf_model.config.name_or_path}")
    cos, sin = hf_rotary_emb_module(sample_input_torch, position_ids_torch)  # type: ignore

    torch_zero_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=sample_input_torch.dtype)
    positional_embeddings = fartsovka_model.global_rope(position_ids)

    hf_output = from_torch(
        hf_attention_layer(sample_input_torch, position_embeddings=(cos, sin), attention_mask=torch_zero_mask)[
            0
        ].squeeze(0),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,  # Assuming same for this test
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


def test_gemma3_attention(
    layers: LayerPair,  # Fixture iterated_layers_data yields LayerPair
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    if "gemma" not in hf_model.config.model_type.lower():  # type: ignore
        pytest.skip("Skipping Gemma-specific attention test for non-Gemma models.")

    hf_attention_layer = layers.reference_layer.self_attn  # type: ignore
    fs_attention_layer = layers.fartsovka_layer.attention
    fs_layer_forward = checkify_forward(fs_attention_layer)

    input_dim = fs_attention_layer.model_dim
    sequence_length = 64

    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)
    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    hf_rotary_emb_module = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rotary_emb_module is None:
        pytest.skip(f"Could not find rotary_emb for HF model {hf_model.config.name_or_path}")
    cos, sin = hf_rotary_emb_module(sample_input_torch, position_ids_torch)  # type: ignore

    global_positional_embeddings = fartsovka_model.global_rope(position_ids)

    if hasattr(fartsovka_model, "local_rope"):  # Gemma specific
        local_positional_embeddings = fartsovka_model.local_rope(position_ids)  # type: ignore
    else:
        local_positional_embeddings = global_positional_embeddings

    torch_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=sample_input_torch.dtype)
        * torch.finfo(sample_input_torch.dtype).min,
        diagonal=1,
    )
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    hf_output = from_torch(
        hf_attention_layer(sample_input_torch, position_embeddings=(cos, sin), attention_mask=torch_mask)[0].squeeze(
            0
        ),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=global_positional_embeddings,
        local_positional_embeddings=local_positional_embeddings,
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
        atol=2e-3,
    )


def test_attention_with_mask(
    layers: LayerPair,  # Fixture iterated_layers_data yields LayerPair
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    hf_attention_layer = layers.reference_layer.self_attn  # type: ignore
    if hf_model.config.model_type == "llama":
        assert isinstance(hf_attention_layer, LlamaAttention)
    fs_attention_layer = layers.fartsovka_layer.attention
    fs_layer_forward = checkify_forward(fs_attention_layer)

    input_dim = fs_attention_layer.model_dim
    sequence_length = 512

    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)
    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    hf_rotary_emb_module = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rotary_emb_module is None:
        pytest.skip(f"Could not find rotary_emb for HF model {hf_model.config.name_or_path}")
    cos, sin = hf_rotary_emb_module(sample_input_torch, position_ids_torch)  # type: ignore

    positional_embeddings = fartsovka_model.global_rope(position_ids)

    torch_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=sample_input_torch.dtype)
        * torch.finfo(sample_input_torch.dtype).min,
        diagonal=1,
    )
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    hf_output = from_torch(
        hf_attention_layer(sample_input_torch, attention_mask=torch_mask, position_embeddings=(cos, sin))[0].squeeze(
            0
        ),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,  # Assuming same for this test
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


def test_attention_with_mask_and_kv_cache(
    layers: LayerPair,  # Fixture iterated_layers_data yields LayerPair
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    hf_attention_layer = layers.reference_layer.self_attn  # type: ignore
    if hf_model.config.model_type == "llama":
        assert isinstance(hf_attention_layer, LlamaAttention)
    fs_attention_layer = layers.fartsovka_layer.attention
    fs_layer_forward = checkify_forward(fs_attention_layer)

    input_dim = fs_attention_layer.model_dim
    head_dim = fs_attention_layer.head_dim
    num_groups = fs_attention_layer.num_groups
    sequence_length = 32
    prefix_length = 256

    rng_key_input, rng_key_keys, rng_key_values, rng_key_pos_ids = jax.random.split(rng_key, 4)

    sample_input = jax.random.normal(rng_key_input, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    kv_cache_slice = KVCacheLayerSlice(
        keys=jax.random.normal(rng_key_keys, (prefix_length, num_groups, head_dim)),
        values=jax.random.normal(rng_key_values, (prefix_length, num_groups, head_dim)),
    )

    torch_keys = to_torch(kv_cache_slice.keys).permute(1, 0, 2).unsqueeze(0)
    torch_values = to_torch(kv_cache_slice.values).permute(1, 0, 2).unsqueeze(0)
    torch_cache = DynamicCache()
    # Using 0 for layer_idx as DynamicCache is fresh for this specific layer test and past_key_value
    # in HF attention layers is often indexed starting from 0 for the current layer's cache.
    torch_cache.update(torch_keys, torch_values, layer_idx=0, cache_kwargs=None)  # type: ignore

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    position_ids_torch = to_torch(position_ids).unsqueeze(0)

    hf_rotary_emb_module = getattr(getattr(hf_model, "model", hf_model), "rotary_emb", None)
    if hf_rotary_emb_module is None:
        pytest.skip(f"Could not find rotary_emb for HF model {hf_model.config.name_or_path}")
    cos, sin = hf_rotary_emb_module(sample_input_torch, position_ids_torch)  # type: ignore

    positional_embeddings = fartsovka_model.global_rope(position_ids)

    torch_mask_diag = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=sample_input_torch.dtype)
        * torch.finfo(sample_input_torch.dtype).min,
        diagonal=1,
    )
    torch_mask_prefix = torch.zeros((sequence_length, prefix_length), dtype=sample_input_torch.dtype)
    torch_mask = torch.cat([torch_mask_prefix, torch_mask_diag], dim=1)
    torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)

    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))
    jax_mask_prefix = jnp.ones((sequence_length, prefix_length), dtype=bool)
    jax_mask = jnp.concatenate([jax_mask_prefix, jax_mask], axis=1)

    hf_output = from_torch(
        hf_attention_layer(
            sample_input_torch,
            attention_mask=torch_mask,
            position_embeddings=(cos, sin),
            past_key_value=torch_cache,
        )[0].squeeze(0),
    )
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,  # Assuming same for this test
        mask=jax_mask,
        kv_cache=kv_cache_slice,
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=hf_output,
    )


def test_qlora_attention(
    qlora_layers: LayerPair,  # Fixture qlora_iterated_layers_data yields LayerPair
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    fs_attention_layer = qlora_layers.fartsovka_layer.attention
    fs_layer_forward = checkify_forward(fs_attention_layer)
    et_attention_layer = qlora_layers.reference_layer.attention  # type: ignore

    input_dim = fs_attention_layer.model_dim
    sequence_length = 10

    rng_key_sample, rng_key_pos_ids = jax.random.split(rng_key)
    sample_input = jax.random.normal(rng_key_sample, (sequence_length, input_dim))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    position_ids = jax.random.randint(rng_key_pos_ids, (sequence_length,), minval=0, maxval=MAX_TOKEN_INDEX)
    positional_embeddings = fartsovka_qlora_llama.global_rope(position_ids)
    et_freqs_cos, et_freqs_sin = executorch_llama.rope.get_freqs(None, sequence_length)

    if not (
        hasattr(fs_attention_layer, "head_dim")
        and isinstance(fs_attention_layer.head_dim, int)
        and fs_attention_layer.head_dim > 0
        and fs_attention_layer.head_dim % 2 == 0
    ):
        pytest.skip(f"Invalid head_dim for layer: {getattr(fs_attention_layer, 'head_dim', 'N/A')}")

    freqs_cos = to_torch(positional_embeddings.cosines[:, : fs_attention_layer.head_dim // 2])
    assert et_freqs_cos.shape == freqs_cos.shape, (
        f"Shape mismatch for cos: ET {et_freqs_cos.shape}, FS {freqs_cos.shape}"
    )
    freqs_sin = to_torch(positional_embeddings.sines[:, : fs_attention_layer.head_dim // 2])
    assert et_freqs_sin.shape == freqs_sin.shape, (
        f"Shape mismatch for sin: ET {et_freqs_sin.shape}, FS {freqs_sin.shape}"
    )

    jax_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

    et_output = from_torch(et_attention_layer(sample_input_torch, freqs_cos, freqs_sin).squeeze(0))
    err, fs_output = fs_layer_forward(
        sample_input,
        global_positional_embeddings=positional_embeddings,
        local_positional_embeddings=positional_embeddings,  # Assuming same for this test
        mask=jax_mask,
    )
    err.throw()
    assert_close(
        result=fs_output.attention_output,
        reference=et_output,
        atol=QUANTIZED_ATOL,
        rtol=QUANTIZED_RTOL,
    )
