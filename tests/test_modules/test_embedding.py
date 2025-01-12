import jax
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch


def test_embedding(
    huggingface_llama: transformers.LlamaModel,
    fartsovka_llama: BaselineLlama,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = huggingface_llama.model.embed_tokens
    fs_layer = fartsovka_llama.embedding
    fs_embed = checkify_forward(fs_layer.embed)
    fs_readout = checkify_forward(fs_layer.readout)

    vocab_dim = fs_layer.vocab_dim
    model_dim = fs_layer.model_dim

    # Generate random token IDs
    token_key, _ = jax.random.split(rng_key)
    token_ids = jax.random.randint(token_key, (64,), 0, vocab_dim)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Test embedding
    hf_output = from_torch(hf_layer(token_ids_torch).squeeze(0))
    err, fs_output = fs_embed(token_ids)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        operation_name="embedding",
    )

    # Test readout (weight transpose)
    sample_input = jax.random.normal(rng_key, (model_dim,))
    sample_input_torch = to_torch(sample_input)
    hf_output = from_torch(huggingface_llama.lm_head(sample_input_torch.unsqueeze(0)).squeeze(0))
    err, fs_output = fs_readout(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        operation_name="readout",
    )


def test_quantized_embedding(
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: QLoRALlama,
    rng_key: PRNGKeyArray,
) -> None:
    fs_layer = fartsovka_qlora_llama.embedding
    fs_embed = checkify_forward(fs_layer.embed)
    fs_readout = checkify_forward(fs_layer.readout)
    et_layer = executorch_llama.tok_embeddings

    vocab_dim = fs_layer.vocab_dim
    model_dim = fs_layer.model_dim

    # Generate random token IDs
    token_key, _ = jax.random.split(rng_key)
    token_ids = jax.random.randint(token_key, (64,), 0, vocab_dim)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Test embedding
    et_output = from_torch(et_layer(token_ids_torch).squeeze(0))
    err, fs_output = fs_embed(token_ids)
    err.throw()
    assert_close(
        result=fs_output,
        reference=et_output,
        rtol=QUANTIZED_RTOL,
        operation_name="embedding",
    )

    # Test readout (weight transpose)
    sample_input = jax.random.normal(rng_key, (model_dim,))
    sample_input_torch = to_torch(sample_input)
    et_output = from_torch(executorch_llama.output(sample_input_torch.unsqueeze(0)).squeeze(0))
    err, fs_output = fs_readout(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=et_output,
        rtol=QUANTIZED_RTOL,
        operation_name="readout",
    )
