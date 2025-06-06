import jax
import pytest  # For pytest.skip
import transformers  # For hf_model type hint
from jaxtyping import PRNGKeyArray

from fartsovka.modules import Decoder  # For fartsovka_model type hint
from tests.executorch_llama.transformer import Transformer as ETTransformer  # For test_quantized_embedding

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch


def test_embedding(
    hf_model: transformers.PreTrainedModel,
    fartsovka_model: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    # Attempt to access the standard embedding layer in HF CausalLM models
    hf_embedding_layer = getattr(getattr(hf_model, "model", hf_model), "embed_tokens", None)
    if hf_embedding_layer is None:
        # Some models (like base T5Model) might have embeddings directly under hf_model.encoder.embed_tokens
        # or hf_model.shared. For this test, we assume a CausalLM-like structure.
        pytest.skip(
            f"Model {hf_model.config.model_type} does not have a standard .model.embed_tokens or .embed_tokens attribute.",
        )

    fs_embedding_module = fartsovka_model.embedding
    fs_embed_fn = checkify_forward(fs_embedding_module.embed)
    fs_readout_fn = checkify_forward(fs_embedding_module.readout)

    vocab_size = fs_embedding_module.vocab_size
    model_dim = fs_embedding_module.model_dim

    # Split key for distinct random operations
    rng_key_tokens, rng_key_sample_hidden = jax.random.split(rng_key)

    token_ids = jax.random.randint(rng_key_tokens, (64,), 0, vocab_size)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Test embedding
    hf_embedded_output = from_torch(hf_embedding_layer(token_ids_torch).squeeze(0))
    err, fs_embedded_output = fs_embed_fn(token_ids)
    err.throw()
    assert_close(
        result=fs_embedded_output,
        reference=hf_embedded_output,
        operation_name=f"embedding_for_{hf_model.config.model_type}",
    )

    # Test readout (output projection, often tied to lm_head weights)
    # CausalLM models usually have an lm_head. Base models might not.
    if not hasattr(hf_model, "lm_head"):
        # Some models (like base LlamaModel without LlamaForCausalLM) might not have lm_head.
        # Executorch models might have a direct 'output' linear layer.
        # For generic HF models, if lm_head is missing, we skip this part of the test.
        pytest.skip(f"Model {hf_model.config.model_type} does not have an lm_head for readout test.")

    hf_lm_head_layer = hf_model.lm_head

    sample_hidden_state = jax.random.normal(rng_key_sample_hidden, (model_dim,))
    sample_hidden_state_torch = to_torch(sample_hidden_state).unsqueeze(0)  # Add batch dim

    hf_readout_output = from_torch(hf_lm_head_layer(sample_hidden_state_torch).squeeze(0))
    err, fs_readout_output = fs_readout_fn(sample_hidden_state)
    err.throw()
    assert_close(
        result=fs_readout_output,
        reference=hf_readout_output,
        operation_name=f"readout_for_{hf_model.config.model_type}",
    )


def test_quantized_embedding(  # This test remains unchanged
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    fs_layer = fartsovka_qlora_llama.embedding
    fs_embed = checkify_forward(fs_layer.embed)
    fs_readout = checkify_forward(fs_layer.readout)
    et_tok_embeddings_layer = executorch_llama.tok_embeddings
    et_output_layer = executorch_llama.output

    vocab_size = fs_layer.vocab_size
    model_dim = fs_layer.model_dim

    # Split key for distinct random operations
    rng_key_tokens, rng_key_sample_hidden = jax.random.split(rng_key)

    token_ids = jax.random.randint(rng_key_tokens, (64,), 0, vocab_size)
    token_ids_torch = to_torch(token_ids).unsqueeze(0)

    # Test embedding
    et_embedded_output = from_torch(et_tok_embeddings_layer(token_ids_torch).squeeze(0))
    err, fs_embedded_output = fs_embed(token_ids)
    err.throw()
    assert_close(
        result=fs_embedded_output,
        reference=et_embedded_output,
        rtol=QUANTIZED_RTOL,
        operation_name="quantized_embedding",
    )

    # Test readout (weight transpose)
    sample_hidden_state = jax.random.normal(rng_key_sample_hidden, (model_dim,))
    sample_hidden_state_torch = to_torch(sample_hidden_state).unsqueeze(0)  # Add batch dim

    et_readout_output = from_torch(et_output_layer(sample_hidden_state_torch).squeeze(0))
    err, fs_readout_output = fs_readout(sample_hidden_state)
    err.throw()
    assert_close(
        result=fs_readout_output,
        reference=et_readout_output,
        rtol=QUANTIZED_RTOL,
        operation_name="quantized_readout",
    )
