from itertools import accumulate, pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from einops import rearrange
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)

from lalamo.model_import.loaders.huggingface import load_ngram_injection
from lalamo.module import LogicalAxis
from lalamo.modules import StreamOpContext, TransformerForwardPassConfig
from lalamo.modules.ngram_injection import multiply_limbs, remainder_of_limbs
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config
from tests.unit.modules.qwen4_exp import (
    HIDDEN_SIZE,
    PLE_EMBED_DIM,
    as_stream,
    as_token_ids,
    float_weights,
    load_decoder_layer,
    make_gated_residual_config,
    make_hf_config,
    make_hyper_input,
    make_initializer,
    make_keychain,
    make_ngram_injection_config,
    merge_streams,
    randomized,
)

CHUNK_LENGTHS = (9, 1, 4, 1, 1)


def make_token_ids(eos_token_id: int) -> torch.Tensor:
    torch.manual_seed(2)
    token_ids = torch.randint(0, 128, (2, sum(CHUNK_LENGTHS)))
    token_ids[0, [0, 4, 5, 11]] = eos_token_id
    token_ids[1, [7, 9, 15]] = eos_token_id
    return token_ids


def test_ngram_ids_match_hf() -> None:
    hf_config = make_hf_config(with_ple=True)
    hf_embedding = Qwen4ExpTextNGramEmbedding(hf_config, PLE_EMBED_DIM, layer_idx=0, ple_layer_index=1)
    hash_config = make_ngram_injection_config(hf_config, ple_layer_index=1).hash_config
    with torch.no_grad():
        num_rows, _ = hf_embedding.ngram_embedding.weight.shape
        hf_embedding.ngram_embedding.weight[:, 0] = torch.arange(num_rows)
    token_ids = make_token_ids(hash_config.eos_token_id)
    num_context_tokens = hash_config.ngram_size - 1

    with torch.no_grad():
        embeddings = hf_embedding(token_ids, past_key_values=None)
    reference = rearrange(
        embeddings, "batch tokens (heads channels) -> batch tokens heads channels", heads=hash_config.num_heads
    )[..., 0]

    token_history = jnp.full((2, num_context_tokens), hash_config.eos_token_id, dtype=jnp.int32)
    result = jax.vmap(hash_config.ngram_ids)(
        jnp.concatenate([token_history, jnp.asarray(token_ids.numpy(), dtype=jnp.int32)], axis=1),
    )

    np.testing.assert_array_equal(np.asarray(result[:, num_context_tokens:]), reference.numpy().astype(np.int64))


def test_limb_arithmetic_matches_python_integers() -> None:
    vocab_size = 248_320
    largest_multiplier = (2**63 - 1) // vocab_size
    multipliers = (largest_multiplier, 27_014_362_213_041)
    divisors = (20_000_003, 2**25 - 1, 7)
    rng = np.random.default_rng(0)
    first_tokens, second_tokens = rng.integers(0, vocab_size, size=(2, 64))
    first_tokens[:2] = (0, vocab_size - 1)
    second_tokens[:2] = (vocab_size - 1, vocab_size - 1)

    limbs = multiply_limbs(jnp.asarray(first_tokens, dtype=jnp.uint32), multipliers[0], 14, 5) ^ multiply_limbs(
        jnp.asarray(second_tokens, dtype=jnp.uint32),
        multipliers[1],
        14,
        5,
    )
    result = remainder_of_limbs(limbs, jnp.asarray(divisors, dtype=jnp.uint32), 14, 7)

    expected = [
        [((int(first) * multipliers[0]) ^ (int(second) * multipliers[1])) % divisor for divisor in divisors]
        for first, second in zip(first_tokens, second_tokens, strict=True)
    ]
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected, dtype=np.uint64))


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("num_padding_tokens", [0, 3])
def test_ngram_injection_matches_hf_across_chunks(num_padding_tokens: int) -> None:
    hf_config = make_hf_config(with_ple=True)
    hf_ple = randomized(Qwen4ExpTextPLELayer(hf_config, layer_idx=0, ple_layer_index=0))
    ngram_injection_config = make_ngram_injection_config(hf_config, ple_layer_index=0)
    module = load_ngram_injection(
        ngram_injection_config.init(make_initializer(), make_gated_residual_config(), HIDDEN_SIZE),
        float_weights(hf_ple),
        ParameterPath(),
    )
    hyper_input = make_hyper_input(sum(CHUNK_LENGTHS))
    token_ids = make_token_ids(ngram_injection_config.hash_config.eos_token_id)
    padding_input = torch.randn(2, num_padding_tokens, hyper_input.shape[-1])
    padding_ids = torch.randint(0, 128, (2, num_padding_tokens))
    cache = DynamicCache(config=hf_config)

    references = []
    results = []
    state = None
    for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
        with torch.no_grad():
            injection = hf_ple(hyper_input[:, start:end], token_ids[:, start:end], cache)
        references.append(hyper_input[:, start:end] + injection)
        result, state = module(
            as_stream(torch.cat([hyper_input[:, start:end], padding_input[:, :num_padding_tokens]], dim=1)),
            StreamOpContext(
                token_ids=as_token_ids(
                    torch.cat([token_ids[:, start:end], padding_ids[:, :num_padding_tokens]], dim=1)
                ),
                per_layer_input=None,
                lengths_without_padding=jax.device_put(
                    jnp.full((2,), end - start, dtype=jnp.int32),
                    make_test_sharding_config().resolve_sharding((LogicalAxis.BATCH,)),
                ),
            ),
            state,
            keychain=make_keychain(),
        )
        results.append(merge_streams(result)[:, : end - start])

    assert_close(result=jnp.concatenate(results, axis=1), reference=torch_to_jax(torch.cat(references, dim=1)))


@pytest.mark.usefixtures("fake_mesh")
def test_decoder_layer_with_ngram_injection_matches_hf_across_chunks() -> None:
    hf_config = make_hf_config(with_ple=True)
    hf_layer = randomized(Qwen4ExpTextDecoderLayer(hf_config, layer_idx=0))
    ngram_injection_config = make_ngram_injection_config(hf_config, ple_layer_index=0)
    layer = load_decoder_layer(hf_config, float_weights(hf_layer), ngram_injection_config)
    hyper_input = make_hyper_input(sum(CHUNK_LENGTHS))
    token_ids = make_token_ids(ngram_injection_config.hash_config.eos_token_id)
    cache = DynamicCache(config=hf_config)

    references = []
    results = []
    state = None
    for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
        with torch.no_grad():
            references.append(
                hf_layer(
                    hyper_input[:, start:end],
                    position_embeddings=None,
                    past_key_values=cache,
                    ple_input_ids=token_ids[:, start:end],
                ),
            )
        result = layer(
            as_stream(hyper_input[:, start:end]),
            positional_embeddings=None,
            state=state,
            return_updated_state=True,
            forward_pass_config=TransformerForwardPassConfig.for_tracer_tests(),
            token_ids=as_token_ids(token_ids[:, start:end]),
            keychain=make_keychain(),
        )
        state = result.updated_state
        results.append(merge_streams(result.outputs))

    assert_close(result=jnp.concatenate(results, axis=1), reference=torch_to_jax(torch.cat(references, dim=1)))
