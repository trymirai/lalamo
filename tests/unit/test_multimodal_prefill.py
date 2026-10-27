"""Prefill and generation with caller-provided embeddings and (multi-axis) positions."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import DecoderForwardPassConfig, MRoPEConfig, MRoPELayout
from tests.common import assert_close, tolerance
from tests.helpers import build_tiny_attention_decoder_config

SHARDING = ShardingConfig.replicated(jax.devices("cpu")[:1])
PROMPT_LENGTH = 11


def _model(*, use_mrope: bool) -> LanguageModel:
    decoder_config = build_tiny_attention_decoder_config((None, None))
    if use_mrope:
        layers = tuple(
            replace(
                layer,
                rope_config=MRoPEConfig(
                    base=10_000.0,
                    max_sequence_length=64,
                    head_dim=4,
                    sections=(1, 1, 0),
                    layout=MRoPELayout.CHUNKED,
                ),
            )
            for layer in decoder_config.transformer_config.layer_configs
        )
        decoder_config = replace(
            decoder_config,
            transformer_config=replace(decoder_config.transformer_config, layer_configs=layers),
        )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(temperature=0.0),
    )
    tokenizer = Tokenizer(WordLevel({f"t{index}": index for index in range(32)}, unk_token="t0"))
    return config.init(tokenizer, RandomInitializer(jnp.float32, SHARDING, key=jax.random.key(3)))


def _on_model_mesh(values: jax.Array) -> jax.Array:
    return jax.device_put(values, SHARDING.make_sharding((None,) * values.ndim))


def _token_ids() -> jax.Array:
    return _on_model_mesh(jax.random.randint(jax.random.key(1), (1, PROMPT_LENGTH), 0, 32, dtype=jnp.int32))


def _keychain() -> Keychain:
    return Keychain.init(0, sharding_config=SHARDING)


def _embed(model: LanguageModel, token_ids: jax.Array) -> jax.Array:
    return model.decoder.embedding.embed(token_ids, keychain=_keychain())


def test_explicit_defaults_reproduce_implicit_prefill() -> None:
    model = _model(use_mrope=False)
    token_ids = _token_ids()
    implicit = model.prefill_tokens(token_ids, 32, keychain=_keychain())
    explicit = model.prefill_tokens(
        token_ids,
        32,
        input_embeddings=_embed(model, token_ids),
        token_positions=jnp.arange(PROMPT_LENGTH, dtype=jnp.int32)[None, :],
        keychain=_keychain(),
    )
    assert jnp.array_equal(explicit.last_token_indices, implicit.last_token_indices)
    assert_close(result=explicit.last_token_logits, reference=implicit.last_token_logits)

    implicit_tokens = model.generate_tokens(token_ids, max_output_length=6, keychain=_keychain()).token_ids
    explicit_tokens = model.generate_tokens(
        token_ids,
        max_output_length=6,
        prompt_embeddings=_embed(model, token_ids),
        prompt_positions=jnp.arange(PROMPT_LENGTH, dtype=jnp.int32)[None, :],
        keychain=_keychain(),
    ).token_ids
    assert jnp.array_equal(explicit_tokens, implicit_tokens)


def test_chunked_prefill_with_spliced_embeddings_and_mrope_positions() -> None:
    model = _model(use_mrope=True)
    token_ids = _token_ids()
    # float32 throughout: in bfloat16, chunking alone changes logits by ~0.3% even for plain text.
    forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    # Pretend tokens 3..6 are an image: foreign embeddings and a 2x2 spatial layout at t=3.
    embeddings = model.decoder.embedding.embed(
        token_ids,
        forward_pass_config=forward_pass_config.embedding_forward_pass_config,
        keychain=_keychain(),
    )
    image_features = _on_model_mesh(
        jax.random.normal(jax.random.key(5), (1, 4, embeddings.shape[-1]), dtype=embeddings.dtype),
    )
    embeddings = embeddings.at[:, 3:7].set(image_features)
    text_before = jnp.repeat(jnp.arange(3)[:, None], 3, axis=1)
    image = jnp.asarray([[3, 3, 3], [3, 3, 4], [3, 4, 3], [3, 4, 4]])
    text_after = jnp.repeat(jnp.arange(5, 5 + PROMPT_LENGTH - 7)[:, None], 3, axis=1)
    positions = _on_model_mesh(jnp.concatenate([text_before, image, text_after])[None].astype(jnp.int32))

    reference = model.decoder(
        token_ids,
        positions,
        input_embeddings=embeddings,
        forward_pass_config=forward_pass_config,
        keychain=_keychain(),
    ).logits[:, -1]
    prefilled = model.prefill_tokens(
        token_ids,
        32,
        forward_pass_config=forward_pass_config,
        chunk_size=4,
        input_embeddings=embeddings,
        token_positions=positions,
        keychain=_keychain(),
    )
    with tolerance(atol=1e-5, rtol=1e-5):
        assert_close(result=prefilled.last_token_logits, reference=reference, operation_name="Chunked prefill")
    # Decoding continues after the largest position of the last token, not after the token count.
    assert prefilled.last_token_indices.tolist() == [5 + PROMPT_LENGTH - 7 - 1]


def test_explicit_positions_reject_prefix_lengths() -> None:
    model = _model(use_mrope=False)
    token_ids = _token_ids()
    with pytest.raises(ValueError, match="prefix_lengths"):
        model.prefill_tokens(
            token_ids,
            32,
            token_positions=jnp.arange(PROMPT_LENGTH, dtype=jnp.int32)[None, :],
            prefix_lengths=jnp.zeros((1,), dtype=jnp.int32),
            keychain=_keychain(),
        )
