import jax
import jax.numpy as jnp
from jax.experimental.checkify import checkify
from jaxtyping import PRNGKeyArray

from lalamo.modules import Decoder


def generate(decoder: Decoder, tokens: list[int], num_steps: int = 15, key: PRNGKeyArray | None = None) -> list[int]:
    if key is None:
        key = jax.random.PRNGKey(0)
    # decoder = jit(decoder)
    num_tokens = len(tokens)
    token_ids = jnp.array(tokens, dtype=jnp.int32)
    token_positions = jnp.arange(num_tokens, dtype=jnp.int32)
    mask = jnp.tril(jnp.ones((num_tokens, num_tokens), dtype=jnp.bool))
    err, decoder_outputs = checkify(decoder)(
        token_ids=token_ids,
        token_positions=token_positions,
        kv_cache=None,
        mask=mask,
        return_updated_kv_cache=True,
    )
    err.throw()
    result = []
    for i in range(num_steps):
        assert key is not None
        current_key, key = jax.random.split(key)
        next_token = jax.random.categorical(current_key, decoder_outputs.logits[-1])
        result.append(next_token.item())
        token_ids = next_token[None]
        token_positions = jnp.array([num_tokens + i])
        mask = None

        err, decoder_outputs = checkify(decoder)(
            token_ids=token_ids,
            token_positions=token_positions,
            kv_cache=decoder_outputs.updated_kv_cache,
            mask=mask,
            return_updated_kv_cache=True,
        )
        err.throw()

    return result


# %%
from tokenizers import Tokenizer

from lalamo.generation.generator import generate
from lalamo.model_import import REPO_TO_MODEL, import_model

tokenizer = Tokenizer.from_pretrained("google/gemma-3-1b-it")

decoder = import_model(REPO_TO_MODEL["google/gemma-3-1b-it"]).model

tokens = tokenizer.encode("The purpose of a system is")

tokenizer.decode(generate(decoder, tokens.ids, num_steps=15))
