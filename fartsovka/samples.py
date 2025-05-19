from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Int, Bool, Array, PRNGKeyArray

from fartsovka.common import ParameterDict
from fartsovka.modules.rope import PositionalEmbeddings, RoPE

__all__ = [
    "ModuleSample",
    "ModuleSamplesContext",
    "DecoderSamplesContext"
]


@dataclass
class ModuleSample:
    inputs: tuple[Array, ...]
    outputs: tuple[Array, ...]

    def export(self) -> ParameterDict:
        return ParameterDict(
            inputs=self.inputs,
            outputs=self.outputs
        )


class ModuleSamplesContext:
    pass


class DecoderSamplesContext(ModuleSamplesContext):
    seed: int
    root_key: PRNGKeyArray

    suffix_length: int
    token_ids: Int[Array, " suffix_tokens"]
    token_positions: Int[Array, " suffix_tokens"]
    mask: Bool[Array, "suffix_tokens total_tokens"]

    positional_embeddings: PositionalEmbeddings

    def __init__(self, seed: int, suffix_length: int, rope: RoPE):
        jax.config.update("jax_enable_x64", True)

        key = jax.random.PRNGKey(seed)
        tokens_key, root_key = jax.random.split(key, num=2)

        self.seed = seed
        self.root_key = root_key

        self.suffix_length = suffix_length
        self.token_ids = jax.random.randint(
            tokens_key,
            (suffix_length,),
            minval=0,
            maxval=10000,
            dtype=int
        ).astype(jnp.uint64)
        self.token_positions = jnp.arange(suffix_length).astype(jnp.uint64)
        self.mask = jnp.tril(jnp.ones((suffix_length, suffix_length), dtype=bool))

        self.positional_embeddings = rope(self.token_positions)
