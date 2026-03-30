"""Stub text decoder for NanoCodec TTS pipeline.

Generates random integer codes instead of real text-to-semantic decoding.
This is a placeholder until a real text decoder is implemented for NanoCodec.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from lalamo.modules.common import Initializer
from lalamo.sampling import SamplingPolicy

__all__ = ["StubTextDecoder", "StubTextDecoderConfig"]


@dataclass(frozen=True)
class StubTextDecoderConfig(TTSTextDecoderConfigBase):
    num_codebooks: int
    codebook_size: int
    precision: DTypeLike

    def init(self, initializer: Initializer) -> "StubTextDecoder":  # noqa: ARG002
        return StubTextDecoder(
            seed=123,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            precision=self.precision,
        )


class StubTextDecoder(TTSTextDecoder):
    seed: int = eqx.field(static=True)
    num_codebooks: int = eqx.field(static=True)
    codebook_size: int = eqx.field(static=True)
    precision: DTypeLike = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.precision

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch sequence"],
        sampling_policy: SamplingPolicy | None = None,  # noqa: ARG002
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, "batch num_codebooks tokens"]:
        """Generate random codebook indices with length derived from input tokens."""
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        batch_size = text_tokens.shape[0]
        output_length = text_tokens.shape[1]

        return jax.random.randint(
            key,
            shape=(batch_size, self.num_codebooks, output_length),
            minval=0,
            maxval=self.codebook_size,
            dtype=jnp.int32,
        )

    def export_weights(self) -> ParameterTree[Array]:
        return {"seed": jnp.array(self.seed)}

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["seed"], Array)
        return replace(
            self,
            seed=weights["seed"],
        )
