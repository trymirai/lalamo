"""Stub text decoder for NanoCodec TTS pipeline.

Generates random integer codes instead of real text-to-semantic decoding.
This is a placeholder until a real text decoder is implemented for NanoCodec.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Int, PRNGKeyArray

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
            config=self,
            seed=123,
        )


class StubTextDecoder(TTSTextDecoder[StubTextDecoderConfig]):
    seed: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

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
