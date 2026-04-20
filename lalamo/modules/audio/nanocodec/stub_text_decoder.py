"""Stub text decoder for NanoCodec TTS pipeline.

Generates random integer codes instead of real text-to-semantic decoding.
This is a placeholder until a real text decoder is implemented for NanoCodec.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Key

from lalamo.initializer import Initializer
from lalamo.module import field
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfig
from lalamo.sampling import SamplingPolicy

__all__ = ["StubTextDecoder", "StubTextDecoderConfig"]


@dataclass(frozen=True)
class StubTextDecoderConfig(TTSTextDecoderConfig):
    num_codebooks: int
    codebook_size: int

    def init(self, initializer: Initializer) -> "StubTextDecoder":  # noqa: ARG002
        return StubTextDecoder(
            config=self,
            seed=123,
        )


class StubTextDecoder(TTSTextDecoder[StubTextDecoderConfig]):
    seed: int = field(static=True)

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch sequence"],
        sampling_policy: SamplingPolicy | None = None,  # noqa: ARG002
        *,
        key: Key[Array, ""],
        dequant_key: Key[Array, ""],  # noqa: ARG002
    ) -> Int[Array, "batch num_codebooks tokens"]:
        """Generate random codebook indices with length derived from input tokens."""
        key = jax.random.fold_in(key, self.seed)

        batch_size = text_tokens.shape[0]
        output_length = text_tokens.shape[1]

        return jax.random.randint(
            key,
            shape=(batch_size, self.config.num_codebooks, output_length),
            minval=0,
            maxval=self.config.codebook_size,
            dtype=jnp.int32,
        )
