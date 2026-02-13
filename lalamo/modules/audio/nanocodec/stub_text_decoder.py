"""Stub text decoder for NanoCodec TTS pipeline.

Generates random integer codes instead of real text-to-semantic decoding.
This is a placeholder until a real text decoder is implemented for NanoCodec.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from lalamo.sampling import SamplingPolicy

__all__ = ["StubTextDecoder", "StubTextDecoderConfig"]


@dataclass(frozen=True)
class StubTextDecoderConfig(TTSTextDecoderConfigBase):
    num_codebooks: int
    codebook_size: int
    precision: DTypeLike

    def empty(self) -> "StubTextDecoder":
        return StubTextDecoder(config=self, seed=123)

    def random_init(self, *, key: PRNGKeyArray) -> "StubTextDecoder":  # noqa: ARG002
        return StubTextDecoder(config=self, seed=123)


class StubTextDecoder(TTSTextDecoder["StubTextDecoderConfig"]):
    """Stub text decoder that generates random codebook indices.

    Instead of converting text tokens to semantic codes via a transformer,
    this stub generates random integer indices suitable for NanoCodec decoding.
    The output length is derived from the input text_tokens sequence length.

    Output shape: [batch, num_codebooks, sequence_length]
    """

    seed: int

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch sequence"],
        sampling_policy: SamplingPolicy | None = None,  # noqa: ARG002
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, "batch num_codebooks tokens"]:
        """Generate random codebook indices with length derived from input tokens.

        Args:
            text_tokens: Input token array of shape [batch, sequence].
                         The sequence length determines the output length.
            sampling_policy: Ignored. Present for interface compatibility.
            key: PRNG key for random generation. Uses default seed if None.

        Returns:
            Random integer indices of shape [batch, num_codebooks, sequence_length].
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        batch_size = text_tokens.shape[0]
        output_length = text_tokens.shape[1]

        return jax.random.randint(
            key,
            shape=(batch_size, self.config.num_codebooks, output_length),
            minval=0,
            maxval=self.config.codebook_size,
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
