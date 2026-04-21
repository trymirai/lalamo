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
from lalamo.modules.audio.text_decoder import (
    CodebookCodes,
    TTSDecodingContext,
    TTSTextDecoder,
    TTSTextDecoderConfigBase,
)
from lalamo.sampling import SamplingPolicy

__all__ = ["StubTextDecoder", "StubTextDecoderConfig"]


@dataclass(frozen=True)
class StubTextDecoderConfig(TTSTextDecoderConfigBase):
    num_codebooks: int
    codebook_size: int
    precision: DTypeLike

    @classmethod
    def format_instruction(cls, style: str) -> str | None:  # noqa: ARG003
        return None

    def empty(self) -> "StubTextDecoder":
        return StubTextDecoder(config=self, seed=123)

    def random_init(self, *, key: PRNGKeyArray) -> "StubTextDecoder":  # noqa: ARG002
        return StubTextDecoder(config=self, seed=123)


class StubTextDecoder(TTSTextDecoder["StubTextDecoderConfig"]):
    seed: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch sequence"],
        *,
        context: TTSDecodingContext,  # noqa: ARG002
        sampling_policy: SamplingPolicy | None = None,  # noqa: ARG002
        key: PRNGKeyArray,
    ) -> CodebookCodes:
        batch_size = text_tokens.shape[0]
        output_length = text_tokens.shape[1]
        num_semantic = 1
        num_acoustic = self.config.num_codebooks - num_semantic

        key_semantic, key_acoustic = jax.random.split(key)
        return CodebookCodes(
            semantic=jax.random.randint(
                key_semantic,
                shape=(batch_size, num_semantic, output_length),
                minval=0,
                maxval=self.config.codebook_size,
                dtype=jnp.int32,
            ),
            acoustic=jax.random.randint(
                key_acoustic,
                shape=(batch_size, num_acoustic, output_length),
                minval=0,
                maxval=self.config.codebook_size,
                dtype=jnp.int32,
            ),
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
