from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_mapping, require_tree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase
from lalamo.modules.audio.neutts.codec_modules import (
    NeuCodecLinear,
    NeuCodecLinearConfig,
    NeuCodecResidualFSQ,
    NeuCodecResidualFSQConfig,
    NeuCodecVocosDecoder,
    NeuCodecVocosDecoderConfig,
)
from lalamo.modules.audio.text_decoder import CodebookCodes


@dataclass(frozen=True)
class NeuCodecAudioDecoderConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike
    samplerate: int = 24_000
    codec_repo: str | None = None
    device: str | None = None
    levels: tuple[int, ...] = (4,) * 8
    num_quantizers: int = 1
    quantizer_output_dim: int = 2048
    hidden_dim: int = 1024
    depth: int = 12
    heads: int = 16
    rotary_dim: int = 64
    hop_length: int = 480

    @property
    def quantizer_config(self) -> NeuCodecResidualFSQConfig:
        return NeuCodecResidualFSQConfig(
            levels=self.levels,
            num_quantizers=self.num_quantizers,
            output_dim=self.quantizer_output_dim,
            precision=self.precision,
        )

    @property
    def fc_post_a_config(self) -> NeuCodecLinearConfig:
        return NeuCodecLinearConfig(
            input_dim=self.quantizer_output_dim,
            output_dim=self.hidden_dim,
            precision=self.precision,
        )

    @property
    def vocos_decoder_config(self) -> NeuCodecVocosDecoderConfig:
        return NeuCodecVocosDecoderConfig(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            heads=self.heads,
            rotary_dim=self.rotary_dim,
            hop_length=self.hop_length,
            precision=self.precision,
        )

    def empty(self) -> "NeuCodecAudioDecoder":
        return NeuCodecAudioDecoder(
            config=self,
            quantizer=self.quantizer_config.empty(),
            fc_post_a=self.fc_post_a_config.empty(),
            vocos_decoder=self.vocos_decoder_config.empty(),
        )

    def random_init(self, *, key: PRNGKeyArray) -> "NeuCodecAudioDecoder":  # noqa: ARG002
        return self.empty()


class NeuCodecAudioDecoder(TTSAudioDecoder[NeuCodecAudioDecoderConfig]):
    quantizer: NeuCodecResidualFSQ
    fc_post_a: NeuCodecLinear
    vocos_decoder: NeuCodecVocosDecoder

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizer": self.quantizer.export_weights(),
            "fc_post_a": self.fc_post_a.export_weights(),
            "vocos_decoder": self.vocos_decoder.export_weights(),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        weights = require_mapping(weights)
        quantizer_weights = weights["quantizer"]
        fc_post_a_weights = weights["fc_post_a"]
        vocos_decoder_weights = weights["vocos_decoder"]

        assert isinstance(quantizer_weights, Mapping)
        assert isinstance(fc_post_a_weights, Mapping)
        assert isinstance(vocos_decoder_weights, Mapping)

        return replace(
            self,
            quantizer=self.quantizer.import_weights(require_tree(quantizer_weights)),
            fc_post_a=self.fc_post_a.import_weights(require_tree(fc_post_a_weights)),
            vocos_decoder=self.vocos_decoder.import_weights(require_tree(vocos_decoder_weights)),
        )

    def encode_reference_audio(
        self, audio: Float[Array, " audio_samples"], samplerate: int
    ) -> Int[Array, " speech_tokens"]:
        raise NotImplementedError(
            "NeuTTS reference audio encoding requires a native encoder implementation and is not available in the "
            "native decoder-only path.",
        )

    def _semantic_indices(self, codes: Int[Array, "*shape"] | CodebookCodes) -> Int[Array, "*shape"]:
        if not isinstance(codes, CodebookCodes):
            return codes
        if codes.acoustic is not None and codes.acoustic.size != 0:
            raise ValueError("NeuCodec accepts semantic codes only; acoustic codebooks are not supported.")
        return codes.semantic

    def _normalize_codes(self, indices: Int[Array, "*shape"]) -> Int[Array, "1 1 tokens"]:
        if indices.ndim not in (1, 2, 3):
            raise ValueError(f"NeuCodec code tensor must have 1, 2, or 3 dimensions; got {indices.shape}.")
        if any(dim != 1 for dim in indices.shape[:-1]):
            raise ValueError("NeuCodec expects a single batch item and a single codebook.")
        return indices.reshape((1, 1, indices.shape[-1]))

    def __call__(
        self,
        indices: Int[Array, "*shape"] | CodebookCodes,
    ) -> Float[Array, " samples"]:
        codes = self._normalize_codes(self._semantic_indices(indices))
        fsq_post_emb = self.quantizer.get_output_from_indices(jnp.transpose(codes, (0, 2, 1)))
        hidden_states = self.fc_post_a(fsq_post_emb)
        reconstructed = self.vocos_decoder(hidden_states)
        return reconstructed[0, 0, :].astype(self.config.precision)

    def audio_from_codes(self, indices: Array | CodebookCodes) -> Array:
        return self(indices)
