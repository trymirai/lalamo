from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase
from lalamo.modules.audio.text_decoder import CodebookCodes


class TorchTensorLike(Protocol):
    def squeeze(self, dim: int) -> Self: ...

    def cpu(self) -> Self: ...

    def numpy(self) -> np.ndarray: ...


class NeuCodecLike(Protocol):
    device: Any

    def eval(self) -> Self: ...

    def to(self, device: str) -> Self: ...

    def encode_code(self, audio_or_path: str) -> TorchTensorLike: ...

    def decode_code(self, codes: object) -> TorchTensorLike: ...


@dataclass(frozen=True)
class NeuCodecAudioDecoderConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike
    codec_repo: str = "neuphonic/neucodec"
    device: str = "cpu"
    samplerate: int = 24_000

    def empty(self) -> "NeuCodecAudioDecoder":
        return NeuCodecAudioDecoder(config=self)

    def random_init(self, *, key: PRNGKeyArray) -> "NeuCodecAudioDecoder":  # noqa: ARG002
        return self.empty()


@cache
def _load_neucodec(codec_repo: str, device: str) -> NeuCodecLike:
    try:
        from neucodec import DistillNeuCodec, NeuCodec  # type: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "NeuTTS audio decoding requires the optional neucodec dependency. Install lalamo with the neutts extra.",
        ) from e

    match codec_repo:
        case "neuphonic/neucodec":
            codec = NeuCodec.from_pretrained(codec_repo)
        case "neuphonic/distill-neucodec":
            codec = DistillNeuCodec.from_pretrained(codec_repo)
        case _:
            raise ValueError(
                "NeuTTS codec_repo must be 'neuphonic/neucodec' or 'neuphonic/distill-neucodec'.",
            )

    return cast("NeuCodecLike", codec.eval().to(device))


class NeuCodecAudioDecoder(TTSAudioDecoder[NeuCodecAudioDecoderConfig]):
    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    @property
    def _codec(self) -> NeuCodecLike:
        return _load_neucodec(self.config.codec_repo, self.config.device)

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],  # noqa: ARG002
    ) -> Self:
        return self

    def encode_reference_audio(self, audio_path: Path | str) -> Int[Array, " speech_tokens"]:
        try:
            import torch
        except ImportError as e:
            raise ImportError("NeuTTS reference encoding requires torch.") from e

        codec = self._codec
        with torch.no_grad():
            codes = codec.encode_code(audio_or_path=str(audio_path)).squeeze(0).squeeze(0)
        return jnp.asarray(codes.cpu().numpy(), dtype=jnp.int32)

    def _semantic_indices(self, codes: Int[Array, "*shape"] | CodebookCodes) -> Int[Array, "*shape"]:
        if not isinstance(codes, CodebookCodes):
            return codes
        if codes.acoustic is not None and codes.acoustic.size != 0:
            raise ValueError("NeuCodec accepts semantic codes only; acoustic codebooks are not supported.")
        return codes.semantic

    def _normalize_codes(self, indices: Int[Array, "*shape"]) -> Int[Array, "1 1 tokens"]:
        if indices.ndim == 1:
            return indices[None, None, :]
        if indices.ndim == 2:
            if indices.shape[0] != 1:
                raise ValueError("NeuCodec expects a single codebook.")
            return indices[None, :, :]
        if indices.ndim == 3:
            if indices.shape[0] != 1:
                raise ValueError("NeuCodec expects a single batch item.")
            if indices.shape[1] != 1:
                raise ValueError("NeuCodec expects a single codebook.")
            return indices
        raise ValueError(f"NeuCodec code tensor must have 1, 2, or 3 dimensions; got {indices.shape}.")

    def __call__(
        self,
        indices: Int[Array, "*shape"] | CodebookCodes,
    ) -> Float[Array, " samples"]:
        try:
            import torch
        except ImportError as e:
            raise ImportError("NeuTTS audio decoding requires torch.") from e

        codes = self._normalize_codes(self._semantic_indices(indices))
        codec = self._codec
        with torch.no_grad():
            code_tensor = torch.as_tensor(
                np.asarray(jax.device_get(codes)),
                dtype=torch.long,
                device=codec.device,
            )
            reconstructed = codec.decode_code(code_tensor).cpu().numpy()
        return jnp.asarray(reconstructed[0, 0, :], dtype=self.config.precision)

    def audio_from_codes(self, indices: Array | CodebookCodes) -> Array:
        return self(indices)
