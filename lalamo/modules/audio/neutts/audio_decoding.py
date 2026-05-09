from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Protocol, Self, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase
from lalamo.modules.audio.text_decoder import CodebookCodes


class _TorchTensor(Protocol):
    def squeeze(self, dim: int) -> Self: ...

    def cpu(self) -> Self: ...

    def numpy(self) -> np.ndarray: ...


class _NeuCodec(Protocol):
    device: object

    def eval(self) -> Self: ...

    def to(self, device: str) -> Self: ...

    def encode_code(self, audio_or_path: str) -> _TorchTensor: ...

    def decode_code(self, codes: object) -> _TorchTensor: ...


class _NeuCodecFactory(Protocol):
    def from_pretrained(self, repo_id: str) -> _NeuCodec: ...


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
def _load_neucodec(codec_repo: str, device: str) -> _NeuCodec:
    try:
        neucodec_module = import_module("neucodec")
    except ImportError as e:
        raise ImportError(
            "NeuTTS audio decoding requires the optional neucodec dependency. Install lalamo with the neutts extra.",
        ) from e

    neucodec_factory = cast("_NeuCodecFactory", neucodec_module.NeuCodec)
    distill_neucodec_factory = cast("_NeuCodecFactory", neucodec_module.DistillNeuCodec)
    match codec_repo:
        case "neuphonic/neucodec":
            codec = neucodec_factory.from_pretrained(codec_repo)
        case "neuphonic/distill-neucodec":
            codec = distill_neucodec_factory.from_pretrained(codec_repo)
        case _:
            raise ValueError(
                "NeuTTS codec_repo must be 'neuphonic/neucodec' or 'neuphonic/distill-neucodec'.",
            )

    return codec.eval().to(device)


class NeuCodecAudioDecoder(TTSAudioDecoder[NeuCodecAudioDecoderConfig]):
    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    @property
    def _codec(self) -> _NeuCodec:
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

    def _semantic_indices(self, codes: Array | CodebookCodes) -> Array:
        if not isinstance(codes, CodebookCodes):
            return codes
        if codes.acoustic is not None and codes.acoustic.size != 0:
            raise ValueError("NeuCodec accepts semantic codes only; acoustic codebooks are not supported.")
        return codes.semantic

    def _normalize_codes(self, indices: Array) -> np.ndarray:
        codes = np.asarray(indices, dtype=np.int64)
        if codes.ndim == 1:
            return codes[None, None, :]
        if codes.ndim == 2:
            if codes.shape[0] != 1:
                raise ValueError("NeuCodec expects a single codebook.")
            return codes[None, :, :]
        if codes.ndim == 3:
            if codes.shape[1] != 1:
                raise ValueError("NeuCodec expects a single codebook.")
            return codes
        raise ValueError(f"NeuCodec code tensor must have 1, 2, or 3 dimensions; got {codes.shape}.")

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
        device = getattr(codec, "device", self.config.device)
        with torch.no_grad():
            code_tensor = torch.tensor(codes, dtype=torch.long).to(device)
            reconstructed = codec.decode_code(code_tensor).cpu().numpy()
        return jnp.asarray(reconstructed[0, 0, :], dtype=self.config.precision)

    def audio_from_codes(self, indices: Array | CodebookCodes) -> Array:
        return self(indices)
