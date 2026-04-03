import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
from cattrs.gen import make_dict_structure_fn
from jaxtyping import Array, DTypeLike, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.common import LalamoModule, config_converter
from lalamo.registry_abc import RegistryABC

from .audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase

# Side-effect imports: ensure concrete configs register with RegistryABC
from .fishaudio.fishaudio_audio_decoding import DescriptAudioCodecConfig as _DescriptAudioCodecConfig  # noqa: F401
from .fishaudio.fishaudio_text_decoding import FishAudioTextDecoderConfig as _FishAudioTextDecoderConfig  # noqa: F401
from .latent_tts import LatentTTSConfig
from .nanocodec.audio_decoding import NanoCodecConfig as _NanoCodecConfig  # noqa: F401
from .nanocodec.stub_text_decoder import StubTextDecoderConfig as _StubTextDecoderConfig  # noqa: F401
from .qwen3_tts.qwen3_tts_audio_decoding import Qwen3TTSAudioDecoderConfig as _Qwen3TTSAudioDecoderConfig  # noqa: F401
from .qwen3_tts.qwen3_tts_text_decoding import Qwen3TTSTextDecoderConfig as _Qwen3TTSTextDecoderConfig  # noqa: F401
from .text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from .vocoders import Vocoder, VocoderConfig


def _registry_unstructure(base_cls: type) -> Callable[[object], dict | None]:  # noqa: ARG001
    def unstructure(obj: object) -> dict | None:
        if obj is None:
            return None
        fields = {
            f.name: config_converter.unstructure(getattr(obj, f.name), f.type)
            for f in dataclasses.fields(obj)  # type: ignore[arg-type]
        }
        return {"type": type(obj).__name__, **fields}

    return unstructure


def _registry_structure(base_cls: type[RegistryABC]) -> Callable[[dict | None, type], object]:
    def structure(data: dict | None, _type: type) -> object:
        if data is None:
            return None
        new_data = dict(data)
        type_name = new_data.pop("type")
        name_to_type = {t.__name__: t for t in base_cls.__descendants__()}
        target_type = name_to_type.get(type_name)
        if target_type is None:
            # Plugin config classes are often registered via import-time side effects.
            # When loading a converted model directly, those entry points may not have
            # been imported yet, so load them lazily and retry resolution once.
            from lalamo.model_registry import load_third_party_specs

            load_third_party_specs("lalamo_plugins.specs.v1")
            name_to_type = {t.__name__: t for t in base_cls.__descendants__()}
            target_type = name_to_type[type_name]
        return make_dict_structure_fn(target_type, config_converter)(new_data, target_type)

    return structure


for _base in (TTSTextDecoderConfigBase, TTSAudioDecoderConfigBase, LatentTTSConfig):
    config_converter.register_unstructure_hook(_base, _registry_unstructure(_base))
    config_converter.register_structure_hook(_base, _registry_structure(_base))


@dataclass(frozen=True)
class TTSConfig:
    text_decoder_config: TTSTextDecoderConfigBase
    audio_decoder_config: TTSAudioDecoderConfigBase
    vocoder_config: VocoderConfig

    activation_precision: DTypeLike

    def empty(self) -> "TTSModel":
        text_decoder = self.text_decoder_config.empty()
        audio_decoder = self.audio_decoder_config.empty()
        vocoder = self.vocoder_config.empty()
        return TTSModel(config=self, text_decoder=text_decoder, audio_decoder=audio_decoder, vocoder=vocoder)

    def random_init(self, key: PRNGKeyArray) -> "TTSModel":
        key_text, key_audio = jax.random.split(key)
        text_decoder = self.text_decoder_config.random_init(key=key_text)
        audio_decoder = self.audio_decoder_config.random_init(key=key_audio)
        vocoder = self.vocoder_config.empty()
        return TTSModel(config=self, text_decoder=text_decoder, audio_decoder=audio_decoder, vocoder=vocoder)


class TTSModel(LalamoModule[TTSConfig]):
    text_decoder: TTSTextDecoder
    audio_decoder: TTSAudioDecoder
    vocoder: Vocoder

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "text_decoder": self.text_decoder.export_weights(),
            "audio_decoder": self.audio_decoder.export_weights(),
            "vocoder": self.vocoder.export_weights(),
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            text_decoder=self.text_decoder.import_weights(require_tree(weights["text_decoder"])),
            audio_decoder=self.audio_decoder.import_weights(require_tree(weights["audio_decoder"])),
            vocoder=self.vocoder.import_weights(require_tree(weights.get("vocoder", {}))),  # ty: ignore[no-matching-overload]
        )
