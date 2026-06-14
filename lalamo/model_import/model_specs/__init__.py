from lalamo.model_import.model_spec import (
    ClassifierModelSpec,
    ConfigMap,
    FileSpec,
    JSONFieldSpec,
    LanguageModelSpec,
    ModelSpec,
    TTSModelSpec,
)

from .bonsai import BONSAI_MODELS
from .deepseek import DEEPSEEK_MODELS
from .essential_ai import RNJ_MODELS
from .fishaudio import FISHAUDIO_TTS_MODELS
from .gemma import GEMMA_MODELS
from .gpt_oss import GPT_OSS_MODELS
from .granite import GRANITE_MODELS
from .huggingface import HUGGINGFACE_MODELS
from .lfm2 import LFM2_MODELS
from .llama import LLAMA_MODELS
from .llamba import LLAMBA_MODELS
from .mirai import MIRAI_CLASSIFIER_MODELS
from .mistral import MISTRAL_MODELS
from .nanbeige import NANBEIGE_MODELS
from .phi import PHI_MODELS
from .polaris import POLARIS_MODELS
from .qwen import QWEN_MODELS
from .reka import REKA_MODELS

__all__ = [
    "ALL_MODEL_LISTS",
    "ClassifierModelSpec",
    "ConfigMap",
    "FileSpec",
    "JSONFieldSpec",
    "LanguageModelSpec",
    "ModelSpec",
    "TTSModelSpec",
]

TTS_MODELS = FISHAUDIO_TTS_MODELS

ALL_MODEL_LISTS = [
    GRANITE_MODELS,
    LFM2_MODELS,
    LLAMA_MODELS,
    LLAMBA_MODELS,
    DEEPSEEK_MODELS,
    GEMMA_MODELS,
    HUGGINGFACE_MODELS,
    GPT_OSS_MODELS,
    MISTRAL_MODELS,
    PHI_MODELS,
    POLARIS_MODELS,
    QWEN_MODELS,
    REKA_MODELS,
    MIRAI_CLASSIFIER_MODELS,
    NANBEIGE_MODELS,
    RNJ_MODELS,
    BONSAI_MODELS,
    TTS_MODELS,
]
