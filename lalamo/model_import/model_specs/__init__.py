from .common import FileSpec, ModelSpec, ModelType, UseCase, build_quantized_models
from .deepseek import DEEPSEEK_MODELS
from .essential_ai import RNJ_MODELS
from .gemma import GEMMA_MODELS
from .gpt_oss import GPT_OSS_MODELS
from .huggingface import HUGGINGFACE_MODELS
from .lfm2 import LFM2_MODELS
from .llama import LLAMA_MODELS
from .llamba import LLAMBA_MODELS
from .mirai import MIRAI_CLASSIFIER_MODELS
from .mistral import MISTRAL_MODELS

# from .pleias import PLEIAS_MODELS
from .polaris import POLARIS_MODELS
from .qwen import QWEN_MODELS
from .reka import REKA_MODELS

__all__ = [
    "ALL_MODELS",
    "REPO_TO_MODEL",
    "FileSpec",
    "ModelSpec",
    "ModelType",
    "UseCase",
]


ALL_MODEL_LISTS = [
    LFM2_MODELS,
    LLAMA_MODELS,
    LLAMBA_MODELS,
    DEEPSEEK_MODELS,
    GEMMA_MODELS,
    HUGGINGFACE_MODELS,
    GPT_OSS_MODELS,
    MISTRAL_MODELS,
    # PLEIAS_MODELS,  # TODO(norpadon): Add chat template
    POLARIS_MODELS,
    QWEN_MODELS,
    REKA_MODELS,
    MIRAI_CLASSIFIER_MODELS,
    RNJ_MODELS,
]

ALL_MODELS = [model for model_list in ALL_MODEL_LISTS for model in model_list]


QUANTIZED_MODELS = build_quantized_models(ALL_MODELS)
ALL_MODELS = ALL_MODELS + QUANTIZED_MODELS
REPO_TO_MODEL = {model.repo: model for model in ALL_MODELS}
