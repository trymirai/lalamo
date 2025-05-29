from .common import ModelSpec
from .deepseek import DEEPSEEK_MODELS
from .gemma import GEMMA_MODELS
from .huggingface import HUGGINGFACE_MODELS
from .llama import LLAMA_MODELS
from .mistral import MISTRAL_MODELS
from .pleias import PLEIAS_MODELS
from .qwen import QWEN_MODELS

__all__ = [
    "ALL_MODELS",
    "REPO_TO_MODEL",
    "ModelSpec",
]


ALL_MODEL_LISTS = [
    LLAMA_MODELS,
    DEEPSEEK_MODELS,
    GEMMA_MODELS,
    HUGGINGFACE_MODELS,
    MISTRAL_MODELS,
    PLEIAS_MODELS,
    QWEN_MODELS,
]


ALL_MODELS = [model for model_list in ALL_MODEL_LISTS for model in model_list]


REPO_TO_MODEL = {model.repo: model for model in ALL_MODELS}
