from .common import awq_model_spec, ModelSpec, UseCase
from .deepseek import DEEPSEEK_MODELS
from .gemma import GEMMA_MODELS
from .huggingface import HUGGINGFACE_MODELS
from .llama import LLAMA_MODELS
from .mistral import MISTRAL_MODELS
from .pleias import PLEIAS_MODELS
from .polaris import POLARIS_MODELS
from .qwen import QWEN_MODELS

__all__ = [
    "ALL_MODELS",
    "REPO_TO_MODEL",
    "ModelSpec",
    "UseCase",
]


ALL_MODEL_LISTS = [
    LLAMA_MODELS,
    DEEPSEEK_MODELS,
    GEMMA_MODELS,
    HUGGINGFACE_MODELS,
    MISTRAL_MODELS,
    PLEIAS_MODELS,
    POLARIS_MODELS,
    QWEN_MODELS,
]


ALL_MODELS = [model for model_list in ALL_MODEL_LISTS for model in model_list]

def build_quantized_models(model_specs: list[ModelSpec]):
    quantization_compatible_repos: list[str] = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "PleIAs/Pleias-RAG-1B",
    ]

    quantized_model_specs: list[ModelSpec] = []
    for model_spec in model_specs:
        if model_spec.repo not in quantization_compatible_repos:
            continue
        quantized_repo = "trymirai/{}-AWQ".format(model_spec.repo.split("/")[-1])
        quantized_model_spec = awq_model_spec(model_spec, quantized_repo)
        quantized_model_specs.append(quantized_model_spec)
    return quantized_model_specs


QUANTIZED_MODELS = build_quantized_models(ALL_MODELS)
ALL_MODELS = ALL_MODELS + QUANTIZED_MODELS
REPO_TO_MODEL = {model.repo: model for model in ALL_MODELS}
