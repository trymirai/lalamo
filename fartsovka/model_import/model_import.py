import importlib.metadata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
import torch
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors

from fartsovka.common import DType
from fartsovka.modules import Decoder, DecoderConfig

from .configs import ETLlamaConfig, ForeignConfig, HFGemma2Config, HFLlamaConfig, HFMistralConfig, HFQwen2Config

__all__ = [
    "MODELS",
    "REPO_TO_MODEL",
    "ModelMetadata",
    "ModelSpec",
    "import_model",
]


FARTSOVKA_VERSION = importlib.metadata.version("fartsovka")


@torch.no_grad()
def _torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    # Credit: https://github.com/jax-ml/ml_dtypes/issues/81#issuecomment-2399636232
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


def _convert_torch_array(array: torch.Tensor, float_dtype: DType) -> Array:
    array = array.detach().cpu()
    if array.dtype == torch.bfloat16:
        jax_array = _torch_to_jax_bfloat16(array)
    else:
        jax_array = jnp.array(array.numpy())
    return jax_array.astype(float_dtype)


class WeightsType(Enum):
    SAFETENSORS = "safetensors"
    TORCH = "torch"

    def load(self, filename: Path | str, float_dtype: DType) -> dict[str, jnp.ndarray]:
        if self == WeightsType.SAFETENSORS:
            return {k: v.astype(float_dtype) for k, v in load_safetensors(filename).items()}
        torch_weights = torch.load(filename, map_location="cpu", weights_only=True)
        return {k: _convert_torch_array(v, float_dtype) for k, v in torch_weights.items()}


@dataclass
class ModelMetadata:
    fartsovka_version: str
    vendor: str
    name: str
    model_config: DecoderConfig
    tokenizer_file_names: tuple[str, ...]


@dataclass
class ModelSpec:
    vendor: str
    name: str
    size: str
    repo: str
    config_type: type[ForeignConfig]
    config_file_name: str
    weights_file_names: tuple[str, ...]
    weights_type: WeightsType
    tokenizer_file_names: tuple[str, ...] = tuple()


HUGGINGFACE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


MODELS = [
    ModelSpec(
        vendor="Meta",
        name="Llama-3.1-8B-Instruct",
        size="8B",
        repo="meta-llama/Llama-3.1-8B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Meta",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Meta",
        name="Llama-3.2-1B-Instruct-QLoRA",
        size="1B",
        repo="meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
        config_type=ETLlamaConfig,
        config_file_name="params.json",
        weights_file_names=("consolidated.00.pth",),
        weights_type=WeightsType.TORCH,
    ),
    ModelSpec(
        vendor="Meta",
        name="Llama-3.2-3B-Instruct",
        size="3B",
        repo="meta-llama/Llama-3.2-3B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Google",
        name="Gemma-2-2B-Instruct",
        size="2B",
        repo="google/gemma-2-2b-it",
        config_type=HFGemma2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-0.5B-Instruct",
        size="0.5B",
        repo="Qwen/Qwen2.5-0.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-1.5B-Instruct",
        size="1.5B",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-3B-Instruct",
        size="3B",
        repo="Qwen/Qwen2.5-3B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-7B-Instruct",
        size="7B",
        repo="Qwen/Qwen2.5-7B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-0.5B-Instruct",
        size="0.5B",
        repo="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-1.5B-Instruct",
        size="1.5B",
        repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-3B-Instruct",
        size="3B",
        repo="Qwen/Qwen2.5-Coder-3B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-7B-Instruct",
        size="7B",
        repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-14B-Instruct",
        size="14B",
        repo="Qwen/Qwen2.5-Coder-14B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Alibaba",
        name="Qwen2.5-Coder-32B-Instruct",
        size="32B",
        repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00014.safetensors",
            "model-00002-of-00014.safetensors",
            "model-00003-of-00014.safetensors",
            "model-00004-of-00014.safetensors",
            "model-00005-of-00014.safetensors",
            "model-00006-of-00014.safetensors",
            "model-00007-of-00014.safetensors",
            "model-00008-of-00014.safetensors",
            "model-00009-of-00014.safetensors",
            "model-00010-of-00014.safetensors",
            "model-00011-of-00014.safetensors",
            "model-00012-of-00014.safetensors",
            "model-00013-of-00014.safetensors",
            "model-00014-of-00014.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="DeepSeek",
        name="R1-Distill-Qwen-1.5B",
        size="1.5B",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Mistral",
        name="Codestral-22B-v0.1",
        size="22B",
        repo="mistral-community/Codestral-22B-v0.1",
        config_type=HFMistralConfig,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00009.safetensors",
            "model-00002-of-00009.safetensors",
            "model-00003-of-00009.safetensors",
            "model-00004-of-00009.safetensors",
            "model-00005-of-00009.safetensors",
            "model-00006-of-00009.safetensors",
            "model-00007-of-00009.safetensors",
            "model-00008-of-00009.safetensors",
            "model-00009-of-00009.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="Mistral",
        name="Devstral-Small-2505",
        size="24B",
        repo="mistralai/Devstral-Small-2505",
        config_type=HFMistralConfig,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00010.safetensors",
            "model-00002-of-00010.safetensors",
            "model-00003-of-00010.safetensors",
            "model-00004-of-00010.safetensors",
            "model-00005-of-00010.safetensors",
            "model-00006-of-00010.safetensors",
            "model-00007-of-00010.safetensors",
            "model-00008-of-00010.safetensors",
            "model-00009-of-00010.safetensors",
            "model-00010-of-00010.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="HuggingFace",
        name="SmolLM2-1.7B-Instruct",
        size="1.7B",
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
    ModelSpec(
        vendor="PleIAs",
        name="Pleias-RAG-1B",
        size="1B",
        repo="PleIAs/Pleias-RAG-1B",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        tokenizer_file_names=HUGGINGFACE_TOKENIZER_FILES,
    ),
]


REPO_TO_MODEL = {model.repo: model for model in MODELS}


def download_weights(model_spec: ModelSpec, output_dir: Path | str | None = None) -> list[Path]:
    result = [
        huggingface_hub.hf_hub_download(
            repo_id=model_spec.repo,
            local_dir=output_dir,
            filename=filename,
        )
        for filename in model_spec.weights_file_names
    ]
    return [Path(path) for path in result]


def download_config_file(model_spec: ModelSpec, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=model_spec.repo,
        local_dir=output_dir,
        filename=model_spec.config_file_name,
    )
    return Path(result)


def download_tokenizer_files(model_spec: ModelSpec, output_dir: Path | str | None = None) -> list[Path]:
    result = [
        huggingface_hub.hf_hub_download(
            repo_id=model_spec.repo,
            local_dir=output_dir,
            filename=metadata_filename,
        )
        for metadata_filename in model_spec.tokenizer_file_names
    ]
    return [Path(path) for path in result]


class ImportResults(NamedTuple):
    model: Decoder
    metadata: ModelMetadata
    tokenizer_file_paths: list[Path]


def import_model(
    model_spec: ModelSpec,
    *,
    context_length: int = 8192,
    precision: DType | None = None,
    accumulation_precision: DType = jnp.float32,
) -> ImportResults:
    foreign_config_file = download_config_file(model_spec)
    foreign_config = model_spec.config_type.from_json(foreign_config_file)

    tokenizer_file_paths = download_tokenizer_files(model_spec)
    if precision is None:
        precision = foreign_config.default_precision

    weights_paths = download_weights(model_spec)
    weights_dict = {}
    for weights_path in weights_paths:
        weights_dict.update(model_spec.weights_type.load(weights_path, precision))

    model = foreign_config.load_model(context_length, precision, accumulation_precision, weights_dict)
    metadata = ModelMetadata(
        fartsovka_version=FARTSOVKA_VERSION,
        name=model_spec.name,
        vendor=model_spec.vendor,
        model_config=model.config,
        tokenizer_file_names=tuple(p.name for p in tokenizer_file_paths),
    )
    return ImportResults(model, metadata, tokenizer_file_paths)
