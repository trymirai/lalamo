import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import huggingface_hub
import jax.numpy as jnp
import torch
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors

from fartsovka.common import DType
from fartsovka.modules import Decoder
from fartsovka.modules.vision_transformer import VisionTransformer

from .configs import ETLlamaConfig, ForeignConfig, HFGemma2Config, HFLlamaConfig, HFQwen2Config, HFQwen25VLConfig

__all__ = [
    "MODELS",
    "REPO_TO_MODEL",
    "ModelSpec",
    "get_vision_encoder_from_model",
    "import_model",
]

_logger = logging.getLogger(__name__)


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
class ModelSpec:
    name: str
    repo: str
    config_type: type[ForeignConfig]
    config_file_name: str
    weights_file_names: tuple[str, ...]
    weights_type: WeightsType


MODELS = [
    ModelSpec(
        name="SmolLM2-1.7B-Instruct",
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="Llama-3.2-1B-Instruct",
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="Pleias-RAG-1B",
        repo="PleIAs/Pleias-RAG-1B",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="Llama-3.2-1B-Instruct-QLoRA",
        repo="meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
        config_type=ETLlamaConfig,
        config_file_name="params.json",
        weights_file_names=("consolidated.00.pth",),
        weights_type=WeightsType.TORCH,
    ),
    ModelSpec(
        name="Gemma-2-2B-Instruct",
        repo="google/gemma-2-2b-it",
        config_type=HFGemma2Config,
        config_file_name="config.json",
        weights_file_names=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="Qwen2.5-1.5B-Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="Qwen2.5-VL-3B-Instruct",
        repo="Qwen/Qwen2.5-VL-3B-Instruct",
        config_type=HFQwen25VLConfig,
        config_file_name="config.json",
        weights_file_names=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        weights_type=WeightsType.SAFETENSORS,
    ),
    ModelSpec(
        name="R1-Distill-Qwen-1.5B",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
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


def import_model(
    model_spec: ModelSpec,
    *,
    context_length: int = 8192,
    precision: DType | None = None,
    accumulation_precision: DType = jnp.float32,
) -> Decoder:
    config_file = download_config_file(model_spec)
    config = model_spec.config_type.from_json(config_file)
    if precision is None:
        precision = config.default_precision

    weights_paths = download_weights(model_spec)
    weights_dict = {}
    for weights_path in weights_paths:
        weights_dict.update(model_spec.weights_type.load(weights_path, precision))

    result = config.load_model(context_length, precision, accumulation_precision, weights_dict)

    return result


def get_vision_encoder_from_model(model: Decoder) -> VisionTransformer | None:
    """Get the vision encoder from a model if it exists and is the right type."""
    vision_module = model.vision_module
    if vision_module is not None and not isinstance(vision_module, VisionTransformer):
        _logger.warning(
            "Attribute 'vision_module' found on Decoder, but it is of type "
            f"{type(vision_module).__name__}, not VisionTransformer.",
        )
        return None
    return vision_module
