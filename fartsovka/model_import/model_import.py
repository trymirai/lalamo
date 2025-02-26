from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import huggingface_hub
import jax.numpy as jnp
import torch
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors

from fartsovka.common import DType
from fartsovka.language_model import (
    LanguageModel, 
    LanguageModelConfig, 
    MessageFormatSpec,
    MessageFormatType
)
from fartsovka.modules import Decoder
from fartsovka.tokenizer import HFTokenizer, HFTokenizerConfig

from .configs import ETLlamaConfig, ForeignConfig, HFGemma2Config, HFLlamaConfig, HFQwen2Config

__all__ = [
    "ModelSpec",
    "MODELS",
    "REPO_TO_MODEL",
    "import_model",
    "import_language_model",
]


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
    tokenizer_file_name: str = "tokenizer.json"
    message_format_type: MessageFormatType = MessageFormatType.PLAIN
    custom_format_spec: Optional[MessageFormatSpec] = None


MODELS = [
    ModelSpec(
        name="SmolLM2-1.7B-Instruct",
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        message_format_type=MessageFormatType.LLAMA,
    ),
    ModelSpec(
        name="Llama-3.2-1B-Instruct",
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        message_format_type=MessageFormatType.LLAMA,
    ),
    ModelSpec(
        name="Llama-3.2-1B-Instruct-QLoRA",
        repo="meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
        config_type=ETLlamaConfig,
        config_file_name="params.json",
        weights_file_names=("consolidated.00.pth",),
        weights_type=WeightsType.TORCH,
        message_format_type=MessageFormatType.LLAMA,
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
        message_format_type=MessageFormatType.GEMMA,
    ),
    ModelSpec(
        name="Qwen2.5-1.5B-Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        message_format_type=MessageFormatType.QWEN,
    ),
    ModelSpec(
        name="R1-Distill-Qwen-1.5B",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        config_type=HFQwen2Config,
        config_file_name="config.json",
        weights_file_names=("model.safetensors",),
        weights_type=WeightsType.SAFETENSORS,
        message_format_type=MessageFormatType.QWEN,
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


def download_tokenizer_file(model_spec: ModelSpec, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=model_spec.repo,
        local_dir=output_dir,
        filename=model_spec.tokenizer_file_name,
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


def import_language_model(
    model_spec: ModelSpec,
    *,
    context_length: int = 8192,
    precision: DType | None = None,
    accumulation_precision: DType = jnp.float32,
) -> LanguageModel:
    """Import a model along with its tokenizer and create a LanguageModel."""
    # Load the decoder model
    decoder = import_model(
        model_spec, 
        context_length=context_length, 
        precision=precision,
        accumulation_precision=accumulation_precision,
    )
    
    # Load the tokenizer
    tokenizer_path = download_tokenizer_file(model_spec)
    
    # Get foreign config to extract token IDs
    config_file = download_config_file(model_spec)
    foreign_config = model_spec.config_type.from_json(config_file)
    
    # Create HF tokenizer
    bos_token_id = getattr(foreign_config, "bos_token_id", None)
    eos_token_id = getattr(foreign_config, "eos_token_id", None)
    pad_token_id = getattr(foreign_config, "pad_token_id", None)
    
    # Convert to int if list
    if isinstance(bos_token_id, list) and len(bos_token_id) > 0:
        bos_token_id = bos_token_id[0]
    if isinstance(eos_token_id, list) and len(eos_token_id) > 0:
        eos_token_id = eos_token_id[0]
    
    tokenizer_config = HFTokenizerConfig(
        vocab_size=decoder.config.vocab_size,
        tokenizer_path=str(tokenizer_path),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    tokenizer = HFTokenizer(tokenizer_config)
    
    # Get the message format spec
    if model_spec.custom_format_spec is not None:
        message_format_spec = model_spec.custom_format_spec
    else:
        message_format_spec = LanguageModelConfig.get_default_message_format_spec(
            model_spec.message_format_type
        )
    
    # Create the language model config
    lm_config = LanguageModelConfig(
        decoder_config=decoder.config,
        tokenizer_config=tokenizer_config,
        message_format_type=model_spec.message_format_type,
        message_format_spec=message_format_spec,
        model_name=model_spec.name,
    )
    
    # Create and return the language model
    return LanguageModel(
        config=lm_config,
        decoder=decoder,
        tokenizer=tokenizer,
    )
