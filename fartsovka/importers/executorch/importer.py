from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.models.qlora_llama import QLoRALlamaConfig, QLoRALlamaDecoder
from fartsovka.quantization import QuantizationMode

from .config import LlamaConfig as ETLlamaConfig
from .loader import load_llama

__all__ = ["ExecutorchModel", "import_model"]


FLOAT_DTYPES = [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]


class ExecutorchModel(Enum):
    LLAMA32_1B_INSTRUCT_QLORA = auto()


MODEL_TO_REPO: dict[ExecutorchModel, str] = {
    ExecutorchModel.LLAMA32_1B_INSTRUCT_QLORA: "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
}

WEIGHTS_FILE_NAME = "consolidated.00.pth"
CONFIG_FILE_NAME = "params.json"


def download_weights(model: ExecutorchModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=WEIGHTS_FILE_NAME,
    )
    return Path(result)


@torch.no_grad()
def torch_to_jax_bfloat16(tensor: torch.Tensor) -> Array:
    # Credit: https://github.com/jax-ml/ml_dtypes/issues/81#issuecomment-2399636232
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Trying to convert non-bfloat16 tensor to bfloat16")
    intermediate_tensor = tensor.view(torch.uint16)
    return jnp.array(intermediate_tensor).view("bfloat16")


def download_config_file(model: ExecutorchModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=CONFIG_FILE_NAME,
    )
    return Path(result)


def convert_torch_array(array: torch.Tensor, float_dtype: DType) -> Array:
    array = array.detach().cpu()
    if array.dtype == torch.bfloat16:
        jax_array = torch_to_jax_bfloat16(array)
    else:
        jax_array = jnp.array(array.numpy())
    return jax_array.astype(float_dtype)


def load_torch_checkpoint(path: str | Path, float_dtype: DType) -> dict[str, Array]:
    path = Path(path)
    torch_weights = torch.load(path, map_location="cpu", weights_only=True)
    return {k: convert_torch_array(v, float_dtype) for k, v in torch_weights.items()}


def find_hidden_size(model_size: int, ffn_dim_multiplier: float, divisor: int) -> int:
    # Magic formula from executorch
    size_candidate = int(8 / 3 * model_size * ffn_dim_multiplier)
    return size_candidate // divisor * divisor


# These parameters are not present in the config file, and are extracted from the executorch implementation
LOW_FREQ_FACTOR = 1.0
HIGH_FREQ_FACTOR = 4.0
OLD_CONTEXT_LENGTH = 8192
MAX_SEQUENCE_LENGTH = 8192 * 32

ROPE_SCALING_FACTOR = 32.0

EMBEDDING_QUANTIZATION_MODE = QuantizationMode.INT8
ACTIVATION_QUANTIZATION_MODE = QuantizationMode.INT8
WEIGHT_QUANTIZATION_MODE = QuantizationMode.INT4


def get_model_config(
    model: ExecutorchModel,
    *,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlamaConfig:
    config_path = download_config_file(model)
    config = ETLlamaConfig.from_json(config_path)

    if config.lora_args is None:
        raise ValueError("We only support QLoRA models for now.")

    if config.quantization_args is None:
        raise ValueError("Quantization arguments are required for QLoRA models.")

    return QLoRALlamaConfig(
        num_layers=config.n_layers,
        vocab_dim=config.vocab_size,
        model_dim=config.dim,
        hidden_dim=find_hidden_size(config.dim, config.ffn_dim_multiplier, config.multiple_of),
        num_heads=config.n_heads,
        num_groups=config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        rope_theta=config.rope_theta,
        rope_scaling_factor=ROPE_SCALING_FACTOR,
        original_context_length=OLD_CONTEXT_LENGTH,
        rope_low_frequency_factor=LOW_FREQ_FACTOR,
        rope_high_frequency_factor=HIGH_FREQ_FACTOR,
        eps=config.norm_eps,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        weight_quantization_mode=WEIGHT_QUANTIZATION_MODE,
        embedding_quantization_mode=EMBEDDING_QUANTIZATION_MODE,
        activation_quantization_mode=ACTIVATION_QUANTIZATION_MODE,
        quantization_group_size=config.quantization_args.group_size,
        lora_rank=config.lora_args.rank,
        lora_scale=config.lora_args.scale,
        activation_precision=activation_precision,
        accumulation_precision=accumulation_precision,
    )


def import_model(
    model: ExecutorchModel,
    *,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlamaDecoder:
    config = get_model_config(
        model,
        activation_precision=activation_precision,
        accumulation_precision=accumulation_precision,
    )
    result = config(key=jax.random.PRNGKey(0))

    weights_path = download_weights(model)
    weights_dict = load_torch_checkpoint(
        weights_path,
        float_dtype=activation_precision,
    )
    return load_llama(result, weights_dict)
