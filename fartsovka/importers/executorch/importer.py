from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from safetensors.flax import load_file

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.models.qlora_llama import QLoRALlama, get_qlora_llama
from fartsovka.modules.rope import RoPEParams
from fartsovka.quantization import QuantizationMode

from .config import LlamaConfig
from .loader import load_llama

__all__ = ["ExecutorchModel", "import_model"]


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


def download_config_file(model: ExecutorchModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=CONFIG_FILE_NAME,
    )
    return Path(result)


# These parameters are not present in the config file, and are extracted from the executorch implementation
LOW_FREQ_FACTOR = 1
HIGH_FREQ_FACTOR = 4
OLD_CONTEXT_LENGTH = 8192
MAX_SEQUENCE_LENGTH = 8192 * 32
EPS = 1e-5

EMBEDDING_QUANTIZATION_MODE = QuantizationMode.INT8
WEIGHT_QUANTIZATION_MODE = QuantizationMode.INT4


def init_model(
    config_path_or_model: str | Path | ExecutorchModel,
    *,
    key: PRNGKeyArray | None = None,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlama:
    if key is None:
        key = jax.random.PRNGKey(0)
    if isinstance(config_path_or_model, ExecutorchModel):
        config_path = download_config_file(config_path_or_model)
    else:
        config_path = Path(config_path_or_model)
    config = LlamaConfig.from_json(config_path)

    if config.lora_args is None:
        raise ValueError("We only support QLoRA models for now.")

    return get_qlora_llama(
        num_layers=config.n_heads,
        vocab_dim=config.vocab_size,
        model_dim=config.dim,
        hidden_dim=int(config.dim * config.ffn_dim_multiplier),
        num_heads=config.n_heads,
        num_groups=config.n_heads // config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        rope_params=RoPEParams(
            theta=config.rope_theta,
            use_scaling=True,
            scaling_factor=config.rope_theta,
            original_context_length=OLD_CONTEXT_LENGTH,
            low_frequency_factor=LOW_FREQ_FACTOR,
            high_frequency_factor=HIGH_FREQ_FACTOR,
        ),
        eps=EPS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        key=key,
        weight_quantization_mode=WEIGHT_QUANTIZATION_MODE,
        embedding_quantization_mode=EMBEDDING_QUANTIZATION_MODE,
        num_quantization_groups=config.n_heads // config.n_kv_heads,
        lora_rank=config.lora_args.rank,
        lora_scale=config.lora_args.scale,
        activation_precision=activation_precision,
        accumulation_precision=accumulation_precision,
    )


def import_model(
    model: ExecutorchModel | Path | str,
    *,
    key: PRNGKeyArray | None = None,
    activation_precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> QLoRALlama:
    if isinstance(model, ExecutorchModel):
        config_path = download_config_file(model)
        weights_path = download_weights(model)
    else:
        config_path = Path(model) / CONFIG_FILE_NAME
        weights_path = Path(model) / WEIGHTS_FILE_NAME

    result = init_model(
        config_path,
        key=key,
        activation_precision=activation_precision,
        accumulation_precision=accumulation_precision,
    )
    weights_dict = load_file(weights_path)
    return load_llama(result, weights_dict)
