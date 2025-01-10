from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from safetensors.flax import load_file

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.models.baseline_llama import BaselineLlama, get_baseline_llama
from fartsovka.modules.rope import RoPEParams

from .config import LlamaConfig
from .loader import load_llama

__all__ = ["HuggingFaceModel", "import_model"]


class HuggingFaceModel(Enum):
    LLAMA32_1B_INSTRUCT = auto()


MODEL_TO_REPO: dict[HuggingFaceModel, str] = {
    HuggingFaceModel.LLAMA32_1B_INSTRUCT: "meta-llama/Llama-3.2-1B-Instruct",
}

WEIGHTS_FILE_NAME = "model.safetensors"
CONFIG_FILE_NAME = "config.json"


def download_weights(model: HuggingFaceModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=WEIGHTS_FILE_NAME,
    )
    return Path(result)


def download_config_file(model: HuggingFaceModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=CONFIG_FILE_NAME,
    )
    return Path(result)


def init_model(
    config_path_or_model: str | Path | HuggingFaceModel,
    *,
    key: PRNGKeyArray | None = None,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> BaselineLlama:
    if key is None:
        key = jax.random.PRNGKey(0)
    if isinstance(config_path_or_model, HuggingFaceModel):
        config_path = download_config_file(config_path_or_model)
    else:
        config_path = Path(config_path_or_model)
    config = LlamaConfig.from_json(config_path)
    return get_baseline_llama(
        num_layers=config.num_hidden_layers,
        vocab_dim=config.vocab_size,
        model_dim=config.hidden_size,
        hidden_dim=config.intermediate_size,
        num_heads=config.num_attention_heads,
        num_groups=config.num_key_value_heads,
        head_dim=config.head_dim,
        rope_params=RoPEParams(
            theta=config.rope_theta,
            use_scaling=True,
            scaling_factor=config.rope_scaling.factor,
            original_context_length=config.rope_scaling.original_max_position_embeddings,
            low_frequency_factor=config.rope_scaling.low_freq_factor,
            high_frequency_factor=config.rope_scaling.high_freq_factor,
        ),
        eps=config.rms_norm_eps,
        max_sequence_length=config.max_position_embeddings,
        key=key,
        precision=precision,
        accumulation_precision=accumulation_precision,
    )


def import_model(
    model: HuggingFaceModel | Path | str,
    *,
    key: PRNGKeyArray | None = None,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> BaselineLlama:
    if isinstance(model, HuggingFaceModel):
        config_path = download_config_file(model)
        weights_path = download_weights(model)
    else:
        config_path = Path(model) / CONFIG_FILE_NAME
        weights_path = Path(model) / WEIGHTS_FILE_NAME

    result = init_model(config_path, key=key, precision=precision, accumulation_precision=accumulation_precision)
    weights_dict = load_file(weights_path)
    return load_llama(result, weights_dict)
