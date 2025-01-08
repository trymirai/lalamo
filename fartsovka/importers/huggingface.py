from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from fartsovka.modules.common import DEFAULT_PRECISION
from fartsovka.modules.llama import BaselineLlama, get_baseline_llama

from .huggingface_config import LlamaConfig


class HuggingFaceModel(Enum):
    LLaMA1b = auto()


MODEL_TO_REPO: dict[HuggingFaceModel, str] = {
    HuggingFaceModel.LLaMA1b: "meta-llama/Llama-3.2-1B-Instruct",
}

WEIGHTS_FILE_NAME = "model.safetensors"
CONFIG_FILE_NAME = "config.json"


@dataclass
class DownloadedModel:
    config_path: Path
    weights_path: Path


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


def download_model(model: HuggingFaceModel, output_dir: Path | str | None = None) -> DownloadedModel:
    config_path = download_config_file(model, output_dir)
    weights_path = download_weights(model, output_dir)
    return DownloadedModel(config_path=config_path, weights_path=weights_path)


def init_model(
    model: HuggingFaceModel | Path | str,
    *,
    key: PRNGKeyArray | None = None,
    precision: jnp.dtype = DEFAULT_PRECISION,
    accumulation_precision: jnp.dtype = jnp.float32,
) -> BaselineLlama:
    if key is None:
        key = jax.random.PRNGKey(0)
    if isinstance(model, HuggingFaceModel):
        config_file_path = download_config_file(model)
    else:
        config_file_path = Path(model) / CONFIG_FILE_NAME
    config = LlamaConfig.from_json(config_file_path)
    return get_baseline_llama(
        num_layers=config.num_hidden_layers,
        vocab_dim=config.vocab_size,
        model_dim=config.hidden_size,
        hidden_dim=config.intermediate_size,
        num_heads=config.num_attention_heads,
        num_groups=config.num_key_value_heads,
        head_dim=config.head_dim,
        eps=config.rms_norm_eps,
        max_sequence_length=config.max_position_embeddings,
        key=key,
        precision=precision,
        accumulation_precision=accumulation_precision,
    )
