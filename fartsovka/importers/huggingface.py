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


def download_model(model: HuggingFaceModel, output_dir: Path | str) -> None:
    huggingface_hub.snapshot_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
    )


def load_config(model_dir: Path | str) -> LlamaConfig:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    return LlamaConfig.from_json(config_path)


def init_model(
    model_dir: Path | str,
    *,
    key: PRNGKeyArray | None = None,
    precision: jnp.dtype = DEFAULT_PRECISION,
    accumulation_precision: jnp.dtype = jnp.float32,
) -> BaselineLlama:
    if key is None:
        key = jax.random.PRNGKey(0)
    config = load_config(model_dir)
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
