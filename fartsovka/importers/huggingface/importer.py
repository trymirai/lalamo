from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from safetensors.flax import load_file

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.models.baseline_llama import BaselineLlama, get_baseline_llama
from fartsovka.models.qwen2 import Qwen2, get_qwen2

from .config import LlamaConfig, Qwen2Config
from .loader import load_huggingface

__all__ = ["HuggingFaceModel", "import_model"]


class HuggingFaceModel(Enum):
    LLAMA32_1B_INSTRUCT = auto()
    QWEN25_1POINT5B_INSTRUCT = auto()
    R1_DISTILL_QWEN_1POINT5 = auto()


MODEL_TO_REPO = {
    HuggingFaceModel.LLAMA32_1B_INSTRUCT: "meta-llama/Llama-3.2-1B-Instruct",
    HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT: "Qwen/Qwen2.5-1.5B-Instruct",
    HuggingFaceModel.R1_DISTILL_QWEN_1POINT5: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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


QWEN_BETA_FAST = 32.0
QWEN_BETA_SLOW = 1.0
QWEN_ROPE_SCALING_FACTOR = 4.0


def init_model(
    model: HuggingFaceModel,
    *,
    key: PRNGKeyArray | None = None,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> BaselineLlama | Qwen2:
    if key is None:
        key = jax.random.PRNGKey(0)
    config_path = download_config_file(model)
    if model == HuggingFaceModel.LLAMA32_1B_INSTRUCT:
        config = LlamaConfig.from_json(config_path)
        return get_baseline_llama(
            num_layers=config.num_hidden_layers,
            vocab_dim=config.vocab_size,
            model_dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_groups=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling.factor,
            rope_original_context_length=config.rope_scaling.original_max_position_embeddings,
            rope_low_frequency_factor=config.rope_scaling.low_freq_factor,
            rope_high_frequency_factor=config.rope_scaling.high_freq_factor,
            eps=config.rms_norm_eps,
            max_sequence_length=config.max_position_embeddings,
            key=key,
            precision=precision,
            accumulation_precision=accumulation_precision,
        )
    if model in [HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT, HuggingFaceModel.R1_DISTILL_QWEN_1POINT5]:
        config = Qwen2Config.from_json(config_path)
        return get_qwen2(
            num_layers=config.num_hidden_layers,
            vocab_dim=config.vocab_size,
            model_dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_groups=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rope_theta=config.rope_theta,
            rope_scaling_factor=QWEN_ROPE_SCALING_FACTOR,
            rope_beta_fast=QWEN_BETA_FAST,
            rope_beta_slow=QWEN_BETA_SLOW,
            eps=config.rms_norm_eps,
            max_sequence_length=config.max_position_embeddings,
            key=key,
            precision=precision,
            accumulation_precision=accumulation_precision,
        )
    raise ValueError(f"Unsupported model: {model}")


def import_model(
    model: HuggingFaceModel,
    *,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> BaselineLlama | Qwen2:
    result = init_model(model, precision=precision, accumulation_precision=accumulation_precision)

    weights_path = download_weights(model)
    weights_dict = load_file(weights_path)
    weights_dict = {k: v.astype(precision) for k, v in weights_dict.items()}

    return load_huggingface(result, weights_dict)
