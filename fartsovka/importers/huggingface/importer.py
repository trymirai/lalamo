from enum import Enum, auto
from pathlib import Path

import huggingface_hub
import jax
import jax.numpy as jnp
from safetensors.flax import load_file

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.models.gemma2 import Gemma2Config, Gemma2Decoder
from fartsovka.models.llama import LlamaConfig, LlamaDecoder
from fartsovka.models.qwen2 import Qwen2Config, Qwen2Decoder

from .config import HFGemma2Config as HFGemmaConfig
from .config import HFLlamaConfig as HFLlamaConfig
from .config import HFQwen2Config as HFQwen2Config
from .loader import load_huggingface

__all__ = ["HuggingFaceModel", "import_model", "get_model_config"]


class HuggingFaceModel(Enum):
    LLAMA32_1B_INSTRUCT = auto()

    GEMMA2_2B_INSTRUCT = auto()

    QWEN25_1POINT5B_INSTRUCT = auto()

    R1_DISTILL_QWEN_1POINT5 = auto()


MODEL_TO_REPO = {
    HuggingFaceModel.LLAMA32_1B_INSTRUCT: "meta-llama/Llama-3.2-1B-Instruct",
    HuggingFaceModel.GEMMA2_2B_INSTRUCT: "google/gemma-2-2b-it",
    HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT: "Qwen/Qwen2.5-1.5B-Instruct",
    HuggingFaceModel.R1_DISTILL_QWEN_1POINT5: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

MODEL_TO_ADD_ONE_TO_RMS_NORM_WEIGHTS = {
    HuggingFaceModel.LLAMA32_1B_INSTRUCT: False,
    HuggingFaceModel.GEMMA2_2B_INSTRUCT: True,
    HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT: False,
    HuggingFaceModel.R1_DISTILL_QWEN_1POINT5: False,
}

WEIGHTS_FILE_NAMES = {
    HuggingFaceModel.LLAMA32_1B_INSTRUCT: ["model.safetensors"],
    HuggingFaceModel.GEMMA2_2B_INSTRUCT: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT: ["model.safetensors"],
    HuggingFaceModel.R1_DISTILL_QWEN_1POINT5: ["model.safetensors"],
}
CONFIG_FILE_NAME = "config.json"


def download_weights(model: HuggingFaceModel, output_dir: Path | str | None = None) -> list[Path]:
    result = [
        huggingface_hub.hf_hub_download(
            repo_id=MODEL_TO_REPO[model],
            local_dir=output_dir,
            filename=filename,
        )
        for filename in WEIGHTS_FILE_NAMES[model]
    ]
    return [Path(path) for path in result]


def download_config_file(model: HuggingFaceModel, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=MODEL_TO_REPO[model],
        local_dir=output_dir,
        filename=CONFIG_FILE_NAME,
    )
    return Path(result)


def _get_sliding_window_sizes(hf_config: HFQwen2Config) -> list[int | None]:
    sliding_window_sizes = []
    for i in range(hf_config.num_hidden_layers):
        if i < hf_config.max_window_layers:
            sliding_window_sizes.append(hf_config.sliding_window)
        else:
            sliding_window_sizes.append(None)
    return sliding_window_sizes


def get_model_config(
    model: HuggingFaceModel,
    *,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> LlamaConfig | Qwen2Config | Gemma2Config:
    config_path = download_config_file(model)
    if model == HuggingFaceModel.LLAMA32_1B_INSTRUCT:
        hf_config = HFLlamaConfig.from_json(config_path)
        return LlamaConfig(
            num_layers=hf_config.num_hidden_layers,
            vocab_dim=hf_config.vocab_size,
            model_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_groups=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_sequence_length=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            eps=hf_config.rms_norm_eps,
            original_context_length=hf_config.rope_scaling.original_max_position_embeddings,
            precision=precision,
            accumulation_precision=accumulation_precision,
            rope_scaling_factor=hf_config.rope_scaling.factor,
            rope_low_frequency_factor=hf_config.rope_scaling.low_freq_factor,
            rope_high_frequency_factor=hf_config.rope_scaling.high_freq_factor,
        )

    if model in [HuggingFaceModel.QWEN25_1POINT5B_INSTRUCT, HuggingFaceModel.R1_DISTILL_QWEN_1POINT5]:
        hf_config = HFQwen2Config.from_json(config_path)
        sliding_window_sizes = _get_sliding_window_sizes(hf_config)
        return Qwen2Config(
            num_layers=hf_config.num_hidden_layers,
            vocab_dim=hf_config.vocab_size,
            model_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_groups=hf_config.num_key_value_heads,
            head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
            sliding_window_sizes=sliding_window_sizes,
            max_sequence_length=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            eps=hf_config.rms_norm_eps,
            precision=precision,
            accumulation_precision=accumulation_precision,
        )

    if model in [HuggingFaceModel.GEMMA2_2B_INSTRUCT]:
        hf_config = HFGemmaConfig.from_json(config_path)
        return Gemma2Config(
            num_layers=hf_config.num_hidden_layers,
            vocab_dim=hf_config.vocab_size,
            model_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_groups=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_sequence_length=hf_config.max_position_embeddings,
            sliding_window_size=hf_config.sliding_window,
            query_pre_attn_scalar=hf_config.query_pre_attn_scalar,
            attention_logits_soft_cap=hf_config.attn_logit_softcapping,
            output_logits_soft_cap=hf_config.final_logit_softcapping,
            rope_theta=hf_config.rope_theta,
            eps=hf_config.rms_norm_eps,
            precision=precision,
            accumulation_precision=accumulation_precision,
        )

    raise ValueError(f"Unsupported model: {model}")


def import_model(
    model: HuggingFaceModel,
    *,
    precision: DType = DEFAULT_PRECISION,
    accumulation_precision: DType = jnp.float32,
) -> LlamaDecoder | Qwen2Decoder | Gemma2Decoder:
    config = get_model_config(model, precision=precision, accumulation_precision=accumulation_precision)
    result = config(key=jax.random.PRNGKey(0))

    weights_paths = download_weights(model)
    weights_dict = {}
    for weights_path in weights_paths:
        weights_dict.update(load_file(weights_path))
    weights_dict = {k: v.astype(precision) for k, v in weights_dict.items()}

    return load_huggingface(
        result,
        weights_dict,
        add_one_to_rms_norm_weights=MODEL_TO_ADD_ONE_TO_RMS_NORM_WEIGHTS[model],
    )
