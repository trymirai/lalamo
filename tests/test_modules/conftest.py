import json
from pathlib import Path

import cattrs
import huggingface_hub
import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import PRNGKeyArray
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from fartsovka.common import DType
from fartsovka.model_import import REPO_TO_MODEL, import_model
from fartsovka.modules import Decoder
from fartsovka.modules.vision_transformer import VisionTransformer
from tests.executorch_llama.source_transformation.lora import (
    transform_linear_for_lora_after_quantization,
)
from tests.executorch_llama.source_transformation.pre_quantization import (
    sanitize_checkpoint_from_pre_quantization,
    transform_embedding_for_pre_quantization,
    transform_linear_for_pre_quantization,
    transform_output_linear_for_pre_quantization,
)
from tests.executorch_llama.transformer import ModelArgs as ETModelArgs
from tests.executorch_llama.transformer import Transformer as ETTransformer

RANDOM_SEED = 42

@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)


@pytest.fixture(scope="package")
def huggingface_gemma2() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


@pytest.fixture(scope="package")
def fartsovka_gemma2() -> Decoder:
    model = import_model(
        REPO_TO_MODEL["google/gemma-2-2b-it"],
        precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    return model


@pytest.fixture(scope="package")
def huggingface_llama() -> LlamaForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


@pytest.fixture(scope="package")
def fartsovka_llama() -> Decoder:
    model = import_model(
        REPO_TO_MODEL["meta-llama/Llama-3.2-1B-Instruct"],
        precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    return model


@pytest.fixture(scope="package")
def huggingface_qwen25() -> Qwen2ForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


@pytest.fixture(scope="package")
def fartsovka_qwen25() -> Decoder:
    model = import_model(
        REPO_TO_MODEL["Qwen/Qwen2.5-1.5B-Instruct"],
        precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    return model


@pytest.fixture(scope="package")
def fartsovka_qlora_llama() -> Decoder:
    model = import_model(
        REPO_TO_MODEL["meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"],
        precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    return model


EMBEDDING_BIT_WIDTH = 8

def _apply_et_transforms(
    model: ETTransformer,
    weights: dict[str, torch.Tensor],
    model_args: ETModelArgs,
    dtype: torch.dtype = torch.float32,
) -> None:
    lora_rank = model_args.lora_args["rank"]  # type: ignore
    group_size = model_args.quantization_args["group_size"]  # type: ignore

    sanitize_checkpoint_from_pre_quantization(weights)
    transform_embedding_for_pre_quantization(model, weights, dtype=dtype, bit_width=EMBEDDING_BIT_WIDTH)
    transform_output_linear_for_pre_quantization(model, weights, dtype=dtype)
    transform_linear_for_pre_quantization(model, weights, group_size=group_size, dtype=dtype)
    transform_linear_for_lora_after_quantization(model, weights, lora_rank=lora_rank)

def _upcast_weights_to_float32(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in weights.items():
        if value.dtype == torch.bfloat16:
            result[key] = value.to(torch.float32)
        else:
            result[key] = value
    return result


def _load_et_config(path: str | Path) -> ETModelArgs:
    with open(path) as f:
        params_json = json.load(f)
    params = cattrs.structure(params_json, ETModelArgs)
    params.rope_scale_factor = 32
    return params


@pytest.fixture(scope="package")
def huggingface_qwen25vl():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model.eval()
        print(f"Successfully loaded HuggingFace model: {model_id}")
        return model
    except ImportError as ie:
        print(f"ImportError loading {model_id} from HuggingFace: {ie}. Check transformers installation and model specific classes.")
        return None
    except Exception as e:
        print(f"Failed to load HuggingFace model {model_id}: {e}")
        return None

@pytest.fixture(scope="package")
def fartsovka_qwen25vl_vision() -> VisionTransformer | None:
    model_repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    model_spec = REPO_TO_MODEL.get(model_repo_id)
    if not model_spec:
        pytest.fail(f"ModelSpec not found for {model_repo_id} in REPO_TO_MODEL.")
        return None

    fartsovka_precision: DType = jnp.float32
    fartsovka_accumulation_precision: DType = jnp.float32

    full_decoder_model: Decoder = import_model(
        model_spec=model_spec,
        precision=fartsovka_precision,
        accumulation_precision=fartsovka_accumulation_precision,
    )

    vision_model = full_decoder_model.vision_module

    if vision_model is None:
        pytest.fail(
            f"Failed to extract VisionTransformer from {model_repo_id}. "
            f"Check if 'vision_module' attribute exists on Decoder and is a VisionTransformer, "
            f"and ensure Decoder class in fartsovka/modules/decoder.py is updated.",
        )

    return vision_model

@pytest.fixture(scope="package")
def executorch_llama() -> ETTransformer:
    repo_id = "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
    config_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename="params.json",
    )
    weights_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename="consolidated.00.pth",
    )
    params = _load_et_config(config_path)
    model = ETTransformer(params)

    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = _upcast_weights_to_float32(weights)

    _apply_et_transforms(model, weights, params)

    model.load_state_dict(weights, strict=False, assign=True)
    model.eval()

    return model
