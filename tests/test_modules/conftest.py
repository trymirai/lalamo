import json

import cattrs
import jax
import pytest
import torch
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.importers.executorch.importer import ExecutorchModel
from fartsovka.importers.executorch.importer import download_config_file as download_et_config
from fartsovka.importers.executorch.importer import download_weights as download_et_weights
from fartsovka.importers.executorch.importer import import_model as import_et
from fartsovka.importers.huggingface.importer import HuggingFaceModel
from fartsovka.importers.huggingface.importer import import_model as import_hf
from fartsovka.models.baseline_llama import BaselineLlama
from fartsovka.models.qlora_llama import QLoRALlama
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
def huggingface_llama() -> transformers.LlamaModel:
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model.eval()
    return model


@pytest.fixture(scope="package")
def fartsovka_llama() -> BaselineLlama:
    model = import_hf(HuggingFaceModel.LLAMA32_1B_INSTRUCT)
    return model


@pytest.fixture(scope="package")
def fartsovka_qlora_llama() -> QLoRALlama:
    model = import_et(ExecutorchModel.LLAMA32_1B_INSTRUCT_QLORA)
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


@pytest.fixture(scope="package")
def executorch_llama() -> ETTransformer:
    config_path = download_et_config(ExecutorchModel.LLAMA32_1B_INSTRUCT_QLORA)
    weights_path = download_et_weights(ExecutorchModel.LLAMA32_1B_INSTRUCT_QLORA)
    with open(config_path) as f:
        params_json = json.load(f)
    params = cattrs.structure(params_json, ETModelArgs)
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = ETTransformer(params)
    _apply_et_transforms(model, weights, params)

    weights = _upcast_weights_to_float32(weights)
    model.load_state_dict(weights, strict=False, assign=True)
    model.eval()

    return model
