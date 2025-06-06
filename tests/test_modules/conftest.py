import json
from collections.abc import Callable, Generator  # Added Callable
from dataclasses import dataclass  # Added
from pathlib import Path

import cattrs
import huggingface_hub
import jax
import jax.numpy as jnp
import pytest
import torch
import transformers
from jaxtyping import PRNGKeyArray

from fartsovka.model_import import REPO_TO_MODEL, import_model
from fartsovka.modules import Decoder, DecoderLayer
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
from tests.executorch_llama.transformer import TransformerBlock as ETDecoderLayer  # Added
from tests.test_modules.common import MAX_TOKEN_INDEX

RANDOM_SEED = 42


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)


# --- Generic Model Loading ---

# ModelSpec from fartsovka.model_import is now the source of truth.
# We will use its keys (HF repo IDs) to parameterize tests.
MODEL_REPO_IDS_TO_TEST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # Add other repo IDs from REPO_TO_MODEL (in fartsovka.model_import) as needed
]

# The old specific fixtures huggingface_gemma3, fartsovka_gemma3, huggingface_llama,
# fartsovka_llama, huggingface_qwen25, fartsovka_qwen25 were previously here and are now removed.


@pytest.fixture(scope="session")
def generic_huggingface_model_factory() -> Callable:
    cache = {}

    def _factory(model_id: str, torch_dtype: torch.dtype = torch.float32) -> transformers.PreTrainedModel:
        if model_id not in cache:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            )
            model.eval()
            cache[model_id] = model
        return cache[model_id]

    return _factory


@pytest.fixture(scope="session")
def generic_fartsovka_model_factory() -> Callable:
    cache = {}

    def _factory(fartsovka_model_config_key: str) -> Decoder:
        if fartsovka_model_config_key not in cache:
            model, *_ = import_model(
                REPO_TO_MODEL[fartsovka_model_config_key],
                context_length=MAX_TOKEN_INDEX,
                precision=jnp.float32,
                accumulation_precision=jnp.float32,
            )
            cache[fartsovka_model_config_key] = model
        return cache[fartsovka_model_config_key]

    return _factory


@pytest.fixture(params=MODEL_REPO_IDS_TO_TEST, ids=lambda repo_id: repo_id.split("/")[-1])
def model_repo_id(request: pytest.FixtureRequest) -> str:
    # This fixture provides the Hugging Face repository ID, which is also the key
    # in REPO_TO_MODEL (mapping to a ModelSpec instance).
    return request.param


@pytest.fixture()
def hf_model(
    model_repo_id: str,
    generic_huggingface_model_factory: Callable,
) -> transformers.PreTrainedModel:
    # The model_repo_id is the Hugging Face model identifier.
    return generic_huggingface_model_factory(model_repo_id)


@pytest.fixture()
def fartsovka_model(
    model_repo_id: str,
    generic_fartsovka_model_factory: Callable,
) -> Decoder:
    # The generic_fartsovka_model_factory expects the key from REPO_TO_MODEL,
    # which is the model_repo_id.
    return generic_fartsovka_model_factory(model_repo_id)


# --- Layer Iteration Abstraction ---
# Old individual model fixtures that were here have been removed.


@dataclass(frozen=True)
class LayerPair:
    fartsovka_layer: DecoderLayer
    reference_layer: torch.nn.Module | ETDecoderLayer


@pytest.fixture
def iterated_layers_data(
    request: pytest.FixtureRequest,
    fartsovka_model: Decoder,
    hf_model: transformers.PreTrainedModel,
) -> Generator[LayerPair, None, None]:
    f_layers = fartsovka_model.layers
    # Standard HF model layers are typically at .model.layers
    ref_layers = hf_model.model.layers

    assert len(f_layers) == len(ref_layers), (
        f"Mismatched layer counts: Fartsovka ({len(f_layers)}) vs Reference ({len(ref_layers)})"
    )

    layers_to_process: list[tuple[DecoderLayer, torch.nn.Module | ETDecoderLayer]]
    if request.node.get_closest_marker("heavy_tests"):
        layers_to_process = list(zip(f_layers, ref_layers, strict=False))
    else:  # Default "light" test
        if not f_layers:  # Handle empty model case
            pytest.skip("No layers to test in light mode for an empty model.")
        layers_to_process = [(f_layers[0], ref_layers[0])]

    for f_layer_instance, ref_layer_instance in layers_to_process:
        yield LayerPair(fartsovka_layer=f_layer_instance, reference_layer=ref_layer_instance)


@pytest.fixture
def qlora_iterated_layers_data(
    request: pytest.FixtureRequest,
    fartsovka_qlora_llama: Decoder,
    executorch_llama: ETTransformer,
) -> Generator[LayerPair, None, None]:
    f_layers = fartsovka_qlora_llama.layers
    et_layers = executorch_llama.layers  # ETTransformer has layers directly

    assert len(f_layers) == len(et_layers), (
        f"Mismatched layer counts: Fartsovka QLoRA ({len(f_layers)}) vs Executorch ({len(et_layers)})"
    )

    layers_to_process: list[tuple[DecoderLayer, ETDecoderLayer]]
    if request.node.get_closest_marker("heavy_tests"):
        layers_to_process = list(zip(f_layers, et_layers, strict=False))  # type: ignore
    else:  # Default "light" test
        if not f_layers:
            pytest.skip("No layers to test in light mode for an empty QLoRA model.")
        layers_to_process = [(f_layers[0], et_layers[0])]  # type: ignore

    for f_layer_instance, et_layer_instance in layers_to_process:
        yield LayerPair(fartsovka_layer=f_layer_instance, reference_layer=et_layer_instance)


# --- Legacy/Specific Model Fixtures (some refactored, some kept) ---


@pytest.fixture(scope="package")
def fartsovka_qlora_llama(generic_fartsovka_model_factory: Callable) -> Decoder:
    return generic_fartsovka_model_factory("meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8")


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
