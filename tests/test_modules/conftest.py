import json
from pathlib import Path
from typing import Any

import cattrs
import jax
import jax.numpy as jnp
import pytest
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from jaxtyping import PRNGKeyArray
from safetensors import safe_open

from fartsovka.model_import import REPO_TO_MODEL, import_model
from fartsovka.model_import.model_import import download_config_file, download_weights, ModelSpec
from fartsovka.model_import.configs import HFQwen25VLConfig
from fartsovka.modules import (
    Decoder, 
    VisionTransformer,
)
from fartsovka.common import DType

RANDOM_SEED = 42


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)


@pytest.fixture(scope="package")
def huggingface_gemma2() -> Any:
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
def huggingface_llama() -> Any:
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
def huggingface_qwen25() -> Any:
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


@pytest.fixture(scope="package")
def huggingface_qwen25vl():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    try:
        import torch 
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32, # Using float32 for consistency with other fixtures
            device_map="auto" # Let transformers handle device mapping
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

# New helper function
def _load_fartsovka_qwen25vl_vision_only(
    model_spec: ModelSpec,
    precision: DType,
    accumulation_precision: DType,
) -> VisionTransformer | None:
    """Helper function to load only the Fartsovka vision model from Qwen2.5-VL spec and weights."""
    try:
        print(f"\nHelper: Attempting to load Fartsovka vision model for {model_spec.repo}...")
        hf_config_path = download_config_file(model_spec)
        with open(hf_config_path, "r") as f:
            full_hf_config_json = json.load(f)

        hf_qwen_vl_config_obj = cattrs.structure(full_hf_config_json, HFQwen25VLConfig)
        
        # Using the passed 'precision' for JAX dtype for weights and model init
        weights_jax_dtype = precision 
        print(f"Helper: Using JAX dtype for weights & Fartsovka model init: {weights_jax_dtype}")

        weight_files_paths = download_weights(model_spec)
        weights_dict_jax: dict[str, jnp.ndarray] = {}

        for shard_path in weight_files_paths:
            with safe_open(str(shard_path), framework="numpy") as f: # framework was "numpy"
                for key_tensor in f.keys(): # renamed key to key_tensor to avoid conflict
                    tensor_np = f.get_tensor(key_tensor)
                    weights_dict_jax[key_tensor] = jnp.array(tensor_np).astype(weights_jax_dtype)
                
        print(f"Helper: Successfully converted all HF weight shards to JAX arrays for {model_spec.repo}.")

        fartsovka_vision_model = hf_qwen_vl_config_obj.load_vision_model(
            precision=precision, 
            accumulation_precision=accumulation_precision,
            weights_dict=weights_dict_jax 
        )
        print(f"Helper: Successfully loaded Fartsovka vision model with HF weights for {model_spec.repo}!")
        return fartsovka_vision_model
        
    except Exception as e:
        import traceback
        # Use a local variable for repo to avoid potential issues if model_spec itself is problematic
        repo_identifier = model_spec.repo if hasattr(model_spec, 'repo') else 'unknown_repo'
        print(f"\n===== ERROR in _load_fartsovka_qwen25vl_vision_only for {repo_identifier} =====")
        print(f"Error type: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("==============================================================================\n")
        return None

@pytest.fixture(scope="package")
def fartsovka_qwen25vl_vision() -> VisionTransformer | None:
    """Load Fartsovka vision model with weights from HuggingFace Qwen2.5-VL."""
    model_repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model_spec = REPO_TO_MODEL.get(model_repo_id)
    if not model_spec:
        pytest.fail(f"ModelSpec not found for {model_repo_id} in REPO_TO_MODEL. Cannot load Fartsovka vision model.")

    # Define precision levels as in other Fartsovka fixtures
    fartsovka_precision = jnp.float32
    fartsovka_accumulation_precision = jnp.float32
    
    return _load_fartsovka_qwen25vl_vision_only(
        model_spec=model_spec,
        precision=fartsovka_precision,
        accumulation_precision=fartsovka_accumulation_precision,
    )
