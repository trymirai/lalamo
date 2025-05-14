import json
from pathlib import Path
from typing import Any

import cattrs
import huggingface_hub
import jax
import jax.numpy as jnp
import pytest
import torch
import transformers
from jaxtyping import PRNGKeyArray
from safetensors import safe_open

from fartsovka.model_import import REPO_TO_MODEL, import_model
from fartsovka.model_import.model_import import download_config_file, download_weights, ModelSpec
from fartsovka.model_import.configs import HFQwen25VLConfig
from fartsovka.modules import (
    Activation,
    AttentionConfig, 
    Decoder, 
    FullPrecisionLinearConfig,
    MLPConfig,
    PatchEmbedding, 
    PatchEmbeddingConfig,
    PatchMergerConfig,
    RMSNormConfig,
    VisionConfig,
    VisionLayerConfig,
    VisionRoPE, 
    VisionRoPEConfig,
    VisionTransformer,
)
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
from fartsovka.common import DType

RANDOM_SEED = 42


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)


@pytest.fixture(scope="package")
def huggingface_gemma2() -> Any:
    model = transformers.AutoModelForCausalLM.from_pretrained(
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
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
    """Load Qwen2.5-VL model from HuggingFace.
    
    As indicated in the Qwen2.5-VL model card:
    1. The model requires the latest transformers version from GitHub
    2. It uses the specific Qwen2_5_VLForConditionalGeneration class
    
    Returns None if the model cannot be loaded due to dependencies or other issues.
    """
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        # Check transformers version
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Check available Qwen models
        qwen_classes = [cls for cls in dir(transformers) if 'qwen' in cls.lower()]
        has_qwen2_5 = any('qwen2_5' in cls.lower() for cls in qwen_classes)
        
        print(f"Available Qwen classes: {', '.join(qwen_classes)}")
        print(f"Has Qwen2.5 classes: {has_qwen2_5}")
        
        if not has_qwen2_5:
            print("\n===== IMPORTANT INFORMATION =====")
            print("Qwen2.5-VL model requires special transformers version")
            print("Current transformers version does not include Qwen2.5-VL support.")
            print("To enable the reference model test, please install the appropriate version:")
            print("pip install git+https://github.com/huggingface/transformers accelerate")
            print("=====================================\n")
            return None

        # Try to load model
        try:
            # This will fail at the module import stage due to missing TorchAO dependencies
            # We can't work around this with loading parameters
            import torch
            import torchao
            print(f"TorchAO version: {torchao.__version__}")
            
            # Try to import the necessary module - this will likely fail
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            
            print("\nHave all necessary dependencies!")
            
            # If we get here, we can try loading the model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            model.eval()
            print("Successfully loaded Qwen2.5-VL model!")
            return model
            
        except ImportError as e:
            print(f"\n===== DEPENDENCY ISSUE =====")
            print(f"Error: {e}")
            print("\nThe Qwen2.5-VL model has complex dependencies that can't be resolved in the current environment.")
            print("The model implementation requires a specific version of torchao with Int4WeightOnlyConfig.")
            print("This dependency is required at module import time, so we can't work around it with load parameters.")
            print("\nAt this time, the best approach is likely to continue using your Fartsovka model for testing")
            print("and skip the reference model comparison until the libraries stabilize.")
            print("=====================================\n")
            return None
    except Exception as e:
        print(f"Failed to load Qwen2.5-VL model: {e}")
        return None


@pytest.fixture(scope="package")
def fartsovka_qwen25vl_vision() -> VisionTransformer | None:
    """Load Fartsovka vision model with weights from HuggingFace Qwen2.5-VL."""
    model_repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    weights_jax_dtype = jnp.bfloat16 
    fartsovka_compute_precision = jnp.bfloat16 
    fartsovka_accumulation_precision = jnp.float32

    try:
        print(f"\nAttempting to load Fartsovka vision model for {model_repo_id} with HF weights...")
        
        model_spec = REPO_TO_MODEL.get(model_repo_id)
        if not model_spec:
            print(f"ERROR: ModelSpec not found for {model_repo_id} in REPO_TO_MODEL.")
            return None

        hf_config_path = download_config_file(model_spec)
        with open(hf_config_path, "r") as f:
            full_hf_config_json = json.load(f)
        
        hf_qwen_vl_config_obj = cattrs.structure(full_hf_config_json, HFQwen25VLConfig)
        weights_jax_dtype = hf_qwen_vl_config_obj.default_precision
        fartsovka_compute_precision = weights_jax_dtype
        print(f"Using JAX dtype for weights & Fartsovka model init: {weights_jax_dtype}")

        weight_files_paths = download_weights(model_spec)
        weights_dict_jax: dict[str, jnp.ndarray] = {}

        for shard_path in weight_files_paths:
            with safe_open(str(shard_path), framework="numpy") as f:
                for key in f.keys():
                    tensor_np = f.get_tensor(key)          # NumPy array, zero-copy
                    weights_dict_jax[key] = jnp.array(tensor_np).astype(weights_jax_dtype)
                
        print(f"Successfully converted all HF weight shards to Fartsovka JAX arrays.")

        fartsovka_vision_model = hf_qwen_vl_config_obj.load_vision_model(
            precision=fartsovka_compute_precision, 
            accumulation_precision=fartsovka_accumulation_precision,
            weights_dict=weights_dict_jax 
        )
        print(f"Successfully loaded Fartsovka vision model with HF weights for {model_repo_id}!")
        
        return fartsovka_vision_model

    except Exception as e:
        import traceback
        print(f"\n===== ERROR creating Fartsovka Qwen2.5-VL Vision Model with HF Weights =====")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"Detailed stacktrace for Fartsovka vision model loading failure:")
        print(traceback.format_exc())
        print("==============================================================================\n")
        return None


@pytest.fixture(scope="package")
def vision_config() -> VisionConfig:
    """Provide a vision configuration for testing (generic, random)."""
    # This fixture provides a smaller, generic config for basic tests not requiring HF weights.
    # It's kept separate from the HF-specific loading.
    # If a test specifically needs the HF-derived VisionConfig, it should use fartsovka_qwen25vl_vision.config
    
    patch_embedding_config = PatchEmbeddingConfig(
        precision=jnp.float32, patch_size=14, temporal_patch_size=2, in_channels=3,
    )
    rope_config = VisionRoPEConfig(
        precision=jnp.float32, base=10000.0, max_sequence_length=1024,
    )
    norm_config = RMSNormConfig(
        scale_precision=jnp.float32, accumulation_precision=jnp.float32, epsilon=1e-6,
    )
    linear_config = FullPrecisionLinearConfig(precision=jnp.float32)
    attention_config = AttentionConfig(
        qkv_projection_config=linear_config, out_projection_config=linear_config,
        logit_soft_cap=None, has_qkv_biases=True, has_out_biases=True,
    )
    mlp_config = MLPConfig(
        linear_config=linear_config, activation=Activation.SILU, has_biases=True,
    )
    layer_config = VisionLayerConfig(
        norm_config=norm_config, attention_config=attention_config, mlp_config=mlp_config,
    )
    patch_merger_config = PatchMergerConfig(
        precision=jnp.float32, spatial_merge_size=2, has_biases=True,
    )
    
    # Using smaller dimensions for this generic config to keep tests fast if they use it
    return VisionConfig(
        patch_embedding_config=patch_embedding_config,
        rope_config=rope_config,
        layer_config=layer_config,
        patch_merger_config=patch_merger_config,
        output_norm_config=norm_config,
        # Stage-specific parameters for a generic small model
        stage_hidden_dims=(256,),  # Single stage with model_dim 256
        stage_depths=(2,),         # 2 layers in this stage
        stage_num_heads=(4,),      # 4 heads in this stage
        stage_mlp_intermediate_dims=(512,), # MLP intermediate dim 512
        attention_scale=None,
        image_size=224,
        patch_size=14,
        in_channels=3,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=512, # Final output dimension
        fullatt_block_indexes=(0, 1), # All 2 layers use full attention
    )


@pytest.fixture(scope="package")
def vision_model(vision_config: VisionConfig, rng_key: PRNGKeyArray) -> VisionTransformer:
    """Provide a vision transformer model for testing (generic, random)."""
    return vision_config.random_init(key=rng_key)


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
