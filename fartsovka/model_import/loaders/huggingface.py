import jax.numpy as jnp
from jaxtyping import Array
from copy import deepcopy
import dataclasses

from fartsovka.common import ParameterPath
from fartsovka.modules import (
    MLP, 
    Attention, 
    DecoderLayer, 
    FullPrecisionLinear, 
    RMSNorm, 
    TiedEmbedding,
    VisionTransformer,
    PatchEmbedding,
    VisionLayer,
    PatchMerger,
    LinearBase,
)
from fartsovka.modules.decoder import Decoder

from .common import load_parameters

__all__ = ["load_huggingface", "load_vision_huggingface"]


def load_linear(
    module: FullPrecisionLinear,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> FullPrecisionLinear:
    if module.biases is None:
        if path / "bias" in weights_dict:
            # If module has no bias but weights_dict does, it might be an issue or expected for some models.
            # For strict loading, one might raise ValueError here.
            # For flexibility (e.g. if a model sometimes has bias, sometimes not, for same param name):
            print(f"WARN: {path / 'bias'} found in weights_dict, but module {module} has no biases. Ignoring HF bias.")
            loaded_bias = None
        else:
            loaded_bias = None
    else:
        loaded_bias_path = path / "bias"
        if loaded_bias_path not in weights_dict:
            # This can happen if HF model has optional bias and it's not present for this instance.
            print(f"WARN: Bias for {path} not found in weights_dict. Module expects bias. Using None or allowing error in load_parameters.")
            # Depending on how load_parameters handles None for an expected field, this might error or use existing.
            # For safety, if module.biases is not None, we should expect a bias or handle its absence explicitly.
            # However, load_parameters should handle this by checking if new_value is None.
            # If we pass None and load_parameters tries to assign None to a non-Optional field, it will fail, which is good.
            loaded_bias = None # Let load_parameters decide if None is acceptable for module.biases
        else:
            loaded_bias = weights_dict[loaded_bias_path]
            
    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (weights_dict[path / "weight"], loaded_bias),
    )


def load_mlp(module: MLP, weights_dict: dict[str, Array], path: ParameterPath) -> MLP:
    if not isinstance(module.up_projection, FullPrecisionLinear):
        raise TypeError(f"Expected up_projection to be FullPrecisionLinear, got {type(module.up_projection)}")
    if not isinstance(module.down_projection, FullPrecisionLinear):
        raise TypeError(f"Expected down_projection to be FullPrecisionLinear, got {type(module.down_projection)}")

    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)

    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    return load_parameters(
        lambda m: (m.up_projection.weights, m.down_projection.weights),  # type: ignore
        module,
        (fused_up_gate_weights, down_proj_weights),
    )


def load_rmsnorm(
    module: RMSNorm,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one: bool,
) -> RMSNorm:
    scales = weights_dict[path / "weight"]
    if add_one:
        scales = scales + 1.0
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> Attention:
    if not isinstance(module.qkv_projection, FullPrecisionLinear):
        raise TypeError(f"Expected qkv_projection to be FullPrecisionLinear, got {type(module.qkv_projection)}")
    if not isinstance(module.out_projection, FullPrecisionLinear):
        raise TypeError(f"Expected out_projection to be FullPrecisionLinear, got {type(module.out_projection)}")
    
    # Load out_projection
    out_proj_path_standard = path / "o_proj"
    out_proj_path_vision = path / "proj"
    actual_out_proj_path = None
    if (out_proj_path_standard / "weight") in weights_dict:
        actual_out_proj_path = out_proj_path_standard
    elif (out_proj_path_vision / "weight") in weights_dict:
        actual_out_proj_path = out_proj_path_vision
    
    if actual_out_proj_path is None:
        raise KeyError(f"Output projection weight not found at {out_proj_path_standard} or {out_proj_path_vision}")

    loaded_out_projection = load_linear(deepcopy(module.out_projection), weights_dict, actual_out_proj_path)

    # Prepare QKV projection weights and biases
    qkv_proj_module_instance = deepcopy(module.qkv_projection)
    qkv_proj_weights: Array
    qkv_bias: Array | None

    qkv_combined_weight_path = path / "qkv" / "weight"
    qkv_combined_bias_path = path / "qkv" / "bias"

    if qkv_combined_weight_path in weights_dict:
        qkv_proj_weights = weights_dict[qkv_combined_weight_path]
        if module.qkv_projection.biases is not None:
            if qkv_combined_bias_path in weights_dict:
                qkv_bias = weights_dict[qkv_combined_bias_path]
            else:
                print(f"WARN: Combined QKV bias expected for {qkv_combined_bias_path} but not found. Module requires bias. This might error.")
                qkv_bias = None # Will cause error if module.qkv_projection.biases is not Optional[Array]
        else:
            qkv_bias = None
            if qkv_combined_bias_path in weights_dict:
                 print(f"WARN: Combined QKV bias found at {qkv_combined_bias_path} but module QKV projection does not have biases.")
    else:
        q_proj_weights = weights_dict[path / "q_proj" / "weight"]
        k_proj_weights = weights_dict[path / "k_proj" / "weight"]
        v_proj_weights = weights_dict[path / "v_proj" / "weight"]
        qkv_proj_weights = jnp.concatenate([q_proj_weights, k_proj_weights, v_proj_weights], axis=0)

        if module.qkv_projection.biases is not None:
            q_bias = weights_dict[path / "q_proj" / "bias"]
            k_bias = weights_dict[path / "k_proj" / "bias"]
            v_bias = weights_dict[path / "v_proj" / "bias"]
            qkv_bias = jnp.concatenate([q_bias, k_bias, v_bias], axis=0)
        else:
            qkv_bias = None
            for p_name in ["q_proj", "k_proj", "v_proj"]:
                if path / p_name / "bias" in weights_dict:
                    print(f"WARN: Separate bias {path / p_name / 'bias'} found but module QKV projection does not support biases.")
                    break
    
    # Load weights and biases into the qkv_projection module instance
    loaded_qkv_projection = load_parameters(
        lambda m_qkv: (m_qkv.weights, m_qkv.biases),
        qkv_proj_module_instance, # This is a FullPrecisionLinear instance
        (qkv_proj_weights, qkv_bias)
    )

    # Load the updated qkv_projection and out_projection into the main Attention module
    return load_parameters(
        lambda m_attn: (m_attn.qkv_projection, m_attn.out_projection),
        module,
        (loaded_qkv_projection, loaded_out_projection),
    )


def load_decoder_layer(
    module: DecoderLayer,
    weights_dict: dict[str, Array],
    path: ParameterPath,
    add_one_to_rms_norm_weights: bool,
) -> DecoderLayer:
    pre_attention_norm = load_rmsnorm(
        module.pre_attention_norm,
        weights_dict,
        path / "input_layernorm",
        add_one_to_rms_norm_weights,
    )
    attention = load_attention(module.attention, weights_dict, path / "self_attn")
    if module.post_attention_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_attention_norm,
            weights_dict,
            path / "post_attention_layernorm",
            add_one_to_rms_norm_weights,
        )

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "pre_feedforward_layernorm",
            add_one_to_rms_norm_weights,
        )
    else:
        post_attention_norm = None

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "post_attention_layernorm",
            add_one_to_rms_norm_weights,
        )

    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    if module.post_mlp_norm is not None:
        post_mlp_norm = load_rmsnorm(
            module.post_mlp_norm,
            weights_dict,
            path / "post_feedforward_layernorm",
            add_one_to_rms_norm_weights,
        )
    else:
        post_mlp_norm = None
    return load_parameters(
        lambda m: (m.pre_attention_norm, m.attention, m.post_attention_norm, m.pre_mlp_norm, m.mlp, m.post_mlp_norm),
        module,
        (pre_attention_norm, attention, post_attention_norm, pre_mlp_norm, mlp, post_mlp_norm),
    )


def load_embedding(module: TiedEmbedding, weights_dict: dict[str, Array], path: ParameterPath) -> TiedEmbedding:
    weights = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_huggingface(
    module: Decoder,
    weights_dict: dict[str, Array],
    add_one_to_rms_norm_weights: bool,
) -> Decoder:
    root_path: ParameterPath = ParameterPath("model")
    if not isinstance(module.embedding, TiedEmbedding):
        raise TypeError(f"Expected embedding to be TiedEmbedding, got {type(module.embedding)}")
    embedding = load_embedding(module.embedding, weights_dict, root_path / "embed_tokens")
    decoder_layers = tuple(
        load_decoder_layer(layer, weights_dict, root_path / "layers" / i, add_one_to_rms_norm_weights)
        for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm", add_one_to_rms_norm_weights)
    return load_parameters(
        lambda m: (m.embedding, m.layers, m.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )


def load_vision_patch_embedding(
    module: PatchEmbedding,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> PatchEmbedding:
    hf_w = weights_dict[path / "weight"]          # (out, in, T, H, W)
    hf_b = weights_dict.get(path / "bias")        # <- exists in Qwen-2.5-VL

    fs_w = jnp.transpose(hf_w, (0, 2, 3, 4, 1))   # (out, T, H, W, in)

    # ─────────────────────────────────────────────────────────────
    # NEW: if the checkpoint provides a bias, keep it.
    # ─────────────────────────────────────────────────────────────
    if hf_b is not None:
        module.config = dataclasses.replace(module.config, has_bias=True)
        fs_b: Array | None = hf_b
    else:
        fs_b = None
    # ─────────────────────────────────────────────────────────────

    return load_parameters(lambda m: (m.weights, m.biases), module, (fs_w, fs_b))


def load_vision_merger(
    module: PatchMerger,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> PatchMerger:
    """Load PatchMerger weights from Hugging Face Qwen2.5-VL structure."""
    # module.norm should map to path / "ln_q" (HF: self.ln_q in Qwen2_5_VLPatchMerger)
    norm = load_rmsnorm(module.norm, weights_dict, path / "ln_q", False) # CORRECTED path
    
    # module.hidden_proj should map to path / "mlp" / "0" (HF: self.mlp[0] in Qwen2_5_VLPatchMerger)
    # module.out_proj should map to path / "mlp" / "2" (HF: self.mlp[2] in Qwen2_5_VLPatchMerger)
    if not isinstance(module.hidden_proj, FullPrecisionLinear):
        raise TypeError(f"Expected hidden_proj to be FullPrecisionLinear, got {type(module.hidden_proj)}")
    if not isinstance(module.out_proj, FullPrecisionLinear):
        raise TypeError(f"Expected out_proj to be FullPrecisionLinear, got {type(module.out_proj)}")
        
    hidden_proj = load_linear(module.hidden_proj, weights_dict, path / "mlp" / "0") # CORRECTED path
    out_proj = load_linear(module.out_proj, weights_dict, path / "mlp" / "2") # CORRECTED path
    
    return load_parameters(
        lambda m: (m.norm, m.hidden_proj, m.out_proj),
        module,
        (norm, hidden_proj, out_proj),
    )


def load_vision_huggingface(
    module: VisionTransformer,
    weights_dict: dict[str, Array],
) -> VisionTransformer:
    """Load VisionTransformer model from Hugging Face weights."""
    root_path: ParameterPath = ParameterPath("visual")
    
    patch_embed = load_vision_patch_embedding(
        module.patch_embed, weights_dict, root_path / "patch_embed" / "proj" 
    )
    
    rope = module.rope 
    print(f"DEBUG: VisionRoPE inv_freq for the current vision model will be taken from Fartsovka module init, not checkpoint.")
    
    # Assuming a single stage for current HF ViT mapping
    if not module.stages or len(module.stages) != 1:
        raise ValueError(f"Expected VisionTransformer to have exactly one stage for current HF loading, got {len(module.stages) if module.stages else 0}")
    actual_layers_tuple = module.stages[0]
    
    loaded_layers = tuple(
        load_vision_layer(layer, weights_dict, root_path / "blocks" / i)
        for i, layer in enumerate(actual_layers_tuple)
    )
    
    output_norm_key = root_path / "norm" / "weight"
    if output_norm_key in weights_dict:
        output_norm = load_rmsnorm(module.output_norm, weights_dict, root_path / "norm", False)
        print("DEBUG: Loaded output_norm from visual.norm.weight")
    else:
        output_norm = module.output_norm 
        print(f"DEBUG: visual.norm.weight (key: {output_norm_key}) not found. Using randomly initialized output_norm.")

    # The HF 'merger' corresponds to our 'final_merger' in the single-stage context
    merger = load_vision_merger(module.final_merger, weights_dict, root_path / "merger")
    
    return load_parameters(
        lambda m: (m.patch_embed, m.rope, m.stages, m.inter_stage_mergers, m.output_norm, m.final_merger),
        module,
        # Store the loaded layers back into a tuple representing the single stage
        (patch_embed, rope, (loaded_layers,), module.inter_stage_mergers, output_norm, merger),
    )


def load_vision_layer(
    module: VisionLayer,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> VisionLayer:
    """Load VisionLayer weights from Hugging Face."""
    norm1 = load_rmsnorm(module.norm1, weights_dict, path / "norm1", False)
    attention = load_attention(module.attention, weights_dict, path / "attn")
    norm2 = load_rmsnorm(module.norm2, weights_dict, path / "norm2", False)
    mlp = load_mlp(module.mlp, weights_dict, path / "mlp")
    
    return load_parameters(
        lambda m: (m.norm1, m.attention, m.norm2, m.mlp),
        module,
        (norm1, attention, norm2, mlp),
    )
