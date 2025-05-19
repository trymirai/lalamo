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
    VisionSdpaAttention
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
        # The Equinox module was instantiated without a bias, but the HF
        # checkpoint *does* contain one.  Adopt the HF bias so we preserve
        # numerical parity.
        loaded_bias = weights_dict.get(path / "bias")
    else:
        loaded_bias_path = path / "bias"
        loaded_bias = weights_dict.get(loaded_bias_path)
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

    # Load weights
    up_proj_weights = weights_dict[path / "up_proj" / "weight"]
    gate_proj_weights = weights_dict[path / "gate_proj" / "weight"]
    fused_up_gate_weights = jnp.concatenate([up_proj_weights, gate_proj_weights], axis=0)
    down_proj_weights = weights_dict[path / "down_proj" / "weight"]

    # Load biases if the module is configured to have them
    fused_up_gate_biases: Array | None = None
    down_proj_biases: Array | None = None

    if module.config.has_biases:
        # Check if HF checkpoint actually has these biases
        up_proj_bias_hf = weights_dict.get(path / "up_proj" / "bias")
        gate_proj_bias_hf = weights_dict.get(path / "gate_proj" / "bias")
        down_proj_bias_hf = weights_dict.get(path / "down_proj" / "bias")

        if up_proj_bias_hf is not None and gate_proj_bias_hf is not None:
            fused_up_gate_biases = jnp.concatenate([up_proj_bias_hf, gate_proj_bias_hf], axis=0)
        elif up_proj_bias_hf is not None:
            zeros_for_gate_bias = jnp.zeros_like(up_proj_bias_hf) # Assuming same shape for fusion
            fused_up_gate_biases = jnp.concatenate([up_proj_bias_hf, zeros_for_gate_bias], axis=0)
        elif gate_proj_bias_hf is not None:
            zeros_for_up_bias = jnp.zeros_like(gate_proj_bias_hf)
            fused_up_gate_biases = jnp.concatenate([zeros_for_up_bias, gate_proj_bias_hf], axis=0)
        else:
            print(f"WARN: MLP at {path} configured with has_biases=True, but no up_proj/gate_proj biases found in checkpoint.")
            intermediate_dim_x2 = fused_up_gate_weights.shape[0]
            fused_up_gate_biases = jnp.zeros((intermediate_dim_x2,), dtype=fused_up_gate_weights.dtype)

        if down_proj_bias_hf is not None:
            down_proj_biases = down_proj_bias_hf
        else:
            print(f"WARN: MLP at {path} configured with has_biases=True, but no down_proj.bias found in checkpoint.")
            down_proj_biases = jnp.zeros((module.down_projection.weights.shape[0],), dtype=down_proj_weights.dtype) # Bias shape is (out_features,)

    return load_parameters(
        lambda m: (m.up_projection.weights, m.up_projection.biases, m.down_projection.weights, m.down_projection.biases), # type: ignore
        module,
        (fused_up_gate_weights, fused_up_gate_biases, down_proj_weights, down_proj_biases),
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
    module: Attention | VisionSdpaAttention,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> Attention | VisionSdpaAttention:
    """Load attention weights from HuggingFace checkpoint.
    
    Handles both standard Attention and VisionSdpaAttention modules.
    """

    # Handle VisionSdpaAttention
    if isinstance(module, VisionSdpaAttention):
        if not isinstance(module.qkv, LinearBase):
            raise TypeError(f"Expected qkv to be LinearBase, got {type(module.qkv)}")
        if not isinstance(module.proj, LinearBase):
            raise TypeError(f"Expected proj to be LinearBase, got {type(module.proj)}")
        
        # Load output projection
        out_proj_path = path / "proj"
        loaded_proj = load_linear(deepcopy(module.proj), weights_dict, out_proj_path)
        
        # Load QKV projection
        qkv_path = path / "qkv"
        loaded_qkv = load_linear(deepcopy(module.qkv), weights_dict, qkv_path)
        
        # Load both projections into the VisionSdpaAttention module
        return load_parameters(
            lambda m: (m.qkv, m.proj),
            module,
            (loaded_qkv, loaded_proj),
        )
    
    # Original code for standard Attention
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
    print(f"DEBUG: Entering load_vision_huggingface for module: {type(module)}")
    root_path: ParameterPath = ParameterPath("visual")
    
    print(f"DEBUG: Attempting to load patch_embed from path: {root_path / 'patch_embed' / 'proj'}")
    patch_embed = load_vision_patch_embedding(
        module.patch_embed, weights_dict, root_path / "patch_embed" / "proj" 
    )
    print(f"DEBUG: Successfully loaded patch_embed")
    
    rope = module.rope 
    print(f"DEBUG: VisionRoPE inv_freq for the current vision model will be taken from Fartsovka module init, not checkpoint.")
    
    # Assuming a single stage for current HF ViT mapping
    if not module.stages or len(module.stages) != 1:
        raise ValueError(f"Expected VisionTransformer to have exactly one stage for current HF loading, got {len(module.stages) if module.stages else 0}")
    actual_layers_tuple = module.stages[0]
    print(f"DEBUG: Attempting to load {len(actual_layers_tuple)} layers in the stage.")
    
    loaded_layers = tuple(
        load_vision_layer(layer, weights_dict, root_path / "blocks" / i)
        for i, layer in enumerate(actual_layers_tuple)
    )
    print(f"DEBUG: Successfully loaded {len(loaded_layers)} layers.")

    output_norm_primary = root_path / "norm"  # visual.norm.*
    output_norm_alt     = root_path / "merger" / "ln_q"  # visual.merger.ln_q.*
    print(f"DEBUG: Attempting to load output_norm. Primary path: {output_norm_primary}, Alt path: {output_norm_alt}")

    if (output_norm_primary / "weight") in weights_dict:
        output_norm = load_rmsnorm(
            module.output_norm, weights_dict, output_norm_primary, False
        )
        print("DEBUG: Loaded output_norm from visual.norm.weight (primary key)")

    elif (output_norm_alt / "weight") in weights_dict:
        # Older checkpoints store the final norm only inside the merger.
        output_norm = load_rmsnorm(
            module.output_norm, weights_dict, output_norm_alt, False
        )
        print("DEBUG: Loaded output_norm from visual.merger.ln_q.weight (fallback)")

    else:
        # No pretrained weights → turn the RMSNorm into an identity op
        identity_scales = jnp.ones_like(module.output_norm.scales)
        output_norm = load_parameters(
            lambda m: (m.scales,), module.output_norm, (identity_scales,)
        )
        print(
            "DEBUG: No pretrained output_norm weights found; using identity RMSNorm (scales = 1)."
        )
    print(f"DEBUG: Successfully processed output_norm.")

    # The HF 'merger' corresponds to our 'final_merger' in the single-stage context
    print(f"DEBUG: Attempting to load final_merger from path: {root_path / 'merger'}")
    merger = load_vision_merger(module.final_merger, weights_dict, root_path / "merger")
    print(f"DEBUG: Successfully loaded final_merger.")
    
    print(f"DEBUG: Finalizing load_vision_huggingface.")
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
