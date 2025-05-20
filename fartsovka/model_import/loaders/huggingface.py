import jax.numpy as jnp
from jaxtyping import Array
from copy import deepcopy
import dataclasses
from typing import TypeVar, Type

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

TLinear = TypeVar("TLinear", bound=LinearBase)

def load_linear(
    module: FullPrecisionLinear,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> FullPrecisionLinear:
    bias_key_in_hf = "bias"
    weight_key_in_hf = "weight"

    loaded_weights = weights_dict[path / weight_key_in_hf]
    loaded_bias: Array | None = None

    if module.biases is not None:
        loaded_bias = weights_dict.get(path / bias_key_in_hf)
        if loaded_bias is None:
            print(f"Warning: Module {path} (FullPrecisionLinear) expects biases, but not found in HuggingFace checkpoint.")
    elif (path / bias_key_in_hf) in weights_dict:
        print(f"Warning: Module {path} (FullPrecisionLinear) has no bias in Fartsovka, but HuggingFace checkpoint provides one. HF bias will be IGNORED.")

    return load_parameters(
        lambda m: (m.weights, m.biases),
        module,
        (loaded_weights, loaded_bias),
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

    if isinstance(module, VisionSdpaAttention):
        if not isinstance(module.qkv, FullPrecisionLinear):
            raise TypeError(f"Expected VisionSdpaAttention.qkv to be FullPrecisionLinear for loader at {path / 'qkv'}, got {type(module.qkv)}")
        if not isinstance(module.proj, FullPrecisionLinear):
            raise TypeError(f"Expected VisionSdpaAttention.proj to be FullPrecisionLinear for loader at {path / 'proj'}, got {type(module.proj)}")
        
        out_proj_path = path / "proj"
        loaded_proj = load_linear(module.proj, weights_dict, out_proj_path)
        
        qkv_path = path / "qkv"
        loaded_qkv = load_linear(module.qkv, weights_dict, qkv_path)
        
        return load_parameters(
            lambda m: (m.qkv, m.proj),
            module,
            (loaded_qkv, loaded_proj),
        )
    
    if not isinstance(module.qkv_projection, FullPrecisionLinear):
        raise TypeError(f"Expected qkv_projection to be FullPrecisionLinear, got {type(module.qkv_projection)}")
    if not isinstance(module.out_projection, FullPrecisionLinear):
        raise TypeError(f"Expected out_projection to be FullPrecisionLinear, got {type(module.out_projection)}")
    
    out_proj = load_linear(module.out_projection, weights_dict, path / "o_proj")
    
    q_proj_weights = weights_dict[path / "q_proj" / "weight"]
    k_proj_weights = weights_dict[path / "k_proj" / "weight"]
    v_proj_weights = weights_dict[path / "v_proj" / "weight"]

    qkv_proj_weights = jnp.concatenate([q_proj_weights, k_proj_weights, v_proj_weights], axis=0)

    bias_paths = [path / p / "bias" for p in ["q_proj", "k_proj", "v_proj"]]
    if module.qkv_projection.biases is None:
        for bias_path in bias_paths:
            if bias_path in weights_dict:
                raise ValueError(f"Bias is not supported for {bias_path} as module.qkv_projection.biases is None, but bias found in checkpoint.")
        qkv_bias = None
    else:
        missing_biases = [bp for bp in bias_paths if bp not in weights_dict]
        if missing_biases:
            raise ValueError(f"Module expects biases, but the following bias paths are missing in weights_dict: {missing_biases}")
        loaded_biases = [weights_dict[bias_path] for bias_path in bias_paths]
        qkv_bias = jnp.concatenate(loaded_biases, axis=0)
 
    loaded_qkv_projection_instance = deepcopy(module.qkv_projection)
 
    loaded_qkv_projection = load_parameters(
        lambda m_qkv: (m_qkv.weights, m_qkv.biases),
        loaded_qkv_projection_instance,
        (qkv_proj_weights, qkv_bias)
    )

    return load_parameters(
        lambda m_attn: (m_attn.qkv_projection, m_attn.out_projection),  # type: ignore
        module,
        (loaded_qkv_projection, out_proj),
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
    hf_b = weights_dict.get(path / "bias")       

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
    norm = load_rmsnorm(module.norm, weights_dict, path / "ln_q", False)

    if not isinstance(module.hidden_proj, FullPrecisionLinear):
        raise TypeError(f"Expected hidden_proj to be FullPrecisionLinear, got {type(module.hidden_proj)}")
    if not isinstance(module.out_proj, FullPrecisionLinear):
        raise TypeError(f"Expected out_proj to be FullPrecisionLinear, got {type(module.out_proj)}")
        
    hidden_proj = load_linear(module.hidden_proj, weights_dict, path / "mlp" / "0")
    out_proj = load_linear(module.out_proj, weights_dict, path / "mlp" / "2")
    
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
    
    if not module.stages or len(module.stages) != 1:
        raise ValueError(f"Expected VisionTransformer to have exactly one stage for current HF loading, got {len(module.stages) if module.stages else 0}")
    actual_layers_tuple = module.stages[0]
    
    loaded_layers = tuple(
        load_vision_layer(layer, weights_dict, root_path / "blocks" / i)
        for i, layer in enumerate(actual_layers_tuple)
    )

    output_norm_primary = root_path / "norm" 
    output_norm_alt     = root_path / "merger" / "ln_q"

    if (output_norm_primary / "weight") in weights_dict:
        output_norm = load_rmsnorm(
            module.output_norm, weights_dict, output_norm_primary, False
        )

    elif (output_norm_alt / "weight") in weights_dict:
        output_norm = load_rmsnorm(
            module.output_norm, weights_dict, output_norm_alt, False
        )

    else:
        identity_scales = jnp.ones_like(module.output_norm.scales)
        output_norm = load_parameters(
            lambda m: (m.scales,), module.output_norm, (identity_scales,)
        )


    merger = load_vision_merger(module.final_merger, weights_dict, root_path / "merger")
    
    return load_parameters(
        lambda m: (m.patch_embed, m.rope, m.stages, m.inter_stage_mergers, m.output_norm, m.final_merger),
        module,
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
