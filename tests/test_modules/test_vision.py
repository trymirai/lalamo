from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from PIL import Image
from .common import assert_close, from_torch, to_torch, checkify_forward
from jaxtyping import Float, Array

hf_activations: Dict[str, Any] = {}
fs_activations: Dict[str, Any] = {}

def _print_hf_tensor_stats(tensor: Union[torch.Tensor, np.ndarray, None], name: str, store_key: Optional[str] = None, num_elements_to_show: int = 5) -> None:
    """Helper function to print tensor statistics for PyTorch or NumPy tensors."""
    if tensor is None:
        print(f"DEBUG HF STATS: --- {name} (Input was None) ---")
        if store_key: hf_activations[store_key] = None
        return

    data_np: np.ndarray
    original_dtype_str = str(tensor.dtype) if hasattr(tensor, 'dtype') else 'N/A'

    if isinstance(tensor, torch.Tensor):
        tensor_name_suffix = "(Torch)"
        try:
            data_np = tensor.detach().cpu().float().numpy() # Convert to float32 numpy for consistent stats
        except Exception as e:
            print(f"DEBUG HF STATS: Could not convert Torch tensor {name} to NumPy array. Error: {e}")
            if store_key: hf_activations[store_key] = "Conversion Error"
            return
    elif isinstance(tensor, np.ndarray):
        tensor_name_suffix = "(NumPy)"
        if tensor.dtype != np.float32:
            data_np = tensor.astype(np.float32) # Ensure float32 for stats if NumPy input
        else:
            data_np = tensor

    flat_tensor = data_np.flatten()
    
    nan_count = np.sum(np.isnan(flat_tensor))
    inf_count = np.sum(np.isinf(flat_tensor))

    print(f"DEBUG HF STATS: --- {name} {tensor_name_suffix} ---")
    print(f"  Shape: {data_np.shape}, Dtype (original): {original_dtype_str}, Dtype (for stats): {data_np.dtype}")
    if flat_tensor.size > 0:
        rms_val = np.sqrt(np.mean(data_np**2))
        print(f"  Min: {np.min(data_np):.6f}, Max: {np.max(data_np):.6f}, Mean: {np.mean(data_np):.6f}, Sum: {np.sum(data_np):.6f}, RMS: {rms_val:.6f}")
        print(f"  NaNs: {nan_count}, Infs: {inf_count}")
        if flat_tensor.size > 2 * num_elements_to_show:
            print(f"  First {num_elements_to_show} elements: {flat_tensor[:num_elements_to_show]}")
            print(f"  Last  {num_elements_to_show} elements: {flat_tensor[-num_elements_to_show:]}")
        else:
            print(f"  Elements: {flat_tensor}")
    else:
        print(f"  Tensor is empty.")
    print(f"DEBUG HF STATS: --- END {name} ---")
    if store_key:
        # Store the float32 numpy version for consistent comparison later if needed
        hf_activations[store_key] = data_np 

def _print_fs_tensor_stats(tensor: Union[jnp.ndarray, np.ndarray, None], name: str, store_key: Optional[str] = None, num_elements_to_show: int = 5) -> None:
    """Helper function to print tensor statistics for JAX or NumPy tensors with FS prefix."""
    if tensor is None:
        print(f"DEBUG FS STATS: --- {name} (Input was None) ---")
        if store_key: fs_activations[store_key] = None
        return

    data_np: np.ndarray
    original_dtype_str = str(tensor.dtype) if hasattr(tensor, 'dtype') else 'N/A'

    if isinstance(tensor, (jnp.ndarray, np.ndarray)):
        tensor_name_suffix = "(JAX)" if isinstance(tensor, jnp.ndarray) else "(NumPy)"
        # Convert to numpy for consistent stats
        try:
            data_np = np.asarray(tensor)
            if data_np.dtype != np.float32:
                data_np = data_np.astype(np.float32)
        except Exception as e:
            print(f"DEBUG FS STATS: Could not convert tensor {name} to NumPy array. Error: {e}")
            if store_key: fs_activations[store_key] = "Conversion Error"
            return

    flat_tensor = data_np.flatten()
    
    nan_count = np.sum(np.isnan(flat_tensor))
    inf_count = np.sum(np.isinf(flat_tensor))

    print(f"DEBUG FS STATS: --- {name} {tensor_name_suffix} ---")
    print(f"  Shape: {data_np.shape}, Dtype (original): {original_dtype_str}, Dtype (for stats): {data_np.dtype}")
    if flat_tensor.size > 0:
        rms_val = np.sqrt(np.mean(data_np**2))
        print(f"  Min: {np.min(data_np):.6f}, Max: {np.max(data_np):.6f}, Mean: {np.mean(data_np):.6f}, Sum: {np.sum(data_np):.6f}, RMS: {rms_val:.6f}")
        print(f"  NaNs: {nan_count}, Infs: {inf_count}")
        if flat_tensor.size > 2 * num_elements_to_show:
            print(f"  First {num_elements_to_show} elements: {flat_tensor[:num_elements_to_show]}")
            print(f"  Last  {num_elements_to_show} elements: {flat_tensor[-num_elements_to_show:]}")
        else:
            print(f"  Elements: {flat_tensor}")
    else:
        print(f"  Tensor is empty.")
    print(f"DEBUG FS STATS: --- END {name} ---")
    if store_key:
        # Store the float32 numpy version for consistent comparison later if needed
        fs_activations[store_key] = data_np

# Hook function template
def get_hf_hook(name_prefix: str, store_key_prefix: str):
    def hook(module, input_tensors, output_tensors):
        # Inputs can be tuples, handle them carefully
        if isinstance(input_tensors, tuple):
            for i, inp_tensor in enumerate(input_tensors):
                if isinstance(inp_tensor, torch.Tensor):
                    _print_hf_tensor_stats(inp_tensor, f"{name_prefix} - Input {i}", f"{store_key_prefix}_Input_{i}")
        elif isinstance(input_tensors, torch.Tensor):
             _print_hf_tensor_stats(input_tensors, f"{name_prefix} - Input", f"{store_key_prefix}_Input")

        if isinstance(output_tensors, tuple):
            if len(output_tensors) > 0 and isinstance(output_tensors[0], torch.Tensor):
                 _print_hf_tensor_stats(output_tensors[0], f"{name_prefix} - Output 0 (Main)", f"{store_key_prefix}_Output_0_Main")
            for i, out_tensor in enumerate(output_tensors):
                if isinstance(out_tensor, torch.Tensor) and i > 0 : # Print others if they exist
                     _print_hf_tensor_stats(out_tensor, f"{name_prefix} - Output {i}", f"{store_key_prefix}_Output_{i}")
        elif isinstance(output_tensors, torch.Tensor):
            _print_hf_tensor_stats(output_tensors, f"{name_prefix} - Output", f"{store_key_prefix}_Output")
    return hook

# ----------------------------------------------------------------------------
# Original Utilities
# ----------------------------------------------------------------------------

def generate_gradient_image(h: int = 224, w: int = 224) -> Float[Array, "3 h w"]:
    rng = np.random.default_rng(seed=42)  # stable seed → reproducible noise
    img = rng.random((3, h, w), dtype=np.float32)  # CHW in [0,1]
    return jnp.array(img)



def load_test_image(path: Optional[Path] = None) -> Float[Array, "3 224 224"]:
    if path is None:
        return generate_gradient_image()
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return jnp.array(arr)


 
@pytest.mark.parametrize("dtype", [jnp.float32])
def test_vision_encoder_parity(
    huggingface_qwen25vl,
    fartsovka_qwen25vl_vision,
    dtype,
):
    import torch
    import numpy as np

    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model missing")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka VisionTransformer fixture failed to load")

    global hf_activations, fs_activations
    hf_activations = {}
    fs_activations = {}

    img = generate_gradient_image()  # [C, H, W]
    t_sz = fartsovka_qwen25vl_vision.config.patch_embedding_config.temporal_patch_size
    frames = [img] * t_sz
    img_jax = jnp.expand_dims(jnp.stack(frames, axis=1), 0)  # [B, C, T, H, W]
    if dtype == jnp.bfloat16:
        img_jax = img_jax.astype(jnp.bfloat16)
    img_torch = to_torch(img_jax)
    _print_hf_tensor_stats(img_torch, "HF Input: img_torch", "HF_Input_Image")

    # Get model references
    hf_vis = getattr(huggingface_qwen25vl, "vision_tower", getattr(huggingface_qwen25vl, "visual", None))
    if hf_vis is None:
        pytest.skip("HF model missing vision tower")
    fs_vis = fartsovka_qwen25vl_vision
    
    device = img_torch.device
    if hf_vis is not None and hasattr(hf_vis, 'to'):
        hf_vis = hf_vis.to(device)

    B, C, T_img, H_img, W_img = img_torch.shape
    tp = getattr(hf_vis.config, "temporal_patch_size", 2)
    p_sz = getattr(hf_vis.config, "patch_size", 14)
    
    grid_thw_torch = torch.tensor(
        [[T_img // tp, H_img // p_sz, W_img // p_sz]],
        dtype=torch.long,
        device=device,
    )
    grid_thw_jax = from_torch(grid_thw_torch)
    _print_hf_tensor_stats(grid_thw_torch, "HF Input: grid_thw", "HF_Input_GridTHW")
    
    hooks = []
    hooks.append(hf_vis.patch_embed.register_forward_hook(get_hf_hook("HF_PatchEmbed", "HF_PatchEmbed")))
    
    target_block_idx = 31
    hf_block_prefix = f"HF_Block{target_block_idx}"

    if len(hf_vis.blocks) > target_block_idx:
        block_to_hook = hf_vis.blocks[target_block_idx]
        hooks.append(block_to_hook.register_forward_hook(get_hf_hook(hf_block_prefix, hf_block_prefix))) # Overall block output
        
        if hasattr(block_to_hook, "norm1"): # Norm1 output is input to attention
             hooks.append(block_to_hook.norm1.register_forward_hook(get_hf_hook(f"{hf_block_prefix}_Norm1", f"{hf_block_prefix}_Norm1")))

        if hasattr(block_to_hook, "attn"):
            hooks.append(block_to_hook.attn.register_forward_hook(get_hf_hook(f"{hf_block_prefix}_Attn", f"{hf_block_prefix}_Attn")))
            if hasattr(block_to_hook.attn, "qkv"):
                def qkv_hook_dynamic(module, inputs, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(out_tensor, torch.Tensor):
                        q, k, v = out_tensor.chunk(3, dim=-1)
                        _print_hf_tensor_stats(q, f"{hf_block_prefix}_QKV_Q", f"{hf_block_prefix}_QKV_Q")
                        _print_hf_tensor_stats(k, f"{hf_block_prefix}_QKV_K", f"{hf_block_prefix}_QKV_K")
                        _print_hf_tensor_stats(v, f"{hf_block_prefix}_QKV_V", f"{hf_block_prefix}_QKV_V")
                hooks.append(block_to_hook.attn.qkv.register_forward_hook(qkv_hook_dynamic))
        
        if hasattr(block_to_hook, "norm2"):
            hooks.append(block_to_hook.norm2.register_forward_hook(get_hf_hook(f"{hf_block_prefix}_Norm2", f"{hf_block_prefix}_Norm2")))
        
        if hasattr(block_to_hook, "mlp"):
            hooks.append(block_to_hook.mlp.register_forward_hook(get_hf_hook(f"{hf_block_prefix}_MLP", f"{hf_block_prefix}_MLP")))
    else:
        print(f"WARN: target_block_idx {target_block_idx} is out of bounds for HF model with {len(hf_vis.blocks)} blocks.")

    if hasattr(hf_vis, "merger"):
        hf_final_merger = hf_vis.merger
        hooks.append(hf_final_merger.register_forward_hook(get_hf_hook("HF_FinalMerger", "HF_FinalMerger")))
        if hasattr(hf_final_merger, 'ln_q'):
            hooks.append(hf_final_merger.ln_q.register_forward_hook(get_hf_hook("HF_FinalMerger_Norm", "HF_FinalMerger_Norm")))
        if hasattr(hf_final_merger, 'mlp') and isinstance(hf_final_merger.mlp, torch.nn.Sequential) and len(hf_final_merger.mlp) == 3:
            hooks.append(hf_final_merger.mlp[0].register_forward_hook(get_hf_hook("HF_FinalMerger_MLP_Linear1", "HF_FinalMerger_MLP_Linear1")))
            hooks.append(hf_final_merger.mlp[1].register_forward_hook(get_hf_hook("HF_FinalMerger_MLP_GELU", "HF_FinalMerger_MLP_GELU")))


    print("DEBUG HF: img_torch.shape:", img_torch.shape)
    print("DEBUG HF: grid_thw:", grid_thw_torch)

    print("\n--- Running HuggingFace Model ---")
    hf_out_np = None
    try:
        with torch.no_grad():
            hf_out = hf_vis(img_torch, grid_thw=grid_thw_torch)
            _print_hf_tensor_stats(hf_out, "HF_FinalOutput", "HF_FinalOutput")
            hf_out_np = hf_out.cpu().numpy().astype("float32")
            print("Called HF vision encoder with img_torch and grid_thw.")
    finally:
        for hook in hooks:
            hook.remove()


    print("\n--- Running Fartsovka Model ---")
    
    patch_embed_out_fs = fs_vis.patch_embed(img_jax)
    _print_fs_tensor_stats(np.asarray(patch_embed_out_fs), "FS_PatchEmbed_Output") # This logs to test's fs_activations

    fs_result = fs_vis(img_jax, grid_thw=grid_thw_jax) # Cleaned call
    
    fs_out_final_np = None
    if hasattr(fs_result, "output") and fs_result.output is not None:
        fs_out_final_np = np.asarray(fs_result.output).astype("float32")
    elif isinstance(fs_result, jnp.ndarray):
        fs_out_final_np = np.asarray(fs_result).astype("float32")
    elif isinstance(fs_result, tuple) and len(fs_result) > 0 and isinstance(fs_result[0], jnp.ndarray):
        fs_out_final_np = np.asarray(fs_result[0]).astype("float32")
    else:
        raise ValueError(f"Unexpected output type from Fartsovka: {type(fs_result)}")
    
    _print_fs_tensor_stats(fs_out_final_np, "FS_FinalOutput", "FS_FinalOutput")
    
    # --- Final Output Shape and Difference Comparison ---
    if hf_out_np is not None and fs_out_final_np is not None:
        print(f"\nHF final output shape: {hf_out_np.shape}")
        print(f"FS final output shape: {fs_out_final_np.shape}")
        if hf_out_np.shape == fs_out_final_np.shape:
            final_diff = np.max(np.abs(fs_out_final_np - hf_out_np))
            print(f"Vision-Encoder final output max |Δ|: {float(final_diff):.4g}")
        else:
            print("Vision-Encoder final output shape MISMATCH!")
    else:
        print("Could not compare final outputs as one of them is None.")

    print(f"\n--- Intermediate Value Comparison (HF Block {target_block_idx} vs specific FS points) ---")
    
    comparison_pairs = [
        ("HF_PatchEmbed_Output", "FS_PatchEmbed_Output"),
        (f"{hf_block_prefix}_Output", None),
        ("HF_FinalOutput", "FS_FinalOutput")
    ]

    for hf_key, fs_key_or_none in comparison_pairs:
        print(f"\nComparing: {hf_key} (HF)", end="")
        hf_val = hf_activations.get(hf_key)

        fs_val = None
        if fs_key_or_none:
            print(f" vs {fs_key_or_none} (FS)", end="")
            fs_val = fs_activations.get(fs_key_or_none)
        print()

        if hf_val is None and (fs_key_or_none is None or fs_val is None):
            print(f"At least one value is None or FS key not specified. HF Key exists: {hf_key in hf_activations}, FS Key specified: {fs_key_or_none is not None}, FS Key exists: {fs_key_or_none and fs_key_or_none in fs_activations}")
            continue
        elif hf_val is None:
            print(f"HF value is None (key: {hf_key}). FS value type: {type(fs_val) if fs_val is not None else 'None'}")
            continue
        elif fs_key_or_none and fs_val is None:
            print(f"FS value is None (key: {fs_key_or_none}). HF value type: {type(hf_val)}")

        hf_np = np.asarray(hf_val).astype(np.float32)
        print(f"HF ({hf_key}): Shape {hf_np.shape}, Mean {hf_np.mean():.4g}, RMS {np.sqrt(np.mean(hf_np**2)):.4g}")
        print(f"HF first 5: {hf_np.flatten()[:5]}")

        if fs_val is not None:
            fs_np = np.asarray(fs_val).astype(np.float32)
            print(f"FS ({fs_key_or_none}): Shape {fs_np.shape}, Mean {fs_np.mean():.4g}, RMS {np.sqrt(np.mean(fs_np**2)):.4g}")
            print(f"FS first 5: {fs_np.flatten()[:5]}")
            if hf_np.shape == fs_np.shape:
                val_diff = np.max(np.abs(hf_np - fs_np))
                print(f"Max |Δ|: {float(val_diff):.6g}")
            else:
                print("Shape Mismatch!")
        else:
            print(f"FS value for {fs_key_or_none} not found or not specified.")

    print(f"--- End Intermediate Value Comparison ---")


def test_window_index_parity(huggingface_qwen25vl, fartsovka_qwen25vl_vision):
    """Check that get_window_index is identical in HF and FS."""
    grid_thw_np = np.array([[1, 16, 16]], dtype=np.int64)
    grid_thw_jax = jnp.array(grid_thw_np)
    grid_thw_torch = torch.tensor(grid_thw_np, dtype=torch.long)

    # --- FS
    win_idx_fs, cu_seqlens_fs = fartsovka_qwen25vl_vision.get_window_index(grid_thw_jax)
    win_idx_fs = np.asarray(win_idx_fs)
    cu_seqlens_fs = np.asarray(cu_seqlens_fs)

    hf_vis = getattr(huggingface_qwen25vl, "visual", None) or getattr(huggingface_qwen25vl, "vision_tower")
    win_idx_hf, cu_seqlens_hf = hf_vis.get_window_index(grid_thw_torch)
    win_idx_hf = win_idx_hf.cpu().numpy() if hasattr(win_idx_hf, 'cpu') else np.asarray(win_idx_hf)
    cu_seqlens_hf = cu_seqlens_hf.cpu().numpy() if hasattr(cu_seqlens_hf, 'cpu') else np.asarray(cu_seqlens_hf)

    assert np.array_equal(win_idx_fs, win_idx_hf), (
        f"window_index mismatch:\nFS: {win_idx_fs[:10]}...\nHF: {win_idx_hf[:10]}..."
    )
    assert np.array_equal(cu_seqlens_fs, cu_seqlens_hf), (
        f"cu_seqlens mismatch:\nFS: {cu_seqlens_fs}\nHF: {cu_seqlens_hf}"
    )

    print("window_index and cu_seqlens match perfectly!")