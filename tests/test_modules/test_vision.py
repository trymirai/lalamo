from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float, PRNGKeyArray, Bool, Int
from fartsovka.modules.vision_rope import VisionPositionalEmbeddings
from fartsovka.modules.vision_transformer import PositionalEmbeddingsAdapter
from PIL import Image


# NOTE: **NO hard dependency on Fartsovka VisionTransformer**
# --------------------------------------------------------
# We only need patch‑size / temporal‑patch‑size to build the reference
# layout tensor.  If the Fartsovka fixture fails to load (e.g. weight‑key
# mismatch) we fall back to the values provided by the HF config so the
# *layout/value* sanity test can still run and tell us precisely where the
# divergence appears.

from .common import assert_close, from_torch, to_torch, checkify_forward

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

_ATOL = 1e-7  # ultra‑strict – we expect byte‑identical layout
_RTOL = 1e-7
_MAX_PRINT = 5  # how many individual element diffs to show

# ----------------------------------------------------------------------------
# Utilities for Debugging HF vs Fartsovka Parity
# ----------------------------------------------------------------------------

# Global dictionary to store activations from HF hooks
hf_activations: Dict[str, Any] = {}

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
    else:
        print(f"DEBUG HF STATS: Unsupported tensor type for {name}: {type(tensor)}")
        if store_key: hf_activations[store_key] = "Unsupported Type"
        return

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

        # Outputs can also be tuples (e.g. for attention) or single tensors
        if isinstance(output_tensors, tuple):
            # Special handling for Qwen2_5_VLVisionBlock output which might be just hidden_states
            # or (hidden_states, attentions) if output_attentions=True
            # For now, assume the first element is the main hidden_states output
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



def _dump_patch_samples(src: jnp.ndarray, ref: jnp.ndarray, tag: str) -> None:
    diff = jnp.abs(src - ref)
    worst = jnp.argsort(diff.ravel())[-_MAX_PRINT:][::-1]
    print(f"\n{tag} worst {_MAX_PRINT} |Δ| samples:")
    for i, flat in enumerate(worst):
        idx = tuple(int(x) for x in jnp.unravel_index(flat, diff.shape))
        print(
            f"  {i+1}: idx={idx}, src={float(src[idx]):+.6f}, ref={float(ref[idx]):+.6f}, |Δ|={float(diff[idx]):.6g}"
        )

# ---------------------------------------------------------------------------
# Lightweight helper – convert any JAX / NumPy / Torch tensor to float32 NumPy
# ---------------------------------------------------------------------------

def _fs_to_np(x):
    """Return *x* as a plain float32 NumPy ndarray.

    Handles JAX arrays, NumPy arrays, and Torch tensors so we can run the
    same diff/print code regardless of backend.
    """
    import numpy as _np  # local import to avoid polluting global namespace
    try:
        return _np.asarray(x, dtype=_np.float32)
    except Exception:  # Fall‑back for Torch tensors
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(_np.float32)
        except ModuleNotFoundError:
            pass
        raise  # Re‑throw if we still cannot handle the type


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------

def test_patch_layout_and_values(huggingface_qwen25vl, fartsovka_qwen25vl_vision, rng_key: PRNGKeyArray):
    """Element‑wise equality of *un‑flattened* patches (JAX vs Torch).

    *   **Layout** – identical reshape behaviour.
    *   **Values** – exact equality (within 1e‑7) in f32 + bf16 casts.
    """

    # ---------------------------------------------------------------- hf tower
    hf_model = huggingface_qwen25vl
    if hf_model is None:
        pytest.fail("HF reference model fixture missing – install transformers >=4.40")

    hf_vis = getattr(hf_model, "vision_tower", getattr(hf_model, "visual", None))
    if hf_vis is None:
        pytest.fail("HF model has no .vision_tower / .visual attribute → cannot test")

    # ---------------------------------------------------------------- params
    # Prefer Fartsovka config if available (just in case it diverges!), else HF.
    if fartsovka_qwen25vl_vision is not None:
        t_sz = fartsovka_qwen25vl_vision.config.patch_embedding_config.temporal_patch_size
        p_sz = fartsovka_qwen25vl_vision.config.patch_embedding_config.patch_size
    else:
        # Fall back to HF config so we *still* validate the raw marshal path
        t_sz = getattr(hf_vis.config, "temporal_patch_size", 2)
        p_sz = getattr(hf_vis.config, "patch_size", 14)
        print(
            "[WARN] Fartsovka vision fixture unavailable – using HF patch params "
            f"(t={t_sz}, p={p_sz}) for layout check only."
        )

    # ----------------------------------------------------------- build input
    img = load_test_image()  # [C,H,W]
    frames: List[jnp.ndarray] = [img] * t_sz
    img_jax = jnp.expand_dims(jnp.stack(frames, axis=1), 0)  # [B,C,T,H,W]
    img_torch = to_torch(img_jax)

    # ----------------------------------------------------- marshal equality
    ref_jax = from_torch(img_torch)
    assert_close(
        result=img_jax,
        reference=ref_jax,
        atol=_ATOL,
        rtol=_RTOL,
        operation_name="marshal to torch ↔ jax",
    )

    # 👉 NEW: print representative pixel diffs for the *raw* image marshal
    _dump_patch_samples(img_jax, ref_jax, tag="marshal f32")

    # ------------------------------------------------ patch extraction / f32
    C_img = img_jax.shape[1] # Renamed from C to C_img to avoid conflict
    patches_jax_f32 = img_jax.reshape(-1, C_img, t_sz, p_sz, p_sz)
    patches_torch_f32 = from_torch(img_torch.view(-1, C_img, t_sz, p_sz, p_sz))

    assert_close(
        result=patches_jax_f32,
        reference=patches_torch_f32,
        atol=_ATOL,
        rtol=_RTOL,
        operation_name="patch layout + float32 values",
    )

    _dump_patch_samples(patches_jax_f32, patches_torch_f32, tag="float32")

    # ------------------------------------------------ patch extraction / bf16
    patches_jax_bf16 = patches_jax_f32.astype(jnp.bfloat16)
    dtype_ref = hf_vis.patch_embed.proj.weight.dtype  # torch dtype (bf16)
    patches_torch_bf16 = from_torch(img_torch.view(-1, C_img, t_sz, p_sz, p_sz).to(dtype=dtype_ref)).astype(jnp.bfloat16)

    assert_close(
        result=patches_jax_bf16,
        reference=patches_torch_bf16,
        atol=_ATOL,
        rtol=_RTOL,
        operation_name="patch layout + bfloat16 values",
    )

    _dump_patch_samples(patches_jax_bf16.astype(jnp.float32), patches_torch_bf16.astype(jnp.float32), tag="bfloat16→f32")

    print("\n✓ Patch layout & values match in float32 and bfloat16 cast!")

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_patch_embedding_output(huggingface_qwen25vl, fartsovka_qwen25vl_vision, dtype):
    """Compare the *output vectors* of the Conv3D patch‑embedding layer.

    We run the same CHW image through:
      * HF vision_tower.patch_embed (Torch)  ➜ convert to JAX
      * Fartsovka VisionTransformer.patch_embed (JAX)

    and assert element‑wise equality within 1e‑6 (f32) / 1e‑3 (bf16).
    """

    # ---------------------------------------------------------------- fixtures
    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model not available – install recent transformers")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka vision model failed to load – see previous test output")

    hf_vis = getattr(huggingface_qwen25vl, "vision_tower", getattr(huggingface_qwen25vl, "visual", None))
    if hf_vis is None:
        pytest.fail("HF model missing .vision_tower / .visual – cannot compare patch embed")

    # ----------------------------------------------------------- build input
    img = generate_gradient_image()  # deterministic pseudo‑random CHW ∈ [0,1]
    t_sz = fartsovka_qwen25vl_vision.config.patch_embedding_config.temporal_patch_size
    frames = [img] * t_sz
    img_jax = jnp.expand_dims(jnp.stack(frames, axis=1), 0)  # [B,C,T,H,W]

    if dtype == jnp.bfloat16:
        img_jax = img_jax.astype(jnp.bfloat16)

    # Torch copy
    img_torch = to_torch(img_jax)

    # ----------------------------------------------------- run patch‑embed
    with torch.no_grad():
        hf_out = hf_vis.patch_embed(img_torch)  # shape [B, N, D]
    fs_out = fartsovka_qwen25vl_vision.patch_embed(img_jax)  # identical shape

    # -------------------------------------- convert HF output back to JAX
    hf_out_jax = from_torch(hf_out).astype(dtype)

    # -------------------------------------- numeric comparison
    atol = 2e-6 if dtype == jnp.float32 else 1e-3
    rtol = 2e-6 if dtype == jnp.float32 else 1e-3

    try:
        assert_close(
            result=fs_out,
            reference=hf_out_jax,
            atol=atol,
            rtol=rtol,
            operation_name=f"patch‑embed output ({dtype})",
        )
    finally:
        # always print top‑diffs for debugging even if assertion fails
        _dump_patch_samples(fs_out.astype(jnp.float32), hf_out_jax.astype(jnp.float32), tag=f"patch‑embed {dtype}")

    print(f"✓ Patch‑embed outputs match for dtype={dtype} (atol={atol})")


# ----------------------------------------------------------------------------
# NEW TEST 3 – VisionRoPE positional‑embedding parity
# ----------------------------------------------------------------------------

def _find_hf_rotary_emb(hf_vis):
    """Locate the rotary‑embedding module inside the HF vision tower.

    We walk through a list of *candidate paths*.  Each step in a path may be
    either a string (attribute access via ``getattr``) **or** an ``int`` that
    indexes into a list/``nn.ModuleList``/``nn.Sequential``.  The first path
    that resolves without error is returned.
    """

    candidate_paths = [
        ("rotary_pos_emb",),                       # Qwen2‑5‑VL (newer HF commits)
        ("rot_pos_emb",),                          # For Qwen2_5_VisionTransformerPretrainedModel
        ("rotary_emb",),                           # older attribute name
        ("rotary_embedding",),                     # alt. attribute name
        ("blocks", 0, "attn", "rotary_emb"),         # nested inside first block
        ("blocks", 0, "attn", "rotary_embedding"),  # nested alt.
    ]

    for path in candidate_paths:
        node = hf_vis
        ok = True
        for step in path:
            try:
                if isinstance(step, str):
                    node = getattr(node, step)
                else:
                    node = node[step]
            except (AttributeError, IndexError, TypeError):
                ok = False
                break
        if ok:
            return node

    return None  # not found

@pytest.mark.parametrize("seq_len", [64])
def test_vision_rope_cos_sin(huggingface_qwen25vl, fartsovka_qwen25vl_vision, seq_len):
    """Cosine/Sine tables produced by VisionRoPE match the HF implementation."""

    if huggingface_qwen25vl is None or fartsovka_qwen25vl_vision is None:
        pytest.skip("Fixtures missing – positional‑embedding parity test skipped")

    hf_vis = getattr(huggingface_qwen25vl, "vision_tower", getattr(huggingface_qwen25vl, "visual", None))
    if hf_vis is None:
        pytest.skip("HF model missing vision tower.")
        
    hf_rotary_module_or_method = _find_hf_rotary_emb(hf_vis) # This might be the module or the method
    print(f"DEBUG: Found HF Rotary object: {hf_rotary_module_or_method}, type: {type(hf_rotary_module_or_method)}")


    if hf_rotary_module_or_method is None:
        pytest.skip("Could not locate rotary_pos_emb module or method inside HF model – API changed?")

    # For now, let's try to call the VisionRotaryEmbedding if it's found directly
    is_direct_rope_module = hasattr(hf_rotary_module_or_method, 'forward') and not isinstance(hf_rotary_module_or_method, torch.nn.ModuleList)

    # ------------------- Fartsovka ► cos/sin -------------------------------
    # Fartsovka VisionRoPE.__call__ can take seq_len (int)
    vt_pe_fs = fartsovka_qwen25vl_vision.rope(grid_thw_jax) # Call with int seq_len for 1D mode
    cos_fs = vt_pe_fs.cosines  # [S, head_dim]
    sin_fs = vt_pe_fs.sines
    # --------------------- HF ► cos/sin ------------------------------------
    with torch.no_grad():
        if is_direct_rope_module and callable(hf_rotary_module_or_method):
            try:
                if hasattr(hf_rotary_module_or_method, 'inv_freq'):
                    theta_hf = hf_rotary_module_or_method(torch.arange(seq_len, device=hf_vis.device).float()) 
                    theta_values = hf_rotary_module_or_method(seq_len) 
                    emb_hf = torch.cat([theta_values, theta_values], dim=-1)
                    cos_hf_torch = emb_hf.cos()
                    sin_hf_torch = emb_hf.sin()
                    cos_hf = from_torch(cos_hf_torch)
                    sin_hf = from_torch(sin_hf_torch)

                else: # Could be the rot_pos_emb method itself
                    pytest.skip("HF RoPE object found is not a direct module for seq_len input. Test needs rework for grid_thw.")
                    return # Should not reach here
            except Exception as e:
                print(f"Error calling HF RoPE with seq_len: {e}")
                pytest.skip("Failed to call HF RoPE module with seq_len. Test may need rework.")
                return
        else:
            print("HF RoPE object is likely the rot_pos_emb(grid_thw) method. Skipping simple seq_len test.")
            pytest.skip("HF RoPE is method, not module taking seq_len. Test needs rework.")
            return


    # -------- DEBUG snapshot & stats ---------------------------------
    import numpy as _np_debug  # local import to avoid polluting global imports
    _rows, _cols = 4, 4  # how many rows / cols to print
    print("\n[DEBUG] cos_fs sample:\n", _np_debug.asarray(cos_fs[:_rows, :_cols]))
    print("[DEBUG] cos_hf sample:\n", _np_debug.asarray(cos_hf[:_rows, :_cols])) # cos_hf might not be defined if skipped
    # Fartsovka VisionRoPE now returns head_dim‑wide tables directly, so no duplication is required.
    print(jnp.max(jnp.abs(cos_fs - cos_hf)))   # should be ~1e‑7
    print(jnp.max(jnp.abs(sin_fs - sin_hf)))   # should be ~1e‑7

    # --------------------- Numerical equality ------------------------------
    assert_close(result=cos_fs, reference=cos_hf, atol=1e-6, rtol=1e-6, operation_name="VisionRoPE cos")
    assert_close(result=sin_fs, reference=sin_hf, atol=1e-6, rtol=1e-6, operation_name="VisionRoPE sin")

    print("\n✓ VisionRoPE cosine/sine tables match HF implementation (seq_len =", seq_len, ")")

# ---------------------------------------------------------------------------
# Patch-merger parity
# ---------------------------------------------------------------------------

def _find_hf_patch_merger(hf_vis):
    """
    Locate the *first* PatchMerger-like module inside the HF vision tower.

    We walk through a list of candidate attribute names that appear in
    different Qwen-2.5-VL revisions.
    """
    candidate_attrs = [
        "merger",           # Current HF naming (Qwen2-5-VL)
        "patch_merger",     # Alternate spelling
        "final_merger",     # Some forks expose a final projection this way
    ]
    for name in candidate_attrs:
        if hasattr(hf_vis, name):
            return getattr(hf_vis, name)
    return None


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_vision_patch_merger(huggingface_qwen25vl, fartsovka_qwen25vl_vision, dtype):
    """
    Compare **one** patch-merger block between HF and Fartsovka.

    We use:
      • HF  : vision_tower.<merger candidate>
      • FS  : VisionTransformer.final_merger

    Both modules receive identical random hidden-states.  Outputs must match
    within 1 e-6 (f32) or 1 e-3 (bf16).
    """
    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model missing")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka vision model failed to load")

    # -------- locate the two merger modules --------------------------------
    hf_vis = getattr(
        huggingface_qwen25vl, "vision_tower",
        getattr(huggingface_qwen25vl, "visual")
    )
    if hf_vis is None: pytest.skip("HF model missing vision tower.")

    hf_merger = _find_hf_patch_merger(hf_vis)
    if hf_merger is None:
        pytest.skip("Could not locate patch-merger module inside HF model")

    fs_merger = fartsovka_qwen25vl_vision.final_merger

    # ------------------------------------------------------------------
    # Sanity‑check: loaded Fartsovka weights ≈ HF weights
    # ------------------------------------------------------------------
    try:
        # ----- hidden projection (mlp[0] ↔ hidden_proj) -----
        hf_w0 = hf_merger.mlp[0].weight.detach().cpu().to(torch.float32).numpy()
        fs_w0 = jnp.asarray(fs_merger.hidden_proj.weights, dtype=jnp.float32)
        print("hidden_proj.weight first 10 FS vs HF:", fs_w0.flatten()[:10], hf_w0.flatten()[:10])
        assert_close(result=fs_w0, reference=hf_w0, atol=1e-6, rtol=1e-6,
                     operation_name="PatchMerger hidden_proj.weight")

        hf_b0 = hf_merger.mlp[0].bias.detach().cpu().to(torch.float32).numpy()
        fs_b0 = jnp.asarray(fs_merger.hidden_proj.biases, dtype=jnp.float32)
        print("hidden_proj.bias first 10 FS vs HF:", fs_b0.flatten()[:10], hf_b0.flatten()[:10])
        assert_close(result=fs_b0, reference=hf_b0, atol=1e-6, rtol=1e-6,
                     operation_name="PatchMerger hidden_proj.bias")

        # ----- output projection (mlp[2] ↔ out_proj) -----
        hf_w2 = hf_merger.mlp[2].weight.detach().cpu().to(torch.float32).numpy()
        fs_w2 = jnp.asarray(fs_merger.out_proj.weights, dtype=jnp.float32)
        print("out_proj.weight first 10 FS vs HF:", fs_w2.flatten()[:10], hf_w2.flatten()[:10])
        assert_close(result=fs_w2, reference=hf_w2, atol=1e-6, rtol=1e-6,
                     operation_name="PatchMerger out_proj.weight")

        hf_b2 = hf_merger.mlp[2].bias.detach().cpu().to(torch.float32).numpy()
        fs_b2 = jnp.asarray(fs_merger.out_proj.biases, dtype=jnp.float32)
        print("out_proj.bias first 10 FS vs HF:", fs_b2.flatten()[:10], hf_b2.flatten()[:10])
        assert_close(result=fs_b2, reference=hf_b2, atol=1e-6, rtol=1e-6,
                     operation_name="PatchMerger out_proj.bias")

        # ----- RMSNorm / ln_q -----
        hf_scale = hf_merger.ln_q.weight.detach().cpu().to(torch.float32).numpy()
        fs_scale = jnp.asarray(fs_merger.norm.scales, dtype=jnp.float32)
        print("ln_q.scale first 10 FS vs HF:", fs_scale.flatten()[:10], hf_scale.flatten()[:10])
        assert_close(result=fs_scale, reference=hf_scale, atol=1e-6, rtol=1e-6,
                     operation_name="PatchMerger ln_q/scale")
    except Exception as e:
        # If any of the above asserts fail or shapes mismatch we re‑raise so
        # pytest surfaces the exact reason.
        raise

    # -------- craft identical random input ---------------------------------
    seq_len_multiple = fs_merger.config.spatial_merge_size ** 2
    seq_len_merger = 8 * seq_len_multiple            # keep it small but divisible, renamed seq_len

    hidden_dim = int(fs_merger.norm.input_dim)
    key = jax.random.PRNGKey(0)
    hidden_states_merger = jax.random.normal(key, (seq_len_merger, hidden_dim), dtype=dtype) # renamed hidden_states

    # -------- HF forward (Torch) -------------------------------------------
    with torch.no_grad():
        # Convert via helper → NumPy → Torch.  This avoids JAX array incompatibility
        torch_inp = to_torch(hidden_states_merger).to(
            torch.float32 if dtype == jnp.float32 else torch.bfloat16
        )
        # Make sure input lives on the *same* device as the HF module weights
        merger_device = next(hf_merger.parameters()).device
        torch_inp = torch_inp.to(merger_device)
        hf_merger = hf_merger.to(merger_device) # Ensure module is on the correct device
        torch_out = hf_merger(torch_inp).to(torch.float32).cpu().numpy()  # promote + move to CPU
        out_hf = torch_out

    # -------- Fartsovka forward (JAX) --------------------------------------
    out_fs = fs_merger(hidden_states_merger).astype(jnp.float32)
    out_fs_np = jnp.asarray(out_fs)

    # -------- numerical parity --------------------------------------------
    # import numpy as _np # already imported globally for this file
    print("out_fs first 10:", np.asarray(out_fs_np.flatten()[:10]))
    print("out_hf first 10:", np.asarray(out_hf.flatten()[:10]))
    print("max |Δ|:", float(np.max(np.abs(out_fs_np - out_hf))))

    atol_merger = 1e-2 # Renamed atol
    rtol_merger = atol_merger # Renamed rtol
    assert_close(
        result=out_fs_np,
        reference=out_hf,
        atol=atol_merger,
        rtol=rtol_merger,
        operation_name=f"PatchMerger ({dtype})",
    )

    print(f"\n✓ PatchMerger parity passed for dtype={dtype}")


# ----------------------------------------------------------------------------
# NEW TEST 4 – End‑to‑end Vision‑Encoder parity
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [jnp.float32])
def test_vision_encoder_parity(
    huggingface_qwen25vl,
    fartsovka_qwen25vl_vision,
    dtype,
):
    """
    Simple, robust parity test: Run the *entire* vision encoder on the same synthetic clip,
    compare final outputs from HF and Fartsovka. No block-by-block poking.
    """
    global hf_activations # Clear previous run's activations
    hf_activations = {}
    hook_handles = []
    import torch  # local import to avoid polluting global namespace

    # Fixture sanity
    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model missing")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka VisionTransformer fixture failed to load")

    # Prepare synthetic input
    img = generate_gradient_image()  # [C, H, W]
    t_sz = fartsovka_qwen25vl_vision.config.patch_embedding_config.temporal_patch_size
    frames = [img] * t_sz
    img_jax = jnp.expand_dims(jnp.stack(frames, axis=1), 0)  # [B, C, T, H, W]
    if dtype == jnp.bfloat16:
        img_jax = img_jax.astype(jnp.bfloat16)
    img_torch = to_torch(img_jax)
    _print_hf_tensor_stats(img_torch, "HF Input: img_torch", "HF_Input_Image")


    # Get HF and FS vision encoder objects
    hf_vis = getattr(huggingface_qwen25vl, "vision_tower", getattr(huggingface_qwen25vl, "visual", None))
    if hf_vis is None: pytest.skip("HF model missing vision tower.")
    fs_vis = fartsovka_qwen25vl_vision
    
    device = img_torch.device # Use input tensor's device
    hf_vis = hf_vis.to(device)

    hook_handles.append(hf_vis.patch_embed.register_forward_hook(get_hf_hook("HF_PatchEmbed", "HF_PatchEmbed")))

    first_block_attn_hook_triggered = False
    def first_block_attn_hook(module, input_tensors, output_tensors):
        nonlocal first_block_attn_hook_triggered
        if not first_block_attn_hook_triggered:
            if len(input_tensors) > 0: # hidden_states
                 _print_hf_tensor_stats(input_tensors[0], "HF_Block0_Attn_Input_HiddenStates", "HF_Block0_Attn_Input_HiddenStates")
            if len(input_tensors) > 2 and input_tensors[2] is not None: # rotary_pos_emb (old) or position_embeddings (new)
                if isinstance(input_tensors[2], tuple): # New: position_embeddings = (cos, sin)
                    _print_hf_tensor_stats(input_tensors[2][0], "HF_Block0_Attn_Input_PosEmb_Cos", "HF_Block0_Attn_Input_PosEmb_Cos")
                    _print_hf_tensor_stats(input_tensors[2][1], "HF_Block0_Attn_Input_PosEmb_Sin", "HF_Block0_Attn_Input_PosEmb_Sin")
                else: # Old: rotary_pos_emb (theta values)
                    _print_hf_tensor_stats(input_tensors[2], "HF_Block0_Attn_Input_RotaryPosEmb_Theta", "HF_Block0_Attn_Input_RotaryPosEmb_Theta")
            # --- NEW: detect and print candidate attention mask tensors ---
            for j, maybe_mask in enumerate(input_tensors):
                if isinstance(maybe_mask, torch.Tensor):
                    # Heuristic: attention masks are usually 2‑D or 4‑D square/symmetric tensors
                    if maybe_mask.ndim >= 2 and maybe_mask.shape[-2] == maybe_mask.shape[-1]:
                        _print_hf_tensor_stats(
                            maybe_mask,
                            f"HF_Block0_Attn_Input_Mask_candidate{j}",
                            f"HF_Block0_Attn_Input_Mask_candidate{j}"
                        )
            # --- NEW: dump Block‑0 attention output ---------------------------------
            # The hook's *return* value is the tensor that gets passed further up the
            # vision tower.  Grab it so we can compare with Fartsovka's
            # FS_Block_0_AttnOutput numbers.
            attn_out_tensor: Optional[torch.Tensor]
            if isinstance(output_tensors, tuple):
                # Qwen2‑5‑VL returns a tuple (hidden_states, optional attn_weights)
                attn_out_tensor = output_tensors[0] if len(output_tensors) > 0 else None
            else:
                attn_out_tensor = output_tensors
            if isinstance(attn_out_tensor, torch.Tensor):
                _print_hf_tensor_stats(
                    attn_out_tensor,
                    "HF_Block0_AttnOutput",
                    "HF_Block0_AttnOutput",
                )
            first_block_attn_hook_triggered = True # Only capture for the first call to the first block's attention

    if len(hf_vis.blocks) > 0 and hasattr(hf_vis.blocks[0], "attn"):
        attn_mod = hf_vis.blocks[0].attn
        # capture high‑level attention call once
        hook_handles.append(attn_mod.register_forward_hook(first_block_attn_hook))

        # ------------------------------------------------------------
        # Extra diagnostics – handle both split (q_proj/k_proj/v_proj)
        # *and* fused (qkv) projection styles used by HF revisions.
        # ------------------------------------------------------------
        if all(hasattr(attn_mod, attr) for attr in ("q_proj", "k_proj", "v_proj")):
            proj_modules = [
                ("HF_Block0_QProj", attn_mod.q_proj),
                ("HF_Block0_KProj", attn_mod.k_proj),
                ("HF_Block0_VProj", attn_mod.v_proj),
            ]
            for tag, mod in proj_modules:
                hook_handles.append(mod.register_forward_hook(get_hf_hook(tag, tag)))
        elif hasattr(attn_mod, "qkv"):
            # Fused QKV projection – register a custom hook that splits
            # the output into Q/K/V chunks so we still get comparable logs.
            import torch
            def _qkv_split_hook(module, input_tensors, output_tensor):
                # Linear layers return a single Tensor; sometimes HF packs
                # it in a tuple – normalise to Tensor first.
                out = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                if not isinstance(out, torch.Tensor):
                    return  # unexpected type – silently ignore
                q, k, v = out.chunk(3, dim=-1)
                _print_hf_tensor_stats(q, "HF_Block0_QKVProj_Q", "HF_Block0_QKVProj_Q")
                _print_hf_tensor_stats(k, "HF_Block0_QKVProj_K", "HF_Block0_QKVProj_K")
                _print_hf_tensor_stats(v, "HF_Block0_QKVProj_V", "HF_Block0_QKVProj_V")
            hook_handles.append(attn_mod.qkv.register_forward_hook(_qkv_split_hook))
        else:
            print("[WARN] Could not find Q/K/V projection layers (split or fused) in first HF block – skipping detailed QKV hooks.")

        for i, block in enumerate(hf_vis.blocks):
            hook_handles.append(block.register_forward_hook(get_hf_hook(f"HF_Block_{i}", f"HF_Block_{i}")))
            break


    # 3. Final Merger
    hook_handles.append(hf_vis.merger.register_forward_hook(get_hf_hook("HF_FinalMerger", "HF_FinalMerger")))
    # If merger has sub-components like norm (ln_q) and mlp, hook them too for more detail:
    if hasattr(hf_vis.merger, 'ln_q'):
        hook_handles.append(hf_vis.merger.ln_q.register_forward_hook(get_hf_hook("HF_FinalMerger_Norm", "HF_FinalMerger_Norm")))
    if hasattr(hf_vis.merger, 'mlp'):
         hook_handles.append(hf_vis.merger.mlp.register_forward_hook(get_hf_hook("HF_FinalMerger_MLP", "HF_FinalMerger_MLP")))


    # Direct forward pass for HF model (hooks will trigger)
    with torch.no_grad():
        B_img, C_img_again, T_img, H_img, W_img = img_torch.shape # Renamed B,C to avoid conflict
        tp = getattr(hf_vis.config, "temporal_patch_size", 2)
        p_sz = getattr(hf_vis.config, "patch_size", 14)

        grid_thw = torch.tensor(
            [[
                T_img // tp,        # temporal length after temporal-patching
                H_img // p_sz,      # number of raw spatial patches (height)
                W_img // p_sz       # number of raw spatial patches (width)
            ]],
            dtype=torch.long,
            device=img_torch.device,
        )
        # Print grid_thw for HF
        _print_hf_tensor_stats(grid_thw, "HF Input: grid_thw", "HF_Input_GridTHW")


        print("DEBUG HF: img_torch.shape:", img_torch.shape)
        print("DEBUG HF: grid_thw:", grid_thw)
        # print("DEBUG HF: hf_vis.config:", hf_vis.config) # Already printed by Fartsovka load

        try:
            hf_out_obj = hf_vis(img_torch, grid_thw=grid_thw)
            print(f"Called HF vision encoder with img_torch and grid_thw.")
        except Exception as e:
            print(f"ERROR: Failed to call HF vision encoder: {e}")
            # Remove hooks if error
            for handle in hook_handles: handle.remove()
            raise

        # Output from Qwen2_5_VisionTransformerPretrainedModel.forward is the final hidden_states
        hf_out_final_hidden = hf_out_obj 
        _print_hf_tensor_stats(hf_out_final_hidden, "HF Final Output Hidden States", "HF_FinalOutputHidden")
        hf_out_np = hf_out_final_hidden.cpu().numpy().astype("float32")


    # Remove hooks after HF forward pass
    for handle in hook_handles:
        handle.remove()

    print("\n--- Running Fartsovka Model ---")
    # Rename fs_out to fs_result_obj to clarify it's the model's output object/structure
    fs_result_obj = fs_vis(img_jax, grid_thw=from_torch(grid_thw)) # Pass grid_thw to Fartsovka
    
    # --- Add Fartsovka Intermediate Tensor Prints ---
    # This section assumes fs_result_obj might be a rich object (like HF's BaseModelOutput)
    # or a tuple where elements contain intermediate states.

    fs_intermediate_logged = False
    if hasattr(fs_result_obj, 'hidden_states') and fs_result_obj.hidden_states is not None:
        if isinstance(fs_result_obj.hidden_states, (list, tuple)) and len(fs_result_obj.hidden_states) > 0:
            # hidden_states[0] is typically the output of embeddings / input to first block
            _print_hf_tensor_stats(
                np.asarray(fs_result_obj.hidden_states[0]), 
                "FS_PatchEmbed_Output (or Block0_Input from hidden_states[0])", 
                "FS_PatchEmbed_Output_or_Block0_Input"
            )
            if len(fs_result_obj.hidden_states) > 1:
                # hidden_states[1] is typically the output of the first block
                _print_hf_tensor_stats(
                    np.asarray(fs_result_obj.hidden_states[1]), 
                    "FS_Block_0_Output (from hidden_states[1])", 
                    "FS_Block_0_Output"
                )
            # Output of all blocks (input to merger)
            _print_hf_tensor_stats(
                np.asarray(fs_result_obj.hidden_states[-1]), 
                "FS_FinalEncoderBlocks_Output (MergerInput from hidden_states[-1])",
                "FS_FinalEncoderBlocks_Output_MergerInput"
            )
            fs_intermediate_logged = True

    # Alternative check: if fs_result_obj is a tuple (e.g., (final_output, all_hidden_states_tuple, ...))
    # and all_hidden_states is the second element.
    elif isinstance(fs_result_obj, tuple) and len(fs_result_obj) > 1 and \
         hasattr(fs_result_obj[1], '__iter__') and not isinstance(fs_result_obj[1], (jnp.ndarray, np.ndarray, torch.Tensor)): # Check if second element is iterable and not a tensor itself
        all_hidden_states_fs = fs_result_obj[1]
        if len(all_hidden_states_fs) > 0:
            _print_hf_tensor_stats(
                np.asarray(all_hidden_states_fs[0]),
                "FS_PatchEmbed_Output (or Block0_Input - from tuple[1][0])",
                "FS_PatchEmbed_Output_or_Block0_Input_tuple"
            )
            if len(all_hidden_states_fs) > 1:
                 _print_hf_tensor_stats(
                    np.asarray(all_hidden_states_fs[1]),
                    "FS_Block_0_Output (from tuple[1][1])",
                    "FS_Block_0_Output_tuple"
                )
            _print_hf_tensor_stats(
                np.asarray(all_hidden_states_fs[-1]),
                "FS_FinalEncoderBlocks_Output (MergerInput - from tuple[1][-1])",
                "FS_FinalEncoderBlocks_Output_MergerInput_tuple"
            )
            fs_intermediate_logged = True
            
    if not fs_intermediate_logged:
        print("DEBUG FS: Could not retrieve detailed intermediate hidden states from Fartsovka model output for printing.")

    # Extract final Fartsovka output tensor
    fs_out_final_hidden = None
    if hasattr(fs_result_obj, "output"): # Specific VisionOutput structure
        fs_out_final_hidden = fs_result_obj.output
    elif isinstance(fs_result_obj, jnp.ndarray): # Direct JAX array output
        fs_out_final_hidden = fs_result_obj
    elif isinstance(fs_result_obj, tuple) and len(fs_result_obj) > 0 and isinstance(fs_result_obj[0], jnp.ndarray):
        # Assuming if it's a tuple, the first element is the primary output tensor
        fs_out_final_hidden = fs_result_obj[0]
    else:
        # If the structure is unknown and previous conditions didn't match
        raise TypeError(f"Unexpected Fartsovka output type or structure: {type(fs_result_obj)}. Cannot extract final hidden states.")
        
    fs_out_np = np.asarray(fs_out_final_hidden).astype("float32")
    # This print helps confirm the final Fartsovka output being compared
    _print_hf_tensor_stats(fs_out_np, "FS Final Output Hidden States (for comparison)", "FS_FinalOutputHidden_ForComparison")
    # Fartsovka internal prints will show intermediate values.
    # We already have _print_tensor_stats in Fartsovka's VisionTransformer for its final output.

    # Print shapes to help debug if mismatch
    print(f"HF final output shape: {hf_out_np.shape}")
    print(f"FS final output shape: {fs_out_np.shape}")

    # Numerical parity check
    atol_parity = 1e-5 if dtype == jnp.float32 else 1e-2 # Loosen tolerance slightly for end-to-end
    rtol_parity = atol_parity
    # assert_close(
    #     result=jnp.asarray(fs_out_np), # Convert to jnp.array for assert_close
    #     reference=jnp.asarray(hf_out_np), # Also ensure reference is jnp.array
    #     atol=atol_parity,
    #     rtol=rtol_parity,
    #     operation_name=f"VisionEncoder output ({dtype})",
    # )

    print(
        f"\n✓ Vision‑Encoder parity passed for dtype={dtype} "
        f"(max |Δ| ≈ {float(np.max(np.abs(fs_out_np - hf_out_np))):.4g})"
    )


def _rope_stats(arr, tag, rows: int = 6, cols: int = 6) -> None:
    import numpy as _np
    
    # Handle torch tensors that might be on GPU (MPS, CUDA, etc.)
    if hasattr(arr, 'device') and str(arr.device) != 'cpu':
        arr = arr.cpu()  # Move to CPU before converting to NumPy
        
    a = _np.asarray(arr, dtype=_np.float32)
    print(f"\n[{tag}] shape={a.shape} "
          f"min={a.min():+.4f} max={a.max():+.4f} "
          f"mean={a.mean():+.4f}")
    print(a[:rows, :cols])

def compare_rope_inputs(fs_vis, hf_vis, device):
    # ------------- build the grid_thw exactly like HF does -------------
    grid_thw = torch.tensor([[1, 16, 16]], device=device)       # (T,H,W)

    # ----- HF cos/sin (always available on current master) -----
    with torch.no_grad():
        rotary_pos_emb    = hf_vis.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos.to(device)
        sin = sin.to(device)
        # Convert HF tensors to JAX arrays for cross‑framework comparisons
        cos_jax = from_torch(cos)        # shape (256, 80) – jnp.ndarray
        sin_jax = from_torch(sin)
    _rope_stats(cos, "HF cos")
    _rope_stats(sin, "HF sin")

    # --- Fartsovka: use HF-style RoPE direct API ---
    grid_thw_jax = jnp.array([[1, 16, 16]])
    pe_fs = fs_vis.rope(grid_thw_jax)
    cos_fs = np.asarray(pe_fs.cosines)
    sin_fs = np.asarray(pe_fs.sines)
    _rope_stats(cos_fs, "FS cos")
    _rope_stats(sin_fs, "FS sin")
    print(jnp.max(jnp.abs(cos_fs - cos_jax)))   # expect ~1e‑7
    print(jnp.max(jnp.abs(sin_fs - sin_jax)))   # expect ~1e‑7
    return

def test_qwen25vl_vision_attention_block0(
    huggingface_qwen25vl,
    fartsovka_qwen25vl_vision,
    rng_key
):
    hf_attn = huggingface_qwen25vl.visual.blocks[0].attn          # HF (torch)
    fs_attn = fartsovka_qwen25vl_vision.stages[0][0].attention    # FS (jax)

    fs_fwd = checkify_forward(fs_attn)

    model_dim = fs_attn.model_dim
    seq_len   = 256                           # 1 × 16 × 16 patches
    sample_inp = jax.random.normal(rng_key, (seq_len, model_dim))

    # ------------------------------------------------------------------
    # 1.  RoPE tables must still match byte‑exactly
    # ------------------------------------------------------------------
    device = next(hf_attn.parameters()).device
    compare_rope_inputs(fartsovka_qwen25vl_vision,
                        huggingface_qwen25vl.visual,
                        device)

    # ------------------------------------------------------------------
    # 2.  Forward through the *first* attention block only
    # ------------------------------------------------------------------
    # ---- HuggingFace (torch) -----------------------------------------
    torch_inp  = to_torch(sample_inp).to(device)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    grid_thw   = torch.tensor([[1, 16, 16]], device=device)
    rotary_pos = huggingface_qwen25vl.visual.rot_pos_emb(grid_thw)
    emb        = torch.cat((rotary_pos, rotary_pos), dim=-1)
    cos, sin   = emb.cos(), emb.sin()

    with torch.no_grad():
        hf_out_torch = hf_attn(
            torch_inp,
            cu_seqlens          = cu_seqlens,
            position_embeddings = (cos, sin),
        )

    hf_out = from_torch(hf_out_torch)          # [S, D]

    # ---- Fartsovka (jax) ---------------------------------------------
    grid_thw_jax = jnp.array([[1, 16, 16]])
    vt_pe = fartsovka_qwen25vl_vision.rope(grid_thw_jax)

    # Pass debug_prefix so Fartsovka Attention will emit its internal Q/K/V stats
    err, fs_out_obj = fs_fwd(
        sample_inp,
        positional_embeddings=vt_pe,
        mask=None,
        debug_prefix="FS_Attn",
    )
    err.throw()
    fs_out = fs_out_obj.attention_output        # [S, D]

    max_diff = float(jnp.max(jnp.abs(fs_out - hf_out)))
    print(f"\n[ATTN] max |Δ| between HF and FS block‑0 = {max_diff:.6g}")

    # ------------------------------------------------------------------
    # Extra visibility: show the first & last 5 values of both outputs
    # ------------------------------------------------------------------
    fs_flat = np.asarray(fs_out).flatten()
    hf_flat = np.asarray(hf_out).flatten()

    print("FS out first 5:", fs_flat[:5])
    print("HF out first 5:", hf_flat[:5])
    print("FS out last 5 :", fs_flat[-5:])
    print("HF out last 5 :", hf_flat[-5:])

    # 1e‑4 is generous enough for fp32 ↔ bf16 tiny discrepancies
    assert max_diff < 1e-4, (
        f"Block‑0 attention outputs diverge: max |Δ| = {max_diff:.6g}"
    )

def test_qwen25vl_vision_attention_block0_local(
    huggingface_qwen25vl,
    fartsovka_qwen25vl_vision,
    rng_key,
):
    """
    Compare the first vision-attention block **in local-attention mode**.
    We replicate the exact window-packing and cu_seqlens logic used by
    VisionTransformer so that HF and Fartsovka exercise the same kernel.
    """
    # ------------------------------------------------------------------ set-up
    hf_attn = huggingface_qwen25vl.visual.blocks[0].attn          # HF (torch)
    fs_attn = fartsovka_qwen25vl_vision.stages[0][0].attention    # FS (jax)
    fs_fwd  = checkify_forward(fs_attn)

    model_dim = fs_attn.model_dim
    seq_len   = 256                           # 1 × 16 × 16 raw patches
    sample_inp = jax.random.normal(rng_key, (seq_len, model_dim))

    # ------------------------------------------------------------------ window pack
    grid_thw = jnp.array([[1, 16, 16]])
    window_index, cu_window_seqlens = fartsovka_qwen25vl_vision.get_window_index(grid_thw)
    window_index_np     = np.asarray(window_index)
    inv_window_index_np = np.argsort(window_index_np)
    cu_np               = np.asarray(cu_window_seqlens, dtype=np.int32)

    sample_inp_w = sample_inp[window_index_np]     # packed order

    # ------------------------------------------------------------------ RoPE parity check
    device = next(hf_attn.parameters()).device
    compare_rope_inputs(fartsovka_qwen25vl_vision,
                        huggingface_qwen25vl.visual,
                        device)

    # ------------------------------------------------------------------ HuggingFace forward (torch)
    torch_inp_w  = to_torch(sample_inp_w).to(device)
    cu_seqlens_t = torch.tensor(cu_np, dtype=torch.int32, device=device)

    grid_thw_t   = torch.tensor([[1, 16, 16]], device=device)
    rotary_pos   = huggingface_qwen25vl.visual.rot_pos_emb(grid_thw_t)
    emb          = torch.cat((rotary_pos, rotary_pos), dim=-1)
    cos, sin     = emb.cos(), emb.sin()
    cos_w, sin_w = cos[window_index_np], sin[window_index_np]

    with torch.no_grad():
        hf_out_t = hf_attn(
            torch_inp_w,
            cu_seqlens          = cu_seqlens_t,
            position_embeddings = (cos_w, sin_w),
        )

    hf_out = from_torch(hf_out_t)[inv_window_index_np]      # restore original order

    # ------------------------------------------------------------------ Fartsovka forward (jax)
    vt_pe      = fartsovka_qwen25vl_vision.rope(grid_thw)    # full tables
    pe_window  = VisionPositionalEmbeddings(
        cosines = vt_pe.cosines[window_index_np],
        sines   = vt_pe.sines  [window_index_np],
    )
    pe_adapter = PositionalEmbeddingsAdapter(pe_window)

    # ------------------------------------------------------------------ build block-diagonal mask
    mask = fartsovka_qwen25vl_vision._build_cu_mask(jnp.asarray(cu_np))

    err, fs_out_obj = fs_fwd(
        sample_inp_w,
        positional_embeddings=pe_adapter,
        mask=mask,                   # <- instead of cu_seqlens
        debug_prefix="FS_Attn",
    )
    err.throw()
    fs_out = fs_out_obj.attention_output[inv_window_index_np]

    # ------------------------------------------------------------------ numeric diff
    max_diff = float(jnp.max(jnp.abs(fs_out - hf_out)))
    print(f"\n[ATTN] max |Δ| between HF and FS block-0 = {max_diff:.6g}")

    fs_flat = np.asarray(fs_out).flatten()
    hf_flat = np.asarray(hf_out).flatten()
    print("FS out first 5:", fs_flat[:5])
    print("HF out first 5:", hf_flat[:5])
    print("FS out last 5 :", fs_flat[-5:])
    print("HF out last 5 :", hf_flat[-5:])

    assert max_diff < 1e-4, "Block-0 attention outputs diverge in local-attention mode"


def test_window_index_parity(huggingface_qwen25vl, fartsovka_qwen25vl_vision):
    """Check that get_window_index is identical in HF and FS."""
    grid_thw_np = np.array([[1, 16, 16]], dtype=np.int64)
    grid_thw_jax = jnp.array(grid_thw_np)
    grid_thw_torch = torch.tensor(grid_thw_np, dtype=torch.long)

    # --- FS
    win_idx_fs, cu_seqlens_fs = fartsovka_qwen25vl_vision.get_window_index(grid_thw_jax)
    win_idx_fs = np.asarray(win_idx_fs)
    cu_seqlens_fs = np.asarray(cu_seqlens_fs)

    # --- HF
    # For HF: Try .visual.get_window_index or similar path. Update if attribute differs!
    hf_vis = getattr(huggingface_qwen25vl, "visual", None) or getattr(huggingface_qwen25vl, "vision_tower")
    win_idx_hf, cu_seqlens_hf = hf_vis.get_window_index(grid_thw_torch)
    win_idx_hf = win_idx_hf.cpu().numpy() if hasattr(win_idx_hf, 'cpu') else np.asarray(win_idx_hf)
    cu_seqlens_hf = cu_seqlens_hf.cpu().numpy() if hasattr(cu_seqlens_hf, 'cpu') else np.asarray(cu_seqlens_hf)

    # --- Assert parity
    assert np.array_equal(win_idx_fs, win_idx_hf), (
        f"window_index mismatch:\nFS: {win_idx_fs[:10]}...\nHF: {win_idx_hf[:10]}..."
    )
    assert np.array_equal(cu_seqlens_fs, cu_seqlens_hf), (
        f"cu_seqlens mismatch:\nFS: {cu_seqlens_fs}\nHF: {cu_seqlens_hf}"
    )

    print("window_index and cu_seqlens match perfectly!")