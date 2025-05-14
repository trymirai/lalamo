from pathlib import Path
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float, PRNGKeyArray
from PIL import Image

# NOTE: **NO hard dependency on Fartsovka VisionTransformer**
# --------------------------------------------------------
# We only need patch‑size / temporal‑patch‑size to build the reference
# layout tensor.  If the Fartsovka fixture fails to load (e.g. weight‑key
# mismatch) we fall back to the values provided by the HF config so the
# *layout/value* sanity test can still run and tell us precisely where the
# divergence appears.

from .common import assert_close, from_torch, to_torch

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

_ATOL = 1e-7  # ultra‑strict – we expect byte‑identical layout
_RTOL = 1e-7
_MAX_PRINT = 5  # how many individual element diffs to show

# ----------------------------------------------------------------------------
# Utilities
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
    C = img_jax.shape[1]
    patches_jax_f32 = img_jax.reshape(-1, C, t_sz, p_sz, p_sz)
    patches_torch_f32 = from_torch(img_torch.view(-1, C, t_sz, p_sz, p_sz))

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
    patches_torch_bf16 = from_torch(img_torch.view(-1, C, t_sz, p_sz, p_sz).to(dtype=dtype_ref)).astype(jnp.bfloat16)

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
# NEW TEST 3 – VisionRoPE positional‑embedding parity
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

    hf_vis = getattr(huggingface_qwen25vl, "vision_tower", getattr(huggingface_qwen25vl, "visual"))
    hf_rotary = _find_hf_rotary_emb(hf_vis)
    print(hf_rotary)
    if hf_rotary is None:
        pytest.skip("Could not locate rotary_emb module inside HF model – API changed?")

    # ----------------- build dummy (h,w) id grid 0..seq_len‑1 ----------------
    pos_ids_jax = jnp.stack(
        [jnp.arange(seq_len, dtype=jnp.int32),  # h
         jnp.zeros(seq_len, dtype=jnp.int32)],  # w = 0
        axis=-1,
    )  # [S,2]

    # ------------------- Fartsovka ► cos/sin -------------------------------
    vt_pe = fartsovka_qwen25vl_vision.rope(pos_ids_jax)
    cos_fs = vt_pe.cosines  # [S, head_dim]
    sin_fs = vt_pe.sines

    # --------------------- HF ► cos/sin ------------------------------------
    with torch.no_grad():
        # HF implementation expects an *integer sequence length*, not explicit indices
        angles = hf_rotary(seq_len)  # θ values, shape [seq_len, head_dim//2]
        emb = torch.cat([angles, angles], dim=-1)  # duplicate like HF forward pass does later on
        cos_hf = emb.cos()
        sin_hf = emb.sin()
        cos_hf = from_torch(cos_hf)
        sin_hf = from_torch(sin_hf)

    # -------- DEBUG snapshot & stats ---------------------------------
    import numpy as _np  # local import to avoid polluting global imports
    _rows, _cols = 4, 4  # how many rows / cols to print
    print("\n[DEBUG] cos_fs sample:\n", _np.asarray(cos_fs[:_rows, :_cols]))
    print("[DEBUG] cos_hf sample:\n", _np.asarray(cos_hf[:_rows, :_cols]))
    print("[DEBUG] max |Δ cos| =", float(jnp.max(jnp.abs(cos_fs - cos_hf))))
    print("[DEBUG] max |Δ sin| =", float(jnp.max(jnp.abs(sin_fs - sin_hf))))

    # --------------------- Numerical equality ------------------------------
    assert_close(result=cos_fs, reference=cos_hf, atol=1e-6, rtol=1e-6, operation_name="VisionRoPE cos")
    assert_close(result=sin_fs, reference=sin_hf, atol=1e-6, rtol=1e-6, operation_name="VisionRoPE sin")

    print("\n✓ VisionRoPE cosine/sine tables match HF implementation (seq_len =", seq_len, ")")
