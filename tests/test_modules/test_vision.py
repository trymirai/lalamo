from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float
from PIL import Image

from .common import assert_close, to_torch


def generate_gradient_image(h: int = 224, w: int = 224) -> Float[Array, "3 h w"]:
    """Generate a random gradient image for testing."""
    rng = np.random.default_rng(seed=42)  # stable seed → reproducible noise
    img = rng.random((3, h, w), dtype=np.float32)  # CHW in [0,1]
    return jnp.array(img)


def load_test_image(path: Path | None = None) -> Float[Array, "3 224 224"]:
    """Load a test image or generate one if path is None."""
    if path is None:
        return generate_gradient_image()
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return jnp.array(arr)


def prepare_vision_inputs(vision_model) -> tuple[jnp.ndarray, jnp.ndarray, torch.Tensor, torch.Tensor]:
    """Prepare inputs for vision model testing in both JAX and PyTorch formats."""
    img = generate_gradient_image()  # [C, H, W]

    # Get temporal patch size from model config
    t_sz = vision_model.config.patch_embedding_config.temporal_patch_size
    frames = [img] * t_sz

    # Create JAX input tensor
    img_jax = jnp.expand_dims(jnp.stack(frames, axis=1), 0)  # [B, C, T, H, W]

    # Create PyTorch input tensor
    img_torch = to_torch(img_jax)

    # Create grid_thw tensors
    B, C, T_img, H_img, W_img = img_jax.shape
    tp = t_sz
    p_sz = vision_model.config.patch_embedding_config.patch_size

    grid_thw_jax = jnp.array([[T_img // tp, H_img // p_sz, W_img // p_sz]], dtype=jnp.int32)
    grid_thw_torch = to_torch(grid_thw_jax).to(torch.long)

    return img_jax, grid_thw_jax, img_torch, grid_thw_torch


@pytest.mark.parametrize("dtype", [jnp.float32])
def test_vision_encoder_parity(
    huggingface_qwen25vl,
    fartsovka_qwen25vl_vision,
    dtype,
):
    """Test that Fartsovka vision encoder matches HuggingFace implementation."""
    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model missing")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka VisionTransformer fixture failed to load")

    # Get HuggingFace vision model
    hf_vis = getattr(huggingface_qwen25vl, "vision_tower",
                     getattr(huggingface_qwen25vl, "visual", None))
    if hf_vis is None:
        pytest.skip("HF model missing vision tower")

    # Prepare inputs for both models
    img_jax, grid_thw_jax, img_torch, grid_thw_torch = prepare_vision_inputs(fartsovka_qwen25vl_vision)

    # Convert to specified dtype if needed
    if dtype != jnp.float32:
        img_jax = img_jax.astype(dtype)

    # Ensure HF model is on correct device
    device = img_torch.device
    if hasattr(hf_vis, "to"):
        hf_vis = hf_vis.to(device)

    # Run HuggingFace model
    with torch.no_grad():
        hf_out = hf_vis(img_torch, grid_thw=grid_thw_torch)
        hf_out_np = hf_out.cpu().numpy().astype("float32")

    # Run Fartsovka model
    fs_result = fartsovka_qwen25vl_vision(img_jax, grid_thw=grid_thw_jax)

    # Extract output from Fartsovka result
    if hasattr(fs_result, "output") and fs_result.output is not None:
        fs_out_np = np.asarray(fs_result.output).astype("float32")
    elif isinstance(fs_result, jnp.ndarray):
        fs_out_np = np.asarray(fs_result).astype("float32")
    elif isinstance(fs_result, tuple) and len(fs_result) > 0 and isinstance(fs_result[0], jnp.ndarray):
        fs_out_np = np.asarray(fs_result[0]).astype("float32")
    else:
        raise ValueError(f"Unexpected output type from Fartsovka: {type(fs_result)}")

    # Assert outputs are close
    assert hf_out_np.shape == fs_out_np.shape, "Vision encoder output shapes don't match"

    # Print max delta
    max_delta = float(np.max(np.abs(fs_out_np - hf_out_np)))
    print(f"Vision encoder max |Δ|: {max_delta:.6g}")

    assert_close(
        result=jnp.array(fs_out_np),
        reference=jnp.array(hf_out_np),
        rtol=0.01,
        atol=3e-3,
    )


def test_window_index_parity(huggingface_qwen25vl, fartsovka_qwen25vl_vision):
    """Check that get_window_index is identical in HF and FS."""
    if huggingface_qwen25vl is None:
        pytest.skip("HF reference model missing")
    if fartsovka_qwen25vl_vision is None:
        pytest.skip("Fartsovka VisionTransformer fixture failed to load")

    grid_thw_np = np.array([[1, 16, 16]], dtype=np.int64)
    grid_thw_jax = jnp.array(grid_thw_np)
    grid_thw_torch = torch.tensor(grid_thw_np, dtype=torch.long)

    # Get window index from Fartsovka model
    win_idx_fs, cu_seqlens_fs = fartsovka_qwen25vl_vision.get_window_index(grid_thw_jax)
    win_idx_fs = np.asarray(win_idx_fs)
    cu_seqlens_fs = np.asarray(cu_seqlens_fs)

    # Get window index from HuggingFace model
    hf_vis = getattr(huggingface_qwen25vl, "visual", None) or huggingface_qwen25vl.vision_tower
    win_idx_hf, cu_seqlens_hf = hf_vis.get_window_index(grid_thw_torch)
    win_idx_hf = win_idx_hf.cpu().numpy() if hasattr(win_idx_hf, "cpu") else np.asarray(win_idx_hf)
    cu_seqlens_hf = cu_seqlens_hf.cpu().numpy() if hasattr(cu_seqlens_hf, "cpu") else np.asarray(cu_seqlens_hf)

    # Assert indices match
    assert np.array_equal(win_idx_fs, win_idx_hf), "window_index mismatch"
    assert np.array_equal(cu_seqlens_fs, cu_seqlens_hf), "cu_seqlens mismatch"
