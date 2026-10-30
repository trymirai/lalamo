"""Activation-by-activation comparison of a Qwen3.5 vision tower against HuggingFace, on real weights.

The HF reference is a bare ``Qwen3_5VisionModel`` loaded from the same safetensors shard(s) as the
lalamo encoder. The full multimodal model is never instantiated, so even the 27B checkpoint only needs
its vision shard (~1 GB of vision weights) instead of the whole 54 GB.
"""

import gc
from dataclasses import dataclass
from itertools import accumulate

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from PIL import Image, ImageDraw
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from lalamo.model_import.vision import VISION_WEIGHTS_PATH, import_qwen3_5_vision_encoder, resolve_vision_weight_files
from lalamo.module import Keychain
from lalamo.modules.vision import ImageGrid, VisionEncoderResult, VisionForwardPassConfig
from tests.common import assert_close
from tests.tracer.tracer import (
    FRACTION_OF_ALLOWED_VIOLATIONS,
    DType,
    _tracer_sharding_config,
    configure_precision_for_tests,
)

__all__ = ["VisionModelTestSpec", "run_vision_tracer", "synthetic_image"]


@dataclass(frozen=True)
class VisionModelTestSpec:
    model_repo: str
    dtype: DType = DType.FLOAT32
    # (width, height) of each test image; several images exercise the packed multi-image path.
    image_sizes: tuple[tuple[int, int], ...] = ((512, 384), (224, 672))

    @property
    def test_id(self) -> str:
        return f"{self.model_repo}/{self.dtype.value}"


def synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    """A deterministic image with smooth gradients and sharp edges, so every frequency band is exercised."""
    rows, cols = np.mgrid[0:height, 0:width]
    pixels = np.stack(
        [255 * cols / max(width - 1, 1), 255 * rows / max(height - 1, 1), 127 + 127 * np.sin((rows + cols) / 17)],
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        x0, x1 = sorted(rng.integers(0, width, 2).tolist())
        y0, y1 = sorted(rng.integers(0, height, 2).tolist())
        fill = tuple(rng.integers(0, 256, 3).tolist())
        if rng.random() < 0.5:
            draw.rectangle((x0, y0, x1, y1), fill=fill)
        else:
            draw.ellipse((x0, y0, x1, y1), fill=fill)
    return image


def _load_hf_vision_model(repo: str, dtype: DType) -> Qwen3_5VisionModel:
    vision_config = AutoConfig.from_pretrained(repo).vision_config
    model = Qwen3_5VisionModel._from_config(vision_config, attn_implementation="eager", torch_dtype=dtype.torch_dtype)  # noqa: SLF001
    prefix = f"{VISION_WEIGHTS_PATH}."
    state_dict = {}
    for path in resolve_vision_weight_files(repo):
        with safe_open(path, framework="pt") as shard:
            state_dict |= {
                key.removeprefix(prefix): shard.get_tensor(key).to(dtype.torch_dtype)
                for key in shard.keys()  # noqa: SIM118
                if key.startswith(prefix)
            }
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _to_torch(array: Array, dtype: DType) -> torch.Tensor:
    return torch.from_numpy(np.asarray(jax.device_get(array.astype(jnp.float32))).copy()).to(dtype.torch_dtype)


def _to_jax(tensor: torch.Tensor) -> Array:
    return jnp.asarray(tensor.detach().float().numpy())


# The elementwise check shared with the decoder tracer is loose: it lets 3% of values miss by 3%.
# Measured on the real 0.8B and 27B towers in float32, every stage stays below 2e-5 relative RMS error,
# while a merger using tanh instead of erf GELU lands at 2.6e-4 and slips through the elementwise
# check. This bound catches it with a comfortable margin over numerical noise.
MAX_RELATIVE_RMS_ERROR = {DType.FLOAT32: 1e-4, DType.FLOAT16: 1e-2, DType.BFLOAT16: 3e-2}


@dataclass(frozen=True)
class ActivationMatcher:
    dtype: DType

    def __call__(self, result: Array, reference: torch.Tensor, name: str) -> None:
        result = result.astype(jnp.float32)
        reference_array = _to_jax(reference)
        assert_close(
            result=result,
            reference=reference_array,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        error_rms = jnp.sqrt(jnp.mean(jnp.square(result - reference_array)))
        reference_rms = jnp.sqrt(jnp.mean(jnp.square(reference_array)))
        relative_rms_error = float(error_rms / (reference_rms + 1e-12))
        bound = MAX_RELATIVE_RMS_ERROR[self.dtype]
        assert relative_rms_error <= bound, f"{name}: relative RMS error {relative_rms_error:.2e} exceeds {bound:.0e}"


def _hf_rotary(model: Qwen3_5VisionModel, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = model.rot_pos_emb(grid_thw)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos(), angles.sin()


def _match_activations(
    model: Qwen3_5VisionModel,
    result: VisionEncoderResult,
    patches: Array,
    grids: tuple[ImageGrid, ...],
    dtype: DType,
) -> None:
    trace = result.activation_trace
    assert trace is not None
    grid_thw = torch.tensor([[grid.num_frames, grid.height_in_patches, grid.width_in_patches] for grid in grids])
    frame_lengths = [grid.patches_per_frame for grid in grids for _ in range(grid.num_frames)]
    cu_seqlens = torch.tensor(list(accumulate(frame_lengths, initial=0)), dtype=torch.int32)
    hf_rotary = _hf_rotary(model, grid_thw)
    match = ActivationMatcher(dtype)
    cosines, sines = hf_rotary

    match(trace.positional_embeddings.cosines, cosines, "RoPE cosines")
    match(trace.positional_embeddings.sines, sines, "RoPE sines")
    match(trace.patch_embeddings, model.patch_embed(_to_torch(patches, dtype)), "Patch embedding")
    match(trace.position_embeddings, model.fast_pos_embed_interpolate(grid_thw), "Position embedding")

    for index, (block, layer_result) in enumerate(zip(model.blocks, trace.layer_results, strict=True)):
        layer_trace = layer_result.activation_trace
        assert layer_trace is not None
        match(
            layer_trace.pre_attention_norm, block.norm1(_to_torch(layer_trace.inputs, dtype)), f"Layer {index} norm1"
        )
        match(
            layer_trace.attention,
            block.attn(
                _to_torch(layer_trace.pre_attention_norm, dtype),
                cu_seqlens=cu_seqlens,
                position_embeddings=tuple(tensor.to(dtype.torch_dtype) for tensor in hf_rotary),
            ),
            f"Layer {index} attention",
        )
        match(layer_trace.pre_mlp_norm, block.norm2(_to_torch(layer_trace.mlp_inputs, dtype)), f"Layer {index} norm2")
        match(layer_trace.mlp, block.mlp(_to_torch(layer_trace.pre_mlp_norm, dtype)), f"Layer {index} MLP")
        match(
            layer_result.outputs,
            block(_to_torch(layer_trace.inputs, dtype), cu_seqlens=cu_seqlens, position_embeddings=hf_rotary),
            f"Layer {index} full output",
        )

    *_, last_layer = trace.layer_results
    match(result.outputs, model.merger(_to_torch(last_layer.outputs, dtype)), "Patch merger")

    # End to end: run HF on its own and compare every block input plus the final embeddings.
    block_inputs: list[torch.Tensor] = []
    hooks = [
        block.register_forward_pre_hook(lambda _module, args: block_inputs.append(args[0])) for block in model.blocks
    ]
    try:
        hf_outputs = model(_to_torch(patches, dtype), grid_thw=grid_thw).pooler_output
    finally:
        for hook in hooks:
            hook.remove()
    for index, (hf_input, layer_result) in enumerate(zip(block_inputs, trace.layer_results, strict=True)):
        assert layer_result.activation_trace is not None
        match(layer_result.activation_trace.inputs, hf_input, f"End2End layer {index} inputs")
    match(result.outputs, hf_outputs, "End2End vision embeddings")


def run_vision_tracer(spec: VisionModelTestSpec) -> None:
    configure_precision_for_tests()
    sharding_config = _tracer_sharding_config()
    imported = import_qwen3_5_vision_encoder(
        spec.model_repo,
        sharding_config=sharding_config,
        dtype=spec.dtype.jax_dtype,
    )
    images = [synthetic_image(width, height, seed) for seed, (width, height) in enumerate(spec.image_sizes)]
    preprocessed = imported.preprocessing_config.preprocess(images)
    patches = jax.device_put(jnp.asarray(preprocessed.patches), sharding_config.make_sharding((None, None)))

    result = imported.encoder(
        patches,
        preprocessed.grids,
        VisionForwardPassConfig.for_tracer_tests(),
        return_activation_trace=True,
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )
    model = _load_hf_vision_model(spec.model_repo, spec.dtype)
    try:
        with torch.no_grad():
            _match_activations(model, result, patches, preprocessed.grids, spec.dtype)
    finally:
        del model, result, imported
        gc.collect()
        jax.clear_caches()
