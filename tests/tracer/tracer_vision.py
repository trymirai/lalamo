"""Activation-by-activation comparison of a Qwen3.5 vision tower against a bare HF ``Qwen3_5VisionModel``."""

import gc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from PIL import Image, ImageDraw
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from lalamo.model_import.loaders.vision import load_hf_vision
from lalamo.model_import.origins import HuggingFaceOrigin, hf_resolve_weights
from lalamo.module import Keychain
from lalamo.modules.vision import ImageGrid, VisionEncoderResult, VisionForwardPassConfig
from tests.common import assert_close
from tests.tracer.tracer import (
    FRACTION_OF_ALLOWED_VIOLATIONS,
    _tracer_sharding_config,
    configure_precision_for_tests,
)
from tests.unit.modules.vision.common import (
    hf_cu_seqlens,
    hf_grid_thw,
    hf_position_embeddings,
    hf_rotary_embeddings,
    to_jax,
    to_torch,
)

__all__ = ["VisionModelTestSpec", "run_vision_tracer"]


@dataclass(frozen=True)
class VisionModelTestSpec:
    model_repo: str
    # (width, height) of each test image; several images exercise the packed multi-image path.
    image_sizes: tuple[tuple[int, int], ...] = ((512, 384), (224, 672))


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


def _load_hf_vision_model(repo: str) -> Qwen3_5VisionModel:
    vision_config = AutoConfig.from_pretrained(repo).vision_config
    model = Qwen3_5VisionModel._from_config(vision_config, attn_implementation="eager", dtype=torch.float32)  # noqa: SLF001
    prefix = "model.visual."
    state_dict = {}
    for path in hf_resolve_weights(repo, ".safetensors"):
        with safe_open(path, framework="pt") as shard:
            state_dict |= {
                key.removeprefix(prefix): shard.get_tensor(key)
                for key in shard.keys()  # noqa: SIM118
                if key.startswith(prefix)
            }
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _match(result: Array, reference: torch.Tensor, name: str) -> None:
    result = result.astype(jnp.float32)
    reference_array = to_jax(reference)
    assert_close(
        result=result,
        reference=reference_array,
        operation_name=name,
        fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
    )
    # An RMS bound catches systematic errors that the tracer's elementwise allowance can miss.
    error_rms = jnp.sqrt(jnp.mean(jnp.square(result - reference_array)))
    reference_rms = jnp.sqrt(jnp.mean(jnp.square(reference_array)))
    relative_rms_error = float(error_rms / (reference_rms + 1e-12))
    assert relative_rms_error <= 1e-4, f"{name}: relative RMS error {relative_rms_error:.2e} exceeds 1e-4"


def _match_activations(
    model: Qwen3_5VisionModel,
    result: VisionEncoderResult,
    patches: Array,
    grids: tuple[ImageGrid, ...],
) -> None:
    trace = result.activation_trace
    assert trace is not None
    grid_thw = hf_grid_thw(grids)
    cu_seqlens = hf_cu_seqlens(grids)
    hf_rotary = hf_rotary_embeddings(model, grids)
    cosines, sines = hf_rotary

    _match(trace.positional_embeddings.cosines, cosines, "RoPE cosines")
    _match(trace.positional_embeddings.sines, sines, "RoPE sines")
    _match(trace.patch_embeddings, model.patch_embed(to_torch(patches)), "Patch embedding")
    _match(trace.position_embeddings, hf_position_embeddings(model, grids), "Position embedding")

    for index, (block, layer_result) in enumerate(zip(model.blocks, trace.layer_results, strict=True)):
        layer_trace = layer_result.activation_trace
        assert layer_trace is not None
        _match(layer_trace.pre_attention_norm, block.norm1(to_torch(layer_trace.inputs)), f"Layer {index} norm1")
        _match(
            layer_trace.attention,
            block.attn(
                to_torch(layer_trace.pre_attention_norm),
                cu_seqlens=cu_seqlens,
                position_embeddings=hf_rotary,
            ),
            f"Layer {index} attention",
        )
        _match(layer_trace.pre_mlp_norm, block.norm2(to_torch(layer_trace.mlp_inputs)), f"Layer {index} norm2")
        _match(layer_trace.mlp, block.mlp(to_torch(layer_trace.pre_mlp_norm)), f"Layer {index} MLP")
        _match(
            layer_result.outputs,
            block(to_torch(layer_trace.inputs), cu_seqlens=cu_seqlens, position_embeddings=hf_rotary),
            f"Layer {index} full output",
        )

    *_, last_layer = trace.layer_results
    _match(result.outputs, model.merger(to_torch(last_layer.outputs)), "Patch merger")

    # End to end: run HF on its own and compare every block input plus the final embeddings.
    block_inputs: list[torch.Tensor] = []
    hooks = [
        block.register_forward_pre_hook(lambda _module, args: block_inputs.append(args[0])) for block in model.blocks
    ]
    try:
        hf_outputs = model(to_torch(patches), grid_thw=grid_thw).pooler_output
    finally:
        for hook in hooks:
            hook.remove()
    for index, (hf_input, layer_result) in enumerate(zip(block_inputs, trace.layer_results, strict=True)):
        assert layer_result.activation_trace is not None
        _match(layer_result.activation_trace.inputs, hf_input, f"End2End layer {index} inputs")
    _match(result.outputs, hf_outputs, "End2End vision embeddings")


def run_vision_tracer(spec: VisionModelTestSpec) -> None:
    configure_precision_for_tests()
    sharding_config = _tracer_sharding_config()
    imported = load_hf_vision(
        HuggingFaceOrigin(spec.model_repo),
        sharding_config=sharding_config,
        dtype=jnp.float32,
    )
    images = [synthetic_image(width, height, seed) for seed, (width, height) in enumerate(spec.image_sizes)]
    preprocessed = imported.config.preprocess(images)
    patches = jax.device_put(jnp.asarray(preprocessed.patches), sharding_config.make_sharding((None, None)))

    result = imported.encoder(
        patches,
        preprocessed.grids,
        VisionForwardPassConfig.for_tracer_tests(),
        return_activation_trace=True,
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )
    model = _load_hf_vision_model(spec.model_repo)
    try:
        with torch.no_grad():
            _match_activations(model, result, patches, preprocessed.grids)
    finally:
        del model, result, imported
        gc.collect()
        jax.clear_caches()
