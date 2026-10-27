from functools import cache
from itertools import accumulate

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionBlock, Qwen3_5VisionModel

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.vision import load_qwen3_5_vision_encoder
from lalamo.model_import.model_configs.huggingface.qwen3_5_vision import HFQwen35VisionConfig
from lalamo.module import Keychain
from lalamo.modules.vision import ImageGrid, VisionEncoder
from lalamo.utils.parameter_path import ParameterPath
from tests.helpers import make_test_sharding_config

__all__ = [
    "GRID_CASES",
    "hf_block",
    "hf_cu_seqlens",
    "hf_grid_thw",
    "hf_rotary_embeddings",
    "keychain",
    "lalamo_encoder",
    "random_patches",
    "tiny_hf_vision_model",
    "to_jax",
    "to_torch",
]

# A miniature Qwen3.5 vision tower: 2 blocks, 4 heads of width 16, an 8x8 learned position table.
TINY_VISION_CONFIG = Qwen3_5VisionConfig(
    depth=2,
    hidden_size=64,
    hidden_act="gelu_pytorch_tanh",
    intermediate_size=128,
    num_heads=4,
    in_channels=3,
    patch_size=4,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=96,
    num_position_embeddings=64,
)

GRID_CASES = {
    "single-square": (ImageGrid(1, 4, 4),),
    "single-wide": (ImageGrid(1, 4, 10),),
    "larger-than-table": (ImageGrid(1, 12, 14),),
    "two-images": (ImageGrid(1, 4, 6), ImageGrid(1, 8, 4)),
    "video-clip": (ImageGrid(3, 4, 6),),
}


@cache
def tiny_hf_vision_model() -> Qwen3_5VisionModel:
    torch.manual_seed(0)
    model = Qwen3_5VisionModel._from_config(TINY_VISION_CONFIG, attn_implementation="eager").eval()  # noqa: SLF001
    # The default init leaves LayerNorm at (1, 0), which would hide a swapped scale/bias.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return model


@cache
def lalamo_encoder() -> VisionEncoder:
    hf_model = tiny_hf_vision_model()
    weights_dict = {
        f"model.visual.{name}": jnp.asarray(tensor.detach().numpy()) for name, tensor in hf_model.state_dict().items()
    }
    config = HFQwen35VisionConfig._converter.structure(  # noqa: SLF001
        {**TINY_VISION_CONFIG.to_dict(), "model_type": "qwen3_5"},
        HFQwen35VisionConfig,
    ).to_vision_encoder_config()
    template = config.init(EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config()))
    return load_qwen3_5_vision_encoder(template, weights_dict, ParameterPath("model.visual"))


def hf_block(index: int) -> Qwen3_5VisionBlock:
    block = tiny_hf_vision_model().blocks[index]
    assert isinstance(block, Qwen3_5VisionBlock)
    return block


def keychain() -> Keychain:
    return Keychain.init(0, sharding_config=make_test_sharding_config())


def random_patches(grids: tuple[ImageGrid, ...], seed: int = 0) -> Array:
    patch_dim = lalamo_encoder().config.patch_embedding_config.patch_dim
    num_patches = sum(grid.num_patches for grid in grids)
    return jax.random.normal(jax.random.key(seed), (num_patches, patch_dim), dtype=jnp.float32)


def to_torch(array: Array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(jax.device_get(array)).copy())


def to_jax(tensor: torch.Tensor) -> Array:
    return jnp.asarray(tensor.detach().numpy())


def hf_grid_thw(grids: tuple[ImageGrid, ...]) -> torch.Tensor:
    return torch.tensor([[grid.num_frames, grid.height_in_patches, grid.width_in_patches] for grid in grids])


def hf_cu_seqlens(grids: tuple[ImageGrid, ...]) -> torch.Tensor:
    frame_lengths = [grid.patches_per_frame for grid in grids for _ in range(grid.num_frames)]
    return torch.tensor(list(accumulate(frame_lengths, initial=0)), dtype=torch.int32)


def hf_rotary_embeddings(grids: tuple[ImageGrid, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = tiny_hf_vision_model().rot_pos_emb(hf_grid_thw(grids))
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos(), angles.sin()
