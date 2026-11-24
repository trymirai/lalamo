from itertools import accumulate, pairwise

import jax
import jax.numpy as jnp
import pytest
import torch

from lalamo.initializer import EmptyInitializer
from lalamo.modules.vision import ImageGrid
from tests.common import assert_close, tolerance
from tests.helpers import make_test_sharding_config

from .common import (
    GRID_CASES,
    hf_grid_thw,
    keychain,
    lalamo_encoder,
    random_patches,
    tiny_hf_vision_model,
    to_jax,
    to_torch,
)

GRID_PARAMS = [pytest.param(grids, id=name) for name, grids in GRID_CASES.items()]


@pytest.mark.parametrize("grids", GRID_PARAMS)
def test_encoder_matches_huggingface(grids: tuple[ImageGrid, ...]) -> None:
    patches = random_patches(grids)
    result = lalamo_encoder()(patches, grids, keychain=keychain())
    with torch.no_grad():
        reference = tiny_hf_vision_model()(to_torch(patches), grid_thw=hf_grid_thw(grids)).pooler_output

    with tolerance(atol=1e-4, rtol=1e-4):
        assert_close(result=result.outputs, reference=to_jax(reference), operation_name="Encoder output")


@pytest.mark.parametrize("grids", [pytest.param(GRID_CASES[name], id=name) for name in ("two-images", "video-clip")])
def test_frames_are_encoded_independently(grids: tuple[ImageGrid, ...]) -> None:
    patches = random_patches(grids)
    together = lalamo_encoder()(patches, grids, keychain=keychain()).outputs
    frame_grids = tuple(
        ImageGrid(1, grid.height_in_patches, grid.width_in_patches) for grid in grids for _ in range(grid.num_frames)
    )
    frame_bounds = accumulate((grid.num_patches for grid in frame_grids), initial=0)
    independently = jnp.concatenate(
        [
            lalamo_encoder()(patches[start:end], (grid,), keychain=keychain()).outputs
            for (start, end), grid in zip(pairwise(frame_bounds), frame_grids, strict=True)
        ],
        axis=0,
    )
    with tolerance(atol=1e-5, rtol=1e-5):
        assert_close(result=together, reference=independently, operation_name="Independent frames")


def test_rejects_unmergeable_grid() -> None:
    odd_grid = (ImageGrid(1, 3, 4),)
    with pytest.raises(ValueError, match="divisible"):
        lalamo_encoder()(random_patches(odd_grid), odd_grid, keychain=keychain())


def test_encoder_weights_are_replicated_on_model_mesh() -> None:
    initializer = EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config())
    encoder = lalamo_encoder().config.init(initializer)
    assert encoder.sharding_config.mesh == initializer.sharding_config.mesh
    assert all(not any(weight.sharding.spec) for weight in jax.tree.leaves(encoder))
