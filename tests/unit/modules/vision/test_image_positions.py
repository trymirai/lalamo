from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from lalamo.modules.vision import ImageGrid, MRoPEImagePositions

IMAGE = 9
TEXT = 1
MERGE = 2


def _prompt(*pieces: int | ImageGrid) -> tuple[np.ndarray, tuple[ImageGrid, ...]]:
    """Build token ids from text run lengths (ints) and images (grids)."""
    token_ids: list[int] = []
    for piece in pieces:
        if isinstance(piece, ImageGrid):
            token_ids += [IMAGE] * piece.num_merged_tokens(MERGE)
        else:
            token_ids += [TEXT] * piece
    return np.asarray(token_ids), tuple(piece for piece in pieces if isinstance(piece, ImageGrid))


def _hf_positions(token_ids: np.ndarray, grids: tuple[ImageGrid, ...]) -> np.ndarray:
    # get_rope_index only needs the merge size and get_vision_position_ids from `self`, so it is called
    # unbound on a stand-in object; the cast drops HF's annotations, which expect a full Qwen3_5Model.
    model_class = cast("Any", Qwen3_5Model)
    stand_in = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=MERGE)))
    stand_in.get_vision_position_ids = lambda *args, **kwargs: model_class.get_vision_position_ids(
        stand_in,
        *args,
        **kwargs,
    )
    input_ids = torch.from_numpy(token_ids)[None]
    positions, _ = model_class.get_rope_index(
        stand_in,
        input_ids,
        mm_token_type_ids=(input_ids == IMAGE).int(),
        image_grid_thw=torch.tensor([[g.num_frames, g.height_in_patches, g.width_in_patches] for g in grids]),
    )
    return positions[:, 0].T.numpy()


PROMPTS = {
    "text-image-text": (5, ImageGrid(1, 4, 6), 7),
    "image-first": (ImageGrid(1, 8, 4), 3),
    "two-images": (2, ImageGrid(1, 4, 4), 1, ImageGrid(1, 6, 10), 4),
}


@pytest.mark.parametrize("pieces", [pytest.param(pieces, id=name) for name, pieces in PROMPTS.items()])
def test_matches_huggingface(pieces: tuple[int | ImageGrid, ...]) -> None:
    token_ids, grids = _prompt(*pieces)
    result = MRoPEImagePositions().positions(token_ids, IMAGE, grids, MERGE)
    np.testing.assert_array_equal(result, _hf_positions(token_ids, grids))


def test_adjacent_images_are_laid_out_one_after_another() -> None:
    # HF cannot handle this (it groups consecutive image tokens into one image); real prompts always
    # separate images with <|vision_end|><|vision_start|>. Positions should still be well defined.
    first, second = ImageGrid(1, 4, 6), ImageGrid(1, 2, 2)
    token_ids, grids = _prompt(3, first, second, 2)
    result = MRoPEImagePositions().positions(token_ids, IMAGE, grids, MERGE)
    second_start = 3 + first.num_merged_tokens(MERGE)
    # The first image (2 x 3 merged tokens) starts at 3 and spans max(2, 3) = 3 positions.
    np.testing.assert_array_equal(result[second_start], [6, 6, 6])
    np.testing.assert_array_equal(result[-2:, 0], [7, 8])


def test_text_only_is_sequential() -> None:
    token_ids, grids = _prompt(6)
    result = MRoPEImagePositions().positions(token_ids, IMAGE, grids, MERGE)
    np.testing.assert_array_equal(result, np.repeat(np.arange(6)[:, None], 3, axis=1))


def test_rejects_token_count_mismatch() -> None:
    token_ids, grids = _prompt(2, ImageGrid(1, 4, 4), 2)
    with pytest.raises(ValueError, match="image tokens"):
        MRoPEImagePositions().positions(token_ids[1:-3], IMAGE, grids, MERGE)
