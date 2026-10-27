from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Int

from lalamo.module import LalamoConfig
from lalamo.modules.mrope import PositionAxis
from lalamo.utils.registry_abc import RegistryABC

from .image_grid import ImageGrid

__all__ = [
    "ImagePositionScheme",
    "MRoPEImagePositions",
]


@dataclass(frozen=True)
class ImagePositionScheme(LalamoConfig, RegistryABC):
    """How a model assigns RoPE positions to a prompt that contains image tokens.

    This is the one piece of multimodal prompting that differs most between model families, so it is
    its own pluggable config. Qwen-VL models use 3D M-RoPE positions (``MRoPEImagePositions``); LLaVA-
    and Gemma-style models would simply count image tokens like text.

    Positions are computed on the host from concrete token ids: prompts are known before prefill, and
    the image grids decide the result, so there is nothing to trace.
    """

    @property
    @abstractmethod
    def num_position_axes(self) -> int: ...

    @abstractmethod
    def positions(
        self,
        token_ids: Int[np.ndarray, " tokens"],
        image_token_id: int,
        grids: tuple[ImageGrid, ...],
        spatial_merge_size: int,
    ) -> Int[np.ndarray, " tokens *position_axes"]: ...


@dataclass(frozen=True)
class MRoPEImagePositions(ImagePositionScheme):
    """Qwen-VL M-RoPE positions (HF ``get_rope_index``), restricted to still images.

    Text tokens advance one position at a time on all three axes. An image placed at position ``p``
    keeps ``time = p`` for all its tokens and lays out ``height = p + row`` and ``width = p + col``
    over the merged token grid. The next token then resumes at ``p + max(rows, cols)``, so a large
    image costs as many positions as its longer side, not as many as its token count.
    """

    @property
    def num_position_axes(self) -> int:
        return len(PositionAxis)

    def positions(
        self,
        token_ids: Int[np.ndarray, " tokens"],
        image_token_id: int,
        grids: tuple[ImageGrid, ...],
        spatial_merge_size: int,
    ) -> Int[np.ndarray, "tokens 3"]:
        (num_tokens,) = token_ids.shape
        is_image_token = token_ids == image_token_id
        expected_image_tokens = sum(grid.num_merged_tokens(spatial_merge_size) for grid in grids)
        if int(is_image_token.sum()) != expected_image_tokens:
            raise ValueError(
                f"Prompt has {int(is_image_token.sum())} image tokens but the images need {expected_image_tokens}.",
            )

        segments: list[Int[np.ndarray, "segment_tokens 3"]] = []
        next_position = 0
        token_index = 0
        remaining_grids = iter(grids)
        while token_index < num_tokens:
            if not is_image_token[token_index]:
                later_image_tokens = np.flatnonzero(is_image_token[token_index:])
                text_end = token_index + int(later_image_tokens[0]) if later_image_tokens.size else num_tokens
                text_positions = np.arange(next_position, next_position + text_end - token_index)
                segments.append(np.repeat(text_positions[:, None], len(PositionAxis), axis=1))
                next_position += text_end - token_index
                token_index = text_end
                continue

            grid = next(remaining_grids)
            if grid.num_frames != 1:
                raise ValueError("Only still images are supported; video positions use per-frame timestamps.")
            rows = grid.height_in_patches // spatial_merge_size
            cols = grid.width_in_patches // spatial_merge_size
            if not is_image_token[token_index : token_index + rows * cols].all():
                raise ValueError(f"Image at token {token_index} needs {rows * cols} consecutive image tokens.")
            row_index, col_index = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            segments.append(
                np.stack(
                    [
                        np.full(rows * cols, next_position),
                        next_position + row_index.reshape(-1),
                        next_position + col_index.reshape(-1),
                    ],
                    axis=-1,
                ),
            )
            next_position += max(rows, cols)
            token_index += rows * cols

        return np.concatenate(segments).astype(np.int32)
