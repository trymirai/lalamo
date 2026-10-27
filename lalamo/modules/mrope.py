from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int

from .rope import PositionalEmbeddings, RoPEConfig

__all__ = [
    "MRoPEConfig",
    "MRoPELayout",
    "PositionAxis",
]


class PositionAxis(StrEnum):
    TIME = "time"
    HEIGHT = "height"
    WIDTH = "width"


class MRoPELayout(StrEnum):
    # Qwen2-VL / Qwen2.5-VL: contiguous blocks [T T T ... H H H ... W W W].
    CHUNKED = "chunked"
    # Qwen3-VL / Qwen3.5: round-robin [T H W T H W ... T T], so every axis gets low and high frequencies.
    INTERLEAVED = "interleaved"


@dataclass(frozen=True, kw_only=True)
class MRoPEConfig(RoPEConfig):
    """Multimodal rotary embedding: each rotary frequency pair follows one of three position axes.

    Tokens carry a (time, height, width) position. Text tokens have all three equal, and in that case
    the result is identical to plain RoPE. Image tokens share one time position and spread over height
    and width, so attention sees the 2D layout of the picture.

    ``sections`` gives how many of the ``head_dim / 2`` frequency pairs belong to each axis, and
    ``layout`` says how those pairs are arranged. Positions may be given per axis, with shape
    ``(tokens, 3)``, or as a single scalar per token, which is broadcast to all axes.
    """

    sections: tuple[int, ...]
    layout: MRoPELayout

    def __post_init__(self) -> None:
        if len(self.sections) != len(PositionAxis):
            raise ValueError(f"M-RoPE needs one section per position axis, got {self.sections}.")
        if sum(self.sections) != self.head_dim // 2:
            raise ValueError(f"M-RoPE sections {self.sections} must add up to head_dim / 2 = {self.head_dim // 2}.")

    @property
    def num_position_axes(self) -> int:
        return len(PositionAxis)

    def pair_axes(self) -> Int[np.ndarray, " rotary_pairs"]:
        """Index of the position axis (0 = time, 1 = height, 2 = width) driving each frequency pair."""
        match self.layout:
            case MRoPELayout.CHUNKED:
                return np.repeat(np.arange(len(PositionAxis)), self.sections)
            case MRoPELayout.INTERLEAVED:
                # Mirrors HF's apply_interleaved_mrope: start from all-time, then claim every third
                # pair for height (offset 1) and width (offset 2) within that axis' share.
                _, height_pairs, width_pairs = self.sections
                pair_indices = np.arange(self.head_dim // 2)
                is_height = (pair_indices % 3 == 1) & (pair_indices < 3 * height_pairs)
                is_width = (pair_indices % 3 == 2) & (pair_indices < 3 * width_pairs)
                return np.where(is_height, 1, np.where(is_width, 2, 0))

    def compute_positional_embeddings(self, timesteps: Int[Array, " tokens *position_axes"]) -> PositionalEmbeddings:
        match timesteps.ndim:
            case 1:
                axis_positions = jnp.repeat(timesteps[:, None], len(PositionAxis), axis=1)
            case 2:
                axis_positions = timesteps
            case _:
                raise ValueError(f"Expected positions of shape (tokens,) or (tokens, 3), got {timesteps.shape}.")
        pair_positions = axis_positions[:, self.pair_axes()].astype(jnp.float32)
        return self.embeddings_from_angles(pair_positions * self.inverse_frequencies())
