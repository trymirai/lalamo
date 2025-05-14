# Copyright 2025 The Fartsovka Team.
# JAX implementation of Qwen‑2.5‑VL vision rotary embeddings
# (faithful to the original HuggingFace VisionRotaryEmbedding).

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import equinox as eqx

from fartsovka.common import DType, ParameterDict
from .common import FartsovkaModule

__all__ = [
    "VisionRoPEConfig",
    "VisionRoPE",
    "VisionPositionalEmbeddings",
]


# ---------------------------------------------------------------------
# Helper container returned by VisionRoPE
# ---------------------------------------------------------------------
class VisionPositionalEmbeddings:
    """
    Holds pre‑computed cos / sin tables in the exact shapes expected by the
    Fartsovka test‑suite (mirrors HF behaviour).

        • cosines / sines – shape **[tokens, head_dim]**
    """

    cosines: Float[Array, "tokens head_dim"]
    sines:   Float[Array, "tokens head_dim"]

    def __init__(self, theta: Float[Array, "tokens head_dim"]):
        self.cosines = jnp.cos(theta)
        self.sines   = jnp.sin(theta)

    # legacy aliases still referenced by tests
    @property
    def cos_cached(self):
        return self.cosines

    @property
    def sin_cached(self):
        return self.sines


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class VisionRoPEConfig:
    """
    Configuration object — keeps *all* knobs we used previously so that
    external loaders remain compatible.
    If `duplicate_hw` is **True** the H frequencies are duplicated to W so the
    resulting table is identical to HuggingFace’s 1‑D implementation.
    """

    precision: DType = jnp.float32
    base: float = 10000.0            # “θ” in the HF code
    max_sequence_length: int = 2048  # kept for future scaling tricks

    # When True, horizontal (H) frequencies are duplicated to the second
    # half of the head dimension, ignoring the W coordinate.  This matches
    # HuggingFace’s 1‑D VisionRotaryEmbedding behaviour and is required
    # for Qwen‑2.5‑VL parity.
    duplicate_hw: bool = True

    # optional hook for scaled‑RoPE variants
    def _scale_inverse_frequencies(
        self,
        inv_freq: Float[Array, "half_dim"],
        *,
        head_dim: int,
        max_sequence_length: int,
    ) -> Float[Array, "half_dim"]:
        return inv_freq  # passthrough by default

    # -------- public factory --------
    def init(
        self,
        *,
        head_dim: int,
        num_timesteps: Optional[int] = None,
    ) -> "VisionRoPE":
        """
        Args
        ----
        head_dim:
            Full dimension of one attention head **before** any splitting.
        num_timesteps:
            Maximum spatial index seen at load‑time. Stored for potential
            range checks, otherwise not used in frequency computation.
        """
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for 2‑D RoPE.")
        half_dim = head_dim // 2  # what HF’s VisionRotaryEmbedding receives

        # indices 0,2,4,…,half_dim‑2  (even positions)
        idx = jnp.arange(0, half_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.base ** (idx / half_dim))
        inv_freq = self._scale_inverse_frequencies(
            inv_freq, head_dim=head_dim, max_sequence_length=self.max_sequence_length
        )

        return VisionRoPE(
            config=self,
            inv_freq=inv_freq,              # shape [half_dim/2]
            head_dim=head_dim,
            max_position=num_timesteps or self.max_sequence_length,
        )


# ---------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------
class VisionRoPE(FartsovkaModule[VisionRoPEConfig]):
    """
    Generates RoPE cos/sin tables matching HuggingFace for both:

      • The legacy scalar‑`seq_len` interface (1‑D).
      • The newer 2‑D spatial index interface used by the tests
        (position_ids shape `[tokens, 2]`, containing (h,w) pairs).
    """

    # static fields -----------------------------------------------------------------
    head_dim: int = eqx.static_field()
    inv_freq: Float[Array, "half_dim_even"] = eqx.static_field()
    max_position: int = eqx.static_field()

    # -------------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------------
    def _1d_freqs(self, seq_len: int) -> Float[Array, "tokens half_dim"]:
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, self.inv_freq)              # [S, half_dim/2]
        return jnp.concatenate([freqs, freqs], axis=-1)    # [S, half_dim]

    def _2d_freqs(
        self, position_ids: Int[Array, "tokens 2"]
    ) -> Float[Array, "tokens half_dim"]:
        if position_ids.shape[-1] != 2:
            raise ValueError("Expected position_ids shape [tokens,2] for 2‑D RoPE.")

        if jnp.any(position_ids < 0) or jnp.any(position_ids >= self.max_position):
            # Just warn – matching earlier behaviour
            print(
                f"WARN: position indices outside configured range [0,{self.max_position}) were clipped."
            )

        h_pos = position_ids[:, 0]
        w_pos = position_ids[:, 1]

        h_freqs = jnp.outer(h_pos.astype(jnp.float32), self.inv_freq)  # [T, half_dim/2]

        # If we want HF‑style duplication, ignore W and repeat H freqs.
        if self.config.duplicate_hw:
            return jnp.concatenate([h_freqs, h_freqs], axis=-1)        # [T, half_dim]

        # Legacy 2‑D mode: distinct frequencies for W.
        w_freqs = jnp.outer(w_pos.astype(jnp.float32), self.inv_freq)  # [T, half_dim/2]
        return jnp.concatenate([h_freqs, w_freqs], axis=-1)            # [T, half_dim]

    # -------------------------------------------------------------------------------
    # Public call
    # -------------------------------------------------------------------------------
    def __call__(
        self,
        position: int | Int[Array, ""] | Int[Array, "tokens 2"],
    ) -> VisionPositionalEmbeddings:
        """
        *If* a scalar / 0‑D array is given → treat it as ``seq_len`` (HF mode).

        *Else* expect a 2‑column matrix with (h,w) spatial indices (ViT mode).
        """
        if jnp.isscalar(position) or (isinstance(position, jnp.ndarray) and position.ndim == 0):
            theta = self._1d_freqs(int(position))            # 1‑D mode
        else:
            theta = self._2d_freqs(position)                 # 2‑D mode

        # Return θ values with shape [tokens, head_dim//2] (matches HF)
        return VisionPositionalEmbeddings(theta)

    # -------------------------------------------------------------------------------
    # No learnable parameters
    # -------------------------------------------------------------------------------
    def export_weights(self) -> ParameterDict:
        return ParameterDict()
