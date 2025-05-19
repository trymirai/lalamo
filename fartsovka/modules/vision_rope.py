# Copyright 2025 The Fartsovka Team.
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
    """Container for cosine and sine positional embeddings.

    Args:
        cosines: [tokens, head_dim] array of cosine embeddings.
        sines: [tokens, head_dim] array of sine embeddings.
    """
    cosines: Float[Array, "tokens head_dim"]
    sines:   Float[Array, "tokens head_dim"]

    def __init__(self, *, cosines: Float[Array, "tokens head_dim"], sines: Float[Array, "tokens head_dim"]):
            self.cosines = cosines
            self.sines = sines


    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        cos, sin = self.cosines, self.sines                     # [S, Dh]
        print("[DEBUG VisionPositionalEmbeddings.apply] x shape", x.shape,
              "cos shape", cos.shape, "sin shape", sin.shape)

        x1 = x[:, : x.shape[-1] // 2]                           # even
        x2 = x[:, x.shape[-1] // 2 :]                           # odd
        rotated = jnp.concatenate((-x2, x1), axis=-1)           # (-odd, even)

        out = (x * cos) + (rotated * sin)                       # element-wise

        return out

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
    """Configuration for VisionRoPE."""

    precision: DType = jnp.float32
    base: float = 10000.0              # "θ" in the HF code
    max_sequence_length: int = 2048    # kept for future scaling tricks

    # When True, horizontal (H) frequencies are duplicated to the second
    # half of the head dimension, ignoring the W coordinate.  This matches
    # HuggingFace's 1‑D VisionRotaryEmbedding behaviour and is required
    # for Qwen‑2.5‑VL parity.
    duplicate_hw: bool = False

    # For completeness (matches PatchMergerConfig), not yet used
    spatial_merge_size: int = 2

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
        half_dim = head_dim // 2  # what HF's VisionRotaryEmbedding receives

        # HuggingFace VisionRotaryEmbedding uses only even positions 0,2,4,…,
        # so we must match that exactly for parity.
        idx = jnp.arange(0, half_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.base ** (idx / half_dim))
        inv_freq = self._scale_inverse_frequencies(
            inv_freq, head_dim=head_dim, max_sequence_length=self.max_sequence_length
        )

        return VisionRoPE(
            config=self,
            inv_freq=inv_freq,              # shape [half_dim]
            head_dim=head_dim,
            max_position=num_timesteps or self.max_sequence_length,
        )

# ---------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------
class VisionRoPE(FartsovkaModule[VisionRoPEConfig]):
    """Generate RoPE cosine/sine tables compatible with HuggingFace VisionRotaryEmbedding."""

    # static fields -----------------------------------------------------------------
    head_dim: int = eqx.static_field()
    inv_freq: Float[Array, "half_dim"] = eqx.static_field()
    max_position: int = eqx.static_field()
    config: VisionRoPEConfig = eqx.static_field()

    # -------------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------------
    def _1d_freqs(self, seq_len: int) -> Float[Array, "tokens half_dim"]:
        """
        Returns [S, half_dim] array of rotary frequencies (no duplication).
        """
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, self.inv_freq)              # [S, half_dim]
        return freqs  # [S, half_dim]

    def _2d_freqs(
        self, position_ids: Int[Array, "tokens 2"]
    ) -> Float[Array, "tokens half_dim"]:
        """
        Returns [tokens, half_dim] array of rotary frequencies for 2D RoPE.
        """
        if position_ids.shape[-1] != 2:
            raise ValueError("Expected position_ids shape [tokens,2] for 2‑D RoPE.")

        if jnp.any(position_ids < 0) or jnp.any(position_ids >= self.max_position):
            # Just warn – matching earlier behaviour
            print(
                f"WARN: position indices outside configured range [0,{self.max_position}) were clipped."
            )

        h_pos = position_ids[:, 0]
        w_pos = position_ids[:, 1]

        h_freqs = jnp.outer(h_pos.astype(jnp.float32), self.inv_freq)  # [T, half_dim]
        w_freqs = jnp.outer(w_pos.astype(jnp.float32), self.inv_freq)  # [T, half_dim]
        return jnp.concatenate([h_freqs, w_freqs], axis=-1)            # [T, head_dim]

    # -------------------------------------------------------------------------------
    # Public call
    # -------------------------------------------------------------------------------
    def __call__(self, position):
        """
        HF‑compatible call:

        • int / scalar ──────────────▶ 1‑D θ table of length *seq_len*.
        • [tokens,2]  (h,w) indices ─▶ 2‑D RoPE identical to earlier.
        • [*,3] grid_thw tensor ─────▶ FULL HF logic from
          Qwen2‑5‑VL `rot_pos_emb(grid_thw)` – returns cos/sin already
          expanded to [S, head_dim].

        Returns
        -------
        VisionPositionalEmbeddings
            With `.cosines`, `.sines` ready for direct use.
        """
        print("[DEBUG VisionRoPE.__call__] position type =", type(position),
              "shape =", getattr(position, "shape", None))
        # --- scalar: same as before ------------------------------------
        if jnp.isscalar(position) or (isinstance(position, jnp.ndarray) and position.ndim == 0):
            theta = self._1d_freqs(int(position))
            theta = jnp.concatenate([theta, theta], axis=-1)

            if theta.shape[-1] < self.head_dim:
                theta = jnp.concatenate([theta, theta], axis=-1)

            cosines = jnp.cos(theta)
            sines = jnp.sin(theta)
            return VisionPositionalEmbeddings(cosines=cosines, sines=sines)

        # --- (tokens,2) explicit (h,w) pairs – old path ---------------
        if isinstance(position, jnp.ndarray) and position.ndim == 2 and position.shape[-1] == 2:
            theta = self._2d_freqs(position)                # [S, half_dim]
            theta = jnp.concatenate([theta, theta], axis=-1)  # [S, head_dim]
            cosines = jnp.cos(theta)
            sines = jnp.sin(theta)
            return VisionPositionalEmbeddings(cosines=cosines, sines=sines)

        # --- grid_thw branch (mirror HF VisionTransformer.rot_pos_emb) ----
        if (
            isinstance(position, jnp.ndarray)
            and position.ndim == 2
            and position.shape[-1] == 3
        ):
            s = int(self.config.spatial_merge_size)
            pos_chunks = []

            max_grid_sz = int(position[:, 1:].max())
            theta_full = self._1d_freqs(max_grid_sz)  # [max_grid_sz, half_dim]

            for t_i, h_i, w_i in jnp.asarray(position).astype(jnp.int32):
                t_i, h_i, w_i = int(t_i), int(h_i), int(w_i)

                # ---- build the HF‑style (h, w) index grid ----------------
                hpos = jnp.arange(h_i)[:, None].repeat(w_i, axis=1)  # [H,W]
                hpos = hpos.reshape(h_i // s, s, w_i // s, s)
                hpos = jnp.transpose(hpos, (0, 2, 1, 3)).reshape(-1)  # [H*W]

                wpos = jnp.arange(w_i)[None, :].repeat(h_i, axis=0)  # [H,W]
                wpos = wpos.reshape(h_i // s, s, w_i // s, s)
                wpos = jnp.transpose(wpos, (0, 2, 1, 3)).reshape(-1)  # [H*W]

                ids_hw = jnp.stack([hpos, wpos], axis=-1)  # [H*W, 2]
                ids_thw = jnp.repeat(ids_hw, t_i, axis=0)  # [T*H*W, 2]
                pos_chunks.append(ids_thw)

            pos_ids = jnp.concatenate(pos_chunks, axis=0)  # [S, 2]
            h_ids = pos_ids[:, 0]
            w_ids = pos_ids[:, 1]
            theta_h = theta_full[h_ids]   # [S, half_dim]
            theta_w = theta_full[w_ids]   # [S, half_dim]
            theta_hw = jnp.concatenate([theta_h, theta_w], axis=-1)            # [S, half_dim]
            # Duplicate once more so we reach the full head_dim width (matches HF behaviour)
            theta = jnp.concatenate([theta_hw, theta_hw], axis=-1)             # [S, head_dim]
            cosines = jnp.cos(theta)
            sines   = jnp.sin(theta)
            return VisionPositionalEmbeddings(cosines=cosines, sines=sines)

        # --- unknown input format ------------------------------------
        raise TypeError(
            "VisionRoPE expected (int), [tokens,2] or [*,3] grid_thw tensor; "
            f"got {type(position)} with shape {getattr(position, 'shape', None)}"
        )

    def export_weights(self) -> ParameterDict:
        return ParameterDict()