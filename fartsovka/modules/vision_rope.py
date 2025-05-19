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

class VisionPositionalEmbeddings:
    cosines: Float[Array, "tokens head_dim"]
    sines:   Float[Array, "tokens head_dim"]

    def __init__(self, *, cosines: Float[Array, "tokens head_dim"], sines: Float[Array, "tokens head_dim"]):
            self.cosines = cosines
            self.sines = sines


    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        cos, sin = self.cosines, self.sines                     # [S, Dh]
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
    base: float = 10000.0          
    max_sequence_length: int = 2048 

    duplicate_hw: bool = False
    spatial_merge_size: int = 2

    def _scale_inverse_frequencies(
        self,
        inv_freq: Float[Array, "half_dim"],
        *,
        head_dim: int,
        max_sequence_length: int,
    ) -> Float[Array, "half_dim"]:
        return inv_freq  # passthrough by default

    def init(
        self,
        *,
        head_dim: int,
        num_timesteps: Optional[int] = None,
    ) -> "VisionRoPE":
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for 2‑D RoPE.")
        half_dim = head_dim // 2

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

class VisionRoPE(FartsovkaModule[VisionRoPEConfig]):
    head_dim: int = eqx.static_field()
    inv_freq: Float[Array, "half_dim"] = eqx.static_field()
    max_position: int = eqx.static_field()
    config: VisionRoPEConfig = eqx.static_field()

    def freqs(self, seq_len: int) -> Float[Array, "tokens half_dim"]:
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, self.inv_freq)              # [S, half_dim]
        return freqs  # [S, half_dim]

    def __call__(self, position):
        if (
            isinstance(position, jnp.ndarray)
            and position.ndim == 2
            and position.shape[-1] == 3
        ):
            s = int(self.config.spatial_merge_size)
            pos_chunks = []

            max_grid_sz = int(position[:, 1:].max())
            theta_full = self.freqs(max_grid_sz)  # [max_grid_sz, half_dim]

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
            theta_hw = jnp.concatenate([theta_h, theta_w], axis=-1)

            theta = jnp.concatenate([theta_hw, theta_hw], axis=-1)
            cosines = jnp.cos(theta)
            sines   = jnp.sin(theta)

            return VisionPositionalEmbeddings(cosines=cosines, sines=sines)

        raise TypeError(
            "VisionRoPE expected (int), [tokens,2] or [*,3] grid_thw tensor; "
            f"got {type(position)} with shape {getattr(position, 'shape', None)}"
        )

    def export_weights(self) -> ParameterDict:
        return ParameterDict()
