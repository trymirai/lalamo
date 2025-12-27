from collections.abc import Mapping
from dataclasses import dataclass, replace

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterTree
from lalamo.modules import LalamoModule
from lalamo.modules.rope import RoPEConfigBase


@dataclass(frozen=True)
class RoPEConfigFishAudio(RoPEConfigBase):
    @property
    def _attention_scaling_factor(self) -> float:
        return super()._attention_scaling_factor

    def _precompute_freqs_cis_orig(
        self, head_dim: int, seq_len: int
    ) -> tuple[Float[Array, "sequence head_dim"], Float[Array, "sequence head_dim"]]:
        time_steps = jnp.arange(0, head_dim // 2).astype(jnp.bfloat16) * 2 / head_dim
        freqs = 1.0 / (self.base**time_steps)
        t = jnp.arange(seq_len, device=freqs.device)
        freqs = jnp.outer(t, freqs)
        return (jnp.cos(freqs), jnp.sin(freqs))

    def init_orig(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> "RoPEFishAudio":
        cosines_cis, sines_cis = self._precompute_freqs_cis(head_dim, num_timesteps)
        cosines = jnp.zeros((num_timesteps, head_dim), self.precision)
        sines = jnp.zeros((num_timesteps, head_dim), self.precision)
        for k in range(num_timesteps):
            cosines = cosines.at[k, 0::2].set(cosines_cis[k])
            cosines = cosines.at[k, 1::2].set(cosines_cis[k])
            sines = sines.at[k, 0::2].set(sines_cis[k])
            sines = sines.at[k, 1::2].set(sines_cis[k])

        return RoPEFishAudio(config=self, cosines=cosines, sines=sines)

    def _precompute_freqs_cis(
        self, head_dim: int, seq_len: int
    ) -> tuple[Float[Array, "sequence head_dim"], Float[Array, "sequence head_dim"]]:
        # time_steps = jnp.arange(0, head_dim, 2).astype(jnp.bfloat16)[: (head_dim // 2)] / head_dim
        time_steps = jnp.repeat(jnp.arange(0, head_dim // 2).astype(jnp.bfloat16) * 2 / head_dim, 2)
        freqs = 1.0 / (self.base**time_steps)
        t = jnp.arange(seq_len, device=freqs.device)
        freqs = jnp.outer(t, freqs)
        return (jnp.cos(freqs), jnp.sin(freqs))

    def init(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> "RoPEFishAudio":
        cosines_cis, sines_cis = self._precompute_freqs_cis(head_dim, num_timesteps)
        return RoPEFishAudio(config=self, cosines=cosines_cis, sines=sines_cis)


class RoPEFishAudio(LalamoModule[RoPEConfigBase]):
    sines: Float[Array, "tokens head_channels"]
    cosines: Float[Array, "tokens head_channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def head_dim(self) -> int:
        _, result = self.sines.shape
        return result

    @property
    def max_sequence_length(self) -> int:
        result, _ = self.sines.shape
        return result

    # @eqx.filter_jit
    def __call__(self, timesteps: Int[Array, " tokens"]) -> "PositionalEmbeddingsFishAudio":
        return PositionalEmbeddingsFishAudio(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "cosines": self.cosines,
            "sines": self.sines,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> "RoPEFishAudio":
        assert isinstance(weights, Mapping)
        return replace(self, cosines=weights["cosines"], sines=weights["sines"])


class PositionalEmbeddingsFishAudio(eqx.Module):
    cosines: Float[Array, "*batch tokens head_channels"]
    sines: Float[Array, "*batch tokens head_channels"]

    @property
    def head_dim(self) -> int:
        return self.cosines.shape[-1]

    def interleave_for_cis_rope(
        self,
        heads: Float[Array, "*batch tokens head_channels"],
    ) -> Float[Array, "*batch tokens head_channels"]:
        interleaved = jnp.zeros(heads.shape, dtype=heads.dtype)
        interleaved = interleaved.at[..., 0::2].set(-heads[..., 1::2])
        interleaved = interleaved.at[..., 1::2].set(heads[..., 0::2])
        return interleaved

    def apply(self, heads: Float[Array, "*batch tokens head_channels"]) -> Float[Array, "*batch tokens head_channels"]:
        return heads * self.cosines + self.interleave_for_cis_rope(heads) * self.sines
