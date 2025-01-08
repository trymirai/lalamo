from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
from einops import einsum
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from .common import DEFAULT_PRECISION

__all__ = ["PositionalEmbeddings", "RoPE"]


@dataclass
class PositionalEmbeddings:
    cosines: Float[Array, "tokens head_channels"]
    sines: Float[Array, "tokens head_channels"]

    @property
    def head_dim(self) -> int:
        return self.cosines.shape[-1]

    def rotate_half(self, heads: Float[Array, "tokens head_channels"]) -> Float[Array, "tokens head_channels"]:
        x1 = heads[..., : self.head_dim // 2]
        x2 = heads[..., self.head_dim // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply(self, heads: Float[Array, "tokens head_channels"]) -> Float[Array, "tokens head_channels"]:
        return heads * self.cosines + self.rotate_half(heads) * self.sines


# Implementation is based on https://github.com/pytorch/executorch/blob/08770b73212872ba41da6412e951d73e66711c22/examples/models/llama/rope.py#L42
class RoPE(eqx.Module):
    head_dim: int = eqx.field(static=True)
    max_sequence_length: int = eqx.field(static=True)
    precision: jnp.dtype = eqx.field(static=True)

    sines: Float[Array, "tokens head_channels"] = eqx.field(static=True)
    cosines: Float[Array, "tokens head_channels"] = eqx.field(static=True)

    @classmethod
    def _scale_frequencies(
        cls,
        frequencies: Float[Array, " tokens"],
        scaling_factor: float,
        original_context_length: int,
    ) -> Float[Array, " tokens"]:
        # Magic hyperparameters from executorch
        low_frequency_factor = 1
        high_frequency_factor = 4

        low_frequency_wavelength = original_context_length / low_frequency_factor
        high_frequency_wavelength = original_context_length / high_frequency_factor

        wavelengths = 2 * jnp.pi / frequencies

        high_frequency_mask = wavelengths < high_frequency_wavelength
        low_frequency_mask = wavelengths > low_frequency_wavelength
        mid_frequency_mask = ~high_frequency_mask & ~low_frequency_mask

        high_frequencies = frequencies[high_frequency_mask]
        low_frequencies = frequencies[low_frequency_mask]
        mid_frequencies = frequencies[mid_frequency_mask]

        smoothing_factor = original_context_length / wavelengths[mid_frequency_mask] - low_frequency_factor
        smoothing_factor = smoothing_factor / (high_frequency_factor - low_frequency_factor)

        new_high_frequencies = high_frequencies
        new_low_frequencies = low_frequencies / scaling_factor
        new_mid_frequencies = smoothing_factor * mid_frequencies + (1 - smoothing_factor) * (
            mid_frequencies / scaling_factor
        )

        return jnp.concatenate((new_high_frequencies, new_mid_frequencies, new_low_frequencies), axis=0)

    @classmethod
    def _precompute_values(
        cls,
        head_dim: int,
        num_timesteps: int,
        theta: float,
        use_scaling: bool = False,
        scaling_factor: float = 8.0,
        original_context_length: int = 8192,
        precision: jnp.dtype = DEFAULT_PRECISION,
    ) -> tuple[Float[Array, "{num_timesteps} {head_dim}"], Float[Array, "{num_timesteps} {head_dim}"]]:
        channel_indices = jnp.arange(0, head_dim, 2, dtype=jnp.int32)
        timesteps = jnp.arange(num_timesteps, dtype=jnp.float32)
        frequencies = 1.0 / (theta ** (channel_indices.astype(jnp.float32) / head_dim))
        if use_scaling:
            frequencies = cls._scale_frequencies(frequencies, scaling_factor, original_context_length)
        outer_frequencies = einsum(timesteps, frequencies, "t, f -> t f")
        embeddings = jnp.concatenate((outer_frequencies, outer_frequencies), axis=-1)
        cosines = jnp.cos(embeddings).astype(precision)
        sines = jnp.sin(embeddings).astype(precision)
        return cosines, sines

    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        precision: jnp.dtype = DEFAULT_PRECISION,
        theta: float = 500000.0,
        use_scaling: bool = False,
        scaling_factor: float = 8.0,
        original_context_length: int = 8192,
    ) -> None:
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.precision = precision

        self.cosines, self.sines = self._precompute_values(
            head_dim,
            max_sequence_length,
            theta,
            use_scaling,
            scaling_factor,
            original_context_length,
            precision,
        )

    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )


@dataclass
class RoPEFactory:
    precision: jnp.dtype = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, head_dim: int, max_sequence_length: int) -> RoPE:
        return RoPE(head_dim, max_sequence_length, self.precision)
