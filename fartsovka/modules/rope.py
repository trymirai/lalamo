from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from fartsovka.common import DEFAULT_PRECISION, DType

__all__ = ["PositionalEmbeddings", "RoPE", "RoPEParams", "RoPEFactory"]


class PositionalEmbeddings(eqx.Module):
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


@dataclass
class RoPEParams:
    theta: float
    use_scaling: bool
    scaling_factor: float
    original_context_length: int
    low_frequency_factor: float = 1
    high_frequency_factor: float = 4


class RoPE(eqx.Module):
    head_dim: int = eqx.field(static=True)
    max_sequence_length: int = eqx.field(static=True)
    params: RoPEParams = eqx.field(static=True)

    precision: DType = eqx.field(static=True)

    sines: Float[Array, "tokens head_channels"]
    cosines: Float[Array, "tokens head_channels"]

    def _scale_frequencies(
        self,
        frequencies: Float[Array, " tokens"],
    ) -> Float[Array, " tokens"]:
        low_frequency_wavelength = self.params.original_context_length / self.params.low_frequency_factor
        high_frequency_wavelength = self.params.original_context_length / self.params.high_frequency_factor

        wavelengths = 2 * jnp.pi / frequencies

        high_frequency_mask = wavelengths < high_frequency_wavelength
        low_frequency_mask = wavelengths > low_frequency_wavelength
        mid_frequency_mask = (~high_frequency_mask) & (~low_frequency_mask)

        smoothing_factors = self.params.original_context_length / wavelengths - self.params.low_frequency_factor
        smoothing_factors = smoothing_factors / (self.params.high_frequency_factor - self.params.low_frequency_factor)

        scaled_frequencies = frequencies / self.params.scaling_factor
        smoothly_scaled_frequencies = smoothing_factors * frequencies + (1 - smoothing_factors) * scaled_frequencies

        result = frequencies * high_frequency_mask.astype(jnp.float32)
        result = result + smoothly_scaled_frequencies * mid_frequency_mask.astype(jnp.float32)
        result = result + scaled_frequencies * low_frequency_mask.astype(jnp.float32)

        return result

    def _precompute_values(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> tuple[Float[Array, "{num_timesteps} {head_dim}"], Float[Array, "{num_timesteps} {head_dim}"]]:
        timesteps = jnp.arange(num_timesteps, dtype=jnp.float32)
        channel_indices = jnp.arange(0, head_dim, 2, dtype=jnp.int32)
        frequencies = 1.0 / (self.params.theta ** (channel_indices.astype(jnp.float32) / head_dim))
        if self.params.use_scaling:
            frequencies = self._scale_frequencies(frequencies)
        outer_frequencies = jnp.outer(timesteps, frequencies)
        embeddings = jnp.concatenate((outer_frequencies, outer_frequencies), axis=-1)
        cosines = jnp.cos(embeddings).astype(self.precision)
        sines = jnp.sin(embeddings).astype(self.precision)
        return cosines, sines

    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        params: RoPEParams,
        precision: DType = DEFAULT_PRECISION,
    ) -> None:
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.params = params
        self.precision = precision

        self.cosines, self.sines = self._precompute_values(head_dim, max_sequence_length)

    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )


@dataclass
class RoPEFactory:
    precision: DType = dataclass_field(default=DEFAULT_PRECISION)

    def __call__(self, head_dim: int, max_sequence_length: int, params: RoPEParams) -> RoPE:
        return RoPE(head_dim, max_sequence_length, params, precision=self.precision)
