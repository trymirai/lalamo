# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
# Original PyTorch code copyright notice:
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from fartsovka.common import DEFAULT_PRECISION, DType

from .common import FartsovkaModule, ModuleConfig, ParameterDict

__all__ = [
    "PositionalEmbeddings",
    "AbstractRoPE",
    "AbstractRoPEConfig",
    "RoPEConfig",
    "LlamaRoPE",
    "LlamaRoPEConfig",
    "YARNRoPE",
    "YARNRoPEConfig",
]


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


class AbstractRoPE(FartsovkaModule):
    head_dim: int = eqx.field(static=True)
    max_sequence_length: int = eqx.field(static=True)
    theta: float = eqx.field(static=True)

    precision: DType = eqx.field(static=True)

    @property
    def attention_scaling_factor(self) -> float:
        return 1.0

    sines: Float[Array, "tokens head_channels"]
    cosines: Float[Array, "tokens head_channels"]

    def _scale_frequencies(
        self,
        frequencies: Float[Array, " tokens"],
    ) -> Float[Array, " tokens"]:
        return frequencies

    def _precompute_values(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> tuple[Float[Array, "{num_timesteps} {head_dim}"], Float[Array, "{num_timesteps} {head_dim}"]]:
        timesteps = jnp.arange(num_timesteps, dtype=jnp.float32)
        channel_indices = jnp.arange(0, head_dim, 2, dtype=jnp.int32)
        frequencies = 1.0 / (self.theta ** (channel_indices.astype(jnp.float32) / head_dim))
        frequencies = self._scale_frequencies(frequencies)
        outer_frequencies = jnp.outer(timesteps, frequencies)
        embeddings = jnp.concatenate((outer_frequencies, outer_frequencies), axis=-1)
        cosines = jnp.cos(embeddings).astype(self.precision) * self.attention_scaling_factor
        sines = jnp.sin(embeddings).astype(self.precision) * self.attention_scaling_factor
        return cosines, sines

    def __init__(
        self,
        *,
        head_dim: int,
        max_sequence_length: int,
        theta: float,
        precision: DType,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.theta = theta
        self.precision = precision

        self.cosines, self.sines = self._precompute_values(head_dim, max_sequence_length)

    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )

    def export_weights(self) -> ParameterDict:
        return ParameterDict(cosines=self.cosines, sines=self.sines)


@dataclass
class AbstractRoPEConfig[RoPEType: AbstractRoPE](ModuleConfig[RoPEType]):
    precision: DType = DEFAULT_PRECISION

    def __call__(self, head_dim: int, max_sequence_length: int, theta: float) -> RoPEType:
        raise NotImplementedError


@dataclass
class RoPEConfig(AbstractRoPEConfig[AbstractRoPE]):
    def __call__(self, head_dim: int, max_sequence_length: int, theta: float) -> AbstractRoPE:
        return AbstractRoPE(
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            theta=theta,
            precision=self.precision,
        )


class LlamaRoPE(AbstractRoPE):
    scaling_factor: float = eqx.field(static=True)
    original_context_length: int = eqx.field(static=True)
    low_frequency_factor: float = eqx.field(static=True)
    high_frequency_factor: float = eqx.field(static=True)

    def _scale_frequencies(
        self,
        frequencies: Float[Array, " tokens"],
    ) -> Float[Array, " tokens"]:
        low_frequency_wavelength = self.original_context_length / self.low_frequency_factor
        high_frequency_wavelength = self.original_context_length / self.high_frequency_factor

        wavelengths = 2 * jnp.pi / frequencies

        high_frequency_mask = wavelengths < high_frequency_wavelength
        low_frequency_mask = wavelengths > low_frequency_wavelength
        mid_frequency_mask = (~high_frequency_mask) & (~low_frequency_mask)

        smoothing_factors = self.original_context_length / wavelengths - self.low_frequency_factor
        smoothing_factors = smoothing_factors / (self.high_frequency_factor - self.low_frequency_factor)

        scaled_frequencies = frequencies / self.scaling_factor
        smoothly_scaled_frequencies = smoothing_factors * frequencies + (1 - smoothing_factors) * scaled_frequencies

        result = frequencies * high_frequency_mask.astype(jnp.float32)
        result = result + smoothly_scaled_frequencies * mid_frequency_mask.astype(jnp.float32)
        result = result + scaled_frequencies * low_frequency_mask.astype(jnp.float32)

        return result

    def __init__(
        self,
        *,
        head_dim: int,
        max_sequence_length: int,
        theta: float,
        precision: DType,
        scaling_factor: float,
        original_context_length: int,
        low_frequency_factor: float,
        high_frequency_factor: float,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.original_context_length = original_context_length
        self.low_frequency_factor = low_frequency_factor
        self.high_frequency_factor = high_frequency_factor

        super().__init__(
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            theta=theta,
            precision=precision,
        )

    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )

    def export_weights(self) -> ParameterDict:
        return ParameterDict(cosines=self.cosines, sines=self.sines)


class LlamaRoPEConfig(AbstractRoPEConfig[LlamaRoPE]):
    scaling_factor: float
    original_context_length: int
    low_frequency_factor: float
    high_frequency_factor: float

    def __init__(
        self,
        *,
        precision: DType,
        scaling_factor: float,
        original_context_length: int,
        low_frequency_factor: float,
        high_frequency_factor: float,
    ) -> None:
        super().__init__(precision=precision)

        self.scaling_factor = scaling_factor
        self.original_context_length = original_context_length
        self.low_frequency_factor = low_frequency_factor
        self.high_frequency_factor = high_frequency_factor

    def __call__(self, head_dim: int, max_sequence_length: int, theta: float) -> LlamaRoPE:
        return LlamaRoPE(
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            theta=theta,
            precision=self.precision,
            scaling_factor=self.scaling_factor,
            original_context_length=self.original_context_length,
            low_frequency_factor=self.low_frequency_factor,
            high_frequency_factor=self.high_frequency_factor,
        )


class YARNRoPE(AbstractRoPE):
    scaling_factor: float = eqx.field(static=True)
    beta_fast: float = eqx.field(static=True)
    beta_slow: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        head_dim: int,
        max_sequence_length: int,
        theta: float,
        precision: DType,
        scaling_factor: float,
        beta_fast: float,
        beta_slow: float,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        super().__init__(
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            theta=theta,
            precision=precision,
        )

    @classmethod
    def _find_correction_dim(cls, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    @classmethod
    def _find_correction_range(
        cls,
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> tuple[int, int]:
        """Find dimension range bounds based on rotations"""
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    @classmethod
    def _linear_ramp_factor(cls, min_value: float, max_value: float, dim: int) -> Float[Array, " head_dim"]:
        if min_value == max_value:
            max_value += 0.001  # Prevent singularity

        linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_value) / (max_value - min_value)
        ramp_func = jnp.clip(linear_func, 0, 1)
        return ramp_func

    def _scale_frequencies(
        self,
        frequencies: Float[Array, " tokens"],
    ) -> Float[Array, " tokens"]:
        scaled_frequencies = frequencies / self.scaling_factor

        low, high = self._find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.head_dim,
            self.theta,
            self.max_sequence_length,
        )

        # Get n-dimensional rotational scaling corrected for extrapolation
        smoothing_factor = 1 - self._linear_ramp_factor(low, high, self.head_dim // 2)
        return scaled_frequencies * (1 - smoothing_factor) + frequencies * smoothing_factor

    @property
    def attention_scaling_factor(self) -> float:
        return 0.1 * math.log(self.scaling_factor) + 1.0


class YARNRoPEConfig(AbstractRoPEConfig[YARNRoPE]):
    scaling_factor: float
    beta_fast: float
    beta_slow: float

    def __init__(self, *, precision: DType, scaling_factor: float, beta_fast: float, beta_slow: float) -> None:
        super().__init__(precision=precision)

        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

    def __call__(self, head_dim: int, max_sequence_length: int, theta: float) -> YARNRoPE:
        return YARNRoPE(
            head_dim=head_dim,
            max_sequence_length=max_sequence_length,
            theta=theta,
            precision=self.precision,
            scaling_factor=self.scaling_factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )
