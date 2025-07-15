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
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterDict

from .common import LalamoModule, WeightLayout, register_config_union

__all__ = [
    "LinearScalingRoPEConfig",
    "LlamaRoPEConfig",
    "PositionalEmbeddings",
    "RoPE",
    "RoPEConfigBase",
    "UnscaledRoPEConfig",
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

    def export(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(
            cosines=self.cosines,
            sines=self.sines,
        )


@dataclass(frozen=True)
class RoPEConfigBase:
    precision: DTypeLike
    base: float
    max_sequence_length: int

    @property
    def _attention_scaling_factor(self) -> float:
        return 1.0

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " tokens"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " tokens"]:
        return inverse_frequencies

    def init(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> "RoPE":
        timesteps = jnp.arange(num_timesteps, dtype=jnp.float32)
        channel_indices = jnp.arange(0, head_dim, 2, dtype=jnp.int32)
        inverse_frequencies = 1.0 / (self.base ** (channel_indices.astype(jnp.float32) / head_dim))
        inverse_frequencies = self._scale_inverse_frequencies(inverse_frequencies, head_dim, self.max_sequence_length)
        outer_inverse_frequencies = jnp.outer(timesteps, inverse_frequencies)
        embeddings = jnp.concatenate((outer_inverse_frequencies, outer_inverse_frequencies), axis=-1)
        cosines = (jnp.cos(embeddings) * self._attention_scaling_factor).astype(self.precision)
        sines = (jnp.sin(embeddings) * self._attention_scaling_factor).astype(self.precision)
        return RoPE(config=self, cosines=cosines, sines=sines)


class RoPE(LalamoModule[RoPEConfigBase]):
    sines: Float[Array, "tokens head_channels"]
    cosines: Float[Array, "tokens head_channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __post_init__(self) -> None:
        if self.cosines.dtype != self.config.precision:
            raise ValueError(
                f"Cosines dtype {self.cosines.dtype} does not match the specified precision {self.config.precision}",
            )
        if self.sines.dtype != self.config.precision:
            raise ValueError(
                f"Sines dtype {self.sines.dtype} does not match the specified precision {self.config.precision}",
            )
        if self.cosines.shape != self.sines.shape:
            raise ValueError(
                f"Cosines and sines shape mismatch: cosines have shape {self.cosines.shape},"
                f" while sines have shape {self.sines.shape}",
            )

    @property
    def head_dim(self) -> int:
        _, result = self.sines.shape
        return result

    @property
    def max_sequence_length(self) -> int:
        result, _ = self.sines.shape
        return result

    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(cosines=self.cosines, sines=self.sines)


class UnscaledRoPEConfig(RoPEConfigBase):
    pass


@dataclass(frozen=True)
class LlamaRoPEConfig(RoPEConfigBase):
    scaling_factor: float
    original_context_length: int
    low_frequency_factor: float
    high_frequency_factor: float

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " tokens"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " tokens"]:
        low_frequency_wavelength = self.original_context_length / self.low_frequency_factor
        high_frequency_wavelength = self.original_context_length / self.high_frequency_factor

        wavelengths = 2 * math.pi / inverse_frequencies

        high_frequency_mask = wavelengths < high_frequency_wavelength
        low_frequency_mask = wavelengths > low_frequency_wavelength
        mid_frequency_mask = (~high_frequency_mask) & (~low_frequency_mask)

        smoothing_factors = self.original_context_length / wavelengths - self.low_frequency_factor
        smoothing_factors = smoothing_factors / (self.high_frequency_factor - self.low_frequency_factor)

        scaled_frequencies = inverse_frequencies / self.scaling_factor
        smoothly_scaled_frequencies = (
            smoothing_factors * inverse_frequencies + (1 - smoothing_factors) * scaled_frequencies
        )

        result = inverse_frequencies * high_frequency_mask.astype(jnp.float32)
        result = result + smoothly_scaled_frequencies * mid_frequency_mask.astype(jnp.float32)
        result = result + scaled_frequencies * low_frequency_mask.astype(jnp.float32)

        return result


@dataclass(frozen=True)
class YARNRoPEConfig(RoPEConfigBase):
    scaling_factor: float
    beta_fast: float
    beta_slow: float

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

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " tokens"],
        head_dim: int,
        max_sequence_length: int,
    ) -> Float[Array, " tokens"]:
        scaled_frequencies = inverse_frequencies / self.scaling_factor

        low, high = self._find_correction_range(
            self.beta_fast,
            self.beta_slow,
            head_dim,
            self.base,
            max_sequence_length,
        )

        # Get n-dimensional rotational scaling corrected for extrapolation
        smoothing_factor = 1 - self._linear_ramp_factor(low, high, head_dim // 2)
        return scaled_frequencies * (1 - smoothing_factor) + inverse_frequencies * smoothing_factor

    @property
    def attention_scaling_factor(self) -> float:
        return 0.1 * math.log(self.scaling_factor) + 1.0


@dataclass(frozen=True)
class LinearScalingRoPEConfig(RoPEConfigBase):
    scaling_factor: float

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " tokens"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " tokens"]:
        return inverse_frequencies / self.scaling_factor


RoPEConfig = UnscaledRoPEConfig | LlamaRoPEConfig | YARNRoPEConfig | LinearScalingRoPEConfig

register_config_union(RoPEConfig)
