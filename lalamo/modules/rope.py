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
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule, field
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import lookup_sharded_indices

__all__ = [
    "LinearScalingRoPEConfig",
    "LlamaRoPEConfig",
    "PositionalEmbeddings",
    "RoPE",
    "RoPEConfig",
    "UnscaledRoPEConfig",
    "YARNRoPEConfig",
]


class PositionalEmbeddings(Exportable, eqx.Module):
    cosines: Float[Array, "*batch tokens head_channels"]
    sines: Float[Array, "*batch tokens head_channels"]

    @property
    def head_dim(self) -> int:
        return self.cosines.shape[-1]

    def rotate_half(
        self,
        heads: Float[Array, "tokens head_channels"],
    ) -> Float[Array, "tokens head_channels"]:
        half_dim = self.head_dim // 2
        x1 = heads[..., :half_dim]
        x2 = heads[..., half_dim : self.head_dim]
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply(self, heads: Float[Array, "tokens head_channels"]) -> Float[Array, "tokens head_channels"]:
        head_dim = self.head_dim
        if heads.shape[-1] < head_dim:
            raise ValueError(
                f"RoPE head_dim {head_dim} exceeds input head_dim {heads.shape[-1]}",
            )
        cosines = self.cosines.astype(heads.dtype)
        sines = self.sines.astype(heads.dtype)
        rotated = heads[..., :head_dim]
        rotated = rotated * cosines + self.rotate_half(rotated) * sines
        if heads.shape[-1] == head_dim:
            return rotated
        tail = heads[..., head_dim:]
        return jnp.concatenate([rotated, tail], axis=-1)


@dataclass(frozen=True)
class RoPEConfig(LalamoConfig, RegistryABC):
    base: float
    max_sequence_length: int
    head_dim: int | None = field(default=None, kw_only=True)
    partial_rotary_dim: int | None = field(default=None, kw_only=True)

    @property
    def _attention_scaling_factor(self) -> float:
        return 1.0

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " rotary_pairs"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " rotary_pairs"]:
        return inverse_frequencies

    def _mask_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " rotary_pairs"],
        head_dim: int,
    ) -> Float[Array, " rotary_pairs"]:
        if self.partial_rotary_dim is None or self.partial_rotary_dim >= head_dim:
            return inverse_frequencies
        rope_angles = self.partial_rotary_dim // 2
        mask = jnp.arange(head_dim // 2) < rope_angles
        return inverse_frequencies * mask

    def init(
        self,
        initializer: Initializer,
        head_dim: int | None = None,
        num_timesteps: int | None = None,
    ) -> "RoPE":
        resolved_head_dim = head_dim or self.head_dim
        if resolved_head_dim is None:
            raise ValueError("RoPE head_dim must be specified either in the config or at init time.")
        resolved_num_timesteps = num_timesteps or self.max_sequence_length
        timesteps = jnp.arange(resolved_num_timesteps, dtype=jnp.float32)
        channel_indices = jnp.arange(0, resolved_head_dim, 2, dtype=jnp.int32)
        inverse_frequencies = 1.0 / (self.base ** (channel_indices.astype(jnp.float32) / resolved_head_dim))
        inverse_frequencies = self._scale_inverse_frequencies(
            inverse_frequencies,
            resolved_head_dim,
            self.max_sequence_length,
        )
        inverse_frequencies = self._mask_inverse_frequencies(inverse_frequencies, resolved_head_dim)
        outer_inverse_frequencies = jnp.outer(timesteps, inverse_frequencies)
        embeddings = jnp.concatenate((outer_inverse_frequencies, outer_inverse_frequencies), axis=-1)
        table_sharding = initializer.sharding_config.resolve_sharding((None, None))
        cosines = jax.device_put(
            (jnp.cos(embeddings) * self._attention_scaling_factor).astype(initializer.default_dtype),
            table_sharding,
        )
        sines = jax.device_put(
            (jnp.sin(embeddings) * self._attention_scaling_factor).astype(initializer.default_dtype),
            table_sharding,
        )
        return RoPE(
            config=self,
            sharding_config=initializer.sharding_config,
            sines=sines,
            cosines=cosines,
        )


class RoPE(LalamoModule[RoPEConfig]):
    sines: Float[Array, "tokens head_channels"] = field(trainable=False)
    cosines: Float[Array, "tokens head_channels"] = field(trainable=False)

    @property
    def head_dim(self) -> int:
        _, result = self.sines.shape
        return result

    @property
    def max_sequence_length(self) -> int:
        result, _ = self.sines.shape
        return result

    @eqx.filter_jit
    def __call__(self, timesteps: Int[Array, " tokens"]) -> PositionalEmbeddings:
        return PositionalEmbeddings(
            cosines=lookup_sharded_indices(self.cosines, timesteps),
            sines=lookup_sharded_indices(self.sines, timesteps),
        )


class UnscaledRoPEConfig(RoPEConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class LlamaRoPEConfig(RoPEConfig):
    scaling_factor: float
    original_context_length: int
    low_frequency_factor: float
    high_frequency_factor: float

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " rotary_pairs"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " rotary_pairs"]:
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
        return result + scaled_frequencies * low_frequency_mask.astype(jnp.float32)


@dataclass(frozen=True, kw_only=True)
class YARNRoPEConfig(RoPEConfig):
    scaling_factor: float
    original_context_length: int
    beta_fast: float
    beta_slow: float
    truncate: bool

    @classmethod
    def _find_correction_dim(cls, num_rotations: float, dim: int, base: float, original_context_length: int) -> float:
        return (dim * math.log(original_context_length / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    @classmethod
    def _find_correction_range(
        cls,
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        original_context_length: int,
        truncate: bool,
    ) -> tuple[float, float]:
        low = cls._find_correction_dim(low_rot, dim, base, original_context_length)
        high = cls._find_correction_dim(high_rot, dim, base, original_context_length)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return float(max(low, 0.0)), float(min(high, float(dim - 1)))

    @classmethod
    def _linear_ramp_factor(cls, min_value: float, max_value: float, dim: int) -> Float[Array, " rotary_pairs"]:
        if min_value == max_value:
            max_value += 0.001

        min_v = jnp.float32(min_value)
        max_v = jnp.float32(max_value)
        linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_v) / (max_v - min_v)
        return jnp.clip(linear_func, 0, 1)

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " rotary_pairs"],
        head_dim: int,
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " rotary_pairs"]:
        scaled_frequencies = inverse_frequencies / self.scaling_factor

        low, high = self._find_correction_range(
            self.beta_fast,
            self.beta_slow,
            head_dim,
            self.base,
            self.original_context_length,
            self.truncate,
        )

        smoothing_factor = 1 - self._linear_ramp_factor(low, high, head_dim // 2)
        return scaled_frequencies * (1 - smoothing_factor) + inverse_frequencies * smoothing_factor

    @property
    def _attention_scaling_factor(self) -> float:
        return 0.1 * math.log(self.scaling_factor) + 1.0


@dataclass(frozen=True, kw_only=True)
class LinearScalingRoPEConfig(RoPEConfig):
    scaling_factor: float

    def _scale_inverse_frequencies(
        self,
        inverse_frequencies: Float[Array, " rotary_pairs"],
        head_dim: int,  # noqa: ARG002
        max_sequence_length: int,  # noqa: ARG002
    ) -> Float[Array, " rotary_pairs"]:
        return inverse_frequencies / self.scaling_factor
