from abc import abstractmethod

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.random
from jaxtyping import Array, Float, PRNGKeyArray

from .bitshift_codebook_config import BitShiftCodebookConfig

__all__ = [
    "FixedLUTProvider",
    "GaussianLUTProvider",
    "LUTProvider",
    "OneMultiplyAddHashLUTProvider",
    "ThreeInstructionHashLUTProvider",
    "TwoMultiplyAddHashLUTProvider",
]


class LUTProvider(eqx.Module):
    lut: Float[Array, "chunk_size number_of_states"]

    @classmethod
    def create(cls, config: BitShiftCodebookConfig, key: PRNGKeyArray) -> "LUTProvider":
        return cls(lut=cls._initialize_lut(config, key))

    @staticmethod
    @abstractmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]: ...


class FixedLUTProvider(LUTProvider):
    @staticmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]:
        raise NotImplementedError("Use create_from_lut instead")

    @classmethod
    def create_from_lut(cls, lut: Float[Array, "chunk_size number_of_states"]) -> "FixedLUTProvider":
        return cls(lut=lut)


class GaussianLUTProvider(LUTProvider):
    @staticmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]:
        return jax.random.normal(key, (config.chunk_size, config.number_of_states))


class OneMultiplyAddHashLUTProvider(LUTProvider):
    @staticmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]:
        _ = key
        assert config.chunk_size == 1

        state_indices = jnp.arange(config.number_of_states, dtype=jnp.uint32)

        hash_result = state_indices * jnp.uint32(34038481) + jnp.uint32(76625530)
        hash_result = (
            (hash_result & jnp.uint32(255))
            + ((hash_result >> 8) & jnp.uint32(255))
            + ((hash_result >> 16) & jnp.uint32(255))
            + ((hash_result >> 24) & jnp.uint32(255))
        )

        normalized_values = (hash_result.astype(jnp.float32) - 510.0) / 147.800537109375
        return jnp.expand_dims(normalized_values, axis=0)


class TwoMultiplyAddHashLUTProvider(LUTProvider):
    @staticmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]:
        _ = key
        assert config.chunk_size == 1

        state_indices = jnp.arange(config.number_of_states, dtype=jnp.uint32)

        multiplier = jnp.uint32(1664525)
        multiplier_lower_half = multiplier & jnp.uint32(0xFFFF)
        multiplier_upper_half = multiplier >> 16

        hash_result = state_indices * jnp.uint32(264435761) + jnp.uint32(1013904223)
        hash_result_lower_half = hash_result & jnp.uint32(0xFFFF)
        hash_result_upper_half = hash_result >> 16
        hash_result_middle_product = (
            hash_result_upper_half * multiplier_lower_half + hash_result_lower_half * multiplier_upper_half
        )
        hash_result_upper_product = hash_result_upper_half * multiplier_upper_half + (hash_result_middle_product >> 16)
        hash_result = hash_result_upper_product + hash_result
        hash_result = (
            (hash_result & jnp.uint32(255))
            + ((hash_result >> 8) & jnp.uint32(255))
            + ((hash_result >> 16) & jnp.uint32(255))
            + ((hash_result >> 24) & jnp.uint32(255))
        )

        normalized_values = (hash_result.astype(jnp.float32) - 510.0) / 147.800537109375
        return jnp.expand_dims(normalized_values, axis=0)


class ThreeInstructionHashLUTProvider(LUTProvider):
    @staticmethod
    def _initialize_lut(
        config: BitShiftCodebookConfig,
        key: PRNGKeyArray,
    ) -> Float[Array, "chunk_size number_of_states"]:
        _ = key
        assert config.chunk_size == 1

        state_indices = jnp.arange(config.number_of_states, dtype=jnp.uint32)

        float_pattern_mask = jnp.uint32(996162400)
        extraction_mask = jnp.uint32(((1 << 15) + ((1 << 12) - 1)) << 16 | ((1 << 15) + ((1 << 12) - 1)))

        hash_result = state_indices * jnp.uint32(89226354) + jnp.uint32(64248484)
        hash_result = (hash_result & extraction_mask) ^ float_pattern_mask
        hash_result_upper_half = (hash_result >> 16).astype(jnp.int16)
        hash_result_lower_half = (hash_result & jnp.uint32(0xFFFF)).astype(jnp.int16)
        hash_result_upper_float = jax.lax.bitcast_convert_type(hash_result_upper_half, jnp.float16)
        hash_result_lower_float = jax.lax.bitcast_convert_type(hash_result_lower_half, jnp.float16)

        normalized_values = (hash_result_upper_float + hash_result_lower_float).astype(jnp.float32)
        return jnp.expand_dims(normalized_values, axis=0)
