import jax.random

from lalamo.bitshift import (
    GaussianLUTProvider,
    OneMultiplyAddHashLUTProvider,
    ThreeInstructionHashLUTProvider,
    TwoMultiplyAddHashLUTProvider,
)
from lalamo.bitshift.bitshift_codebook import BitShiftCodebook
from lalamo.bitshift.bitshift_codebook_config import BitShiftCodebookConfig

RNG_KEY = jax.random.PRNGKey(42)


def build_config(values_per_step: int) -> BitShiftCodebookConfig:
    return BitShiftCodebookConfig(state_bits=16, bits_per_weight=2, values_per_step=values_per_step)


def test_one_multiply_add_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = OneMultiplyAddHashLUTProvider.create(config, key=RNG_KEY)
    assert lut_provider.lut.shape == (values_per_step, config.number_of_states)


def test_two_multiply_add_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = TwoMultiplyAddHashLUTProvider.create(config, key=RNG_KEY)
    assert lut_provider.lut.shape == (values_per_step, config.number_of_states)


def test_three_instruction_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = ThreeInstructionHashLUTProvider.create(config, key=RNG_KEY)
    assert lut_provider.lut.shape == (values_per_step, config.number_of_states)


def test_gaussian_lut_provider() -> None:
    values_per_step = 2
    config = build_config(values_per_step)
    lut_provider = GaussianLUTProvider.create(config, key=RNG_KEY)
    assert lut_provider.lut.shape == (values_per_step, config.number_of_states)


def test_bitshift_codebook() -> None:
    values_per_step = 2
    config = build_config(values_per_step)
    lut_provider = GaussianLUTProvider.create(config, key=RNG_KEY)
    codebook = BitShiftCodebook.create(config=config, lut_provider=lut_provider)
    assert codebook.lut.shape == (values_per_step, config.number_of_states)
    assert codebook.reconstructed_states.shape == (values_per_step, config.number_of_states)
