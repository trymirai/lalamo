from lalamo.bitshift import (
    OneMultiplyAddHashLUTProvider,
    ThreeInstructionHashLUTProvider,
    TwoMultiplyAddHashLUTProvider,
)
from lalamo.bitshift.bitshift_codebook_config import BitshiftCodebookConfig


def build_config(values_per_step: int) -> BitshiftCodebookConfig:
    return BitshiftCodebookConfig(state_bits=16, bits_per_weight=2, values_per_step=values_per_step)


def test_one_multiply_add_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = OneMultiplyAddHashLUTProvider()
    lut = lut_provider.build_lut(config)
    assert lut.shape == (values_per_step, config.number_of_states)


def test_two_multiply_add_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = TwoMultiplyAddHashLUTProvider()
    lut = lut_provider.build_lut(config)
    assert lut.shape == (values_per_step, config.number_of_states)


def test_three_instruction_hash_lut_provider() -> None:
    values_per_step = 1
    config = build_config(values_per_step)
    lut_provider = ThreeInstructionHashLUTProvider()
    lut = lut_provider.build_lut(config)
    assert lut.shape == (values_per_step, config.number_of_states)
