import jax
import jax.numpy as jnp
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from lalamo.modules.mrope import MRoPEConfig, MRoPELayout
from lalamo.modules.rope import UnscaledRoPEConfig
from tests.common import assert_close, tolerance

# Qwen3.5 text attention: head_dim 256 with partial_rotary_factor 0.25, so 64 rotary channels.
ROPE_BASE = 10_000_000.0
ROTARY_DIM = 64
SECTIONS = (11, 11, 10)
NUM_TOKENS = 40


def _mrope(layout: MRoPELayout = MRoPELayout.INTERLEAVED) -> MRoPEConfig:
    return MRoPEConfig(
        base=ROPE_BASE,
        max_sequence_length=262_144,
        head_dim=ROTARY_DIM,
        sections=SECTIONS,
        layout=layout,
    )


def _random_positions() -> jax.Array:
    return jax.random.randint(jax.random.key(0), (NUM_TOKENS, 3), 0, 5000, dtype=jnp.int32)


def test_matches_huggingface_interleaved_mrope() -> None:
    hf_config = Qwen3_5TextConfig(
        head_dim=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_BASE,
            "partial_rotary_factor": ROTARY_DIM / 256,
            "mrope_section": list(SECTIONS),
            "mrope_interleaved": True,
        },
    )
    positions = _random_positions()
    hf_positions = torch.from_numpy(jax.device_get(positions).T.copy())[:, None, :]
    hf_cosines, hf_sines = Qwen3_5TextRotaryEmbedding(hf_config)(torch.zeros(1, dtype=torch.float32), hf_positions)

    result = _mrope().compute_positional_embeddings(positions)
    # Angles reach thousands of radians, where float32 sin/cos in XLA and torch differ by ~1.5e-5.
    # lalamo's plain RoPE shows the same gap against HF at these positions.
    with tolerance(atol=5e-5, rtol=1e-5):
        assert_close(result=result.cosines, reference=jnp.asarray(hf_cosines[0].numpy()), operation_name="cosines")
        assert_close(result=result.sines, reference=jnp.asarray(hf_sines[0].numpy()), operation_name="sines")


@pytest.mark.parametrize("layout", list(MRoPELayout))
def test_scalar_positions_reduce_to_plain_rope(layout: MRoPELayout) -> None:
    timesteps = jnp.arange(NUM_TOKENS, dtype=jnp.int32) * 37
    plain = UnscaledRoPEConfig(base=ROPE_BASE, max_sequence_length=262_144, head_dim=ROTARY_DIM)
    expected = plain.compute_positional_embeddings(timesteps)
    for positions in (timesteps, jnp.repeat(timesteps[:, None], 3, axis=1)):
        result = _mrope(layout).compute_positional_embeddings(positions)
        assert jnp.array_equal(result.cosines, expected.cosines)
        assert jnp.array_equal(result.sines, expected.sines)


def test_axis_assignment() -> None:
    interleaved = _mrope(MRoPELayout.INTERLEAVED).pair_axes().tolist()
    assert interleaved[:6] == [0, 1, 2, 0, 1, 2]
    assert [interleaved.count(axis) for axis in range(3)] == list(SECTIONS)
    chunked = _mrope(MRoPELayout.CHUNKED).pair_axes().tolist()
    assert chunked == [0] * 11 + [1] * 11 + [2] * 10


def test_rejects_inconsistent_sections() -> None:
    with pytest.raises(ValueError, match="sections"):
        MRoPEConfig(
            base=1.0, max_sequence_length=8, head_dim=ROTARY_DIM, sections=(1, 1, 1), layout=MRoPELayout.CHUNKED
        )
