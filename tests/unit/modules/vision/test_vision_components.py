"""Each vision building block against its HuggingFace counterpart, with shared random weights."""

import pytest
import torch
from jaxtyping import Array

from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.vision import ImageGrid
from tests.common import assert_close, tolerance

from .common import (
    GRID_CASES,
    hf_block,
    hf_cu_seqlens,
    hf_position_embeddings,
    hf_rotary_embeddings,
    keychain,
    lalamo_encoder,
    random_patches,
    tiny_hf_vision_model,
    to_jax,
    to_torch,
)

GRID_PARAMS = [pytest.param(grids, id=name) for name, grids in GRID_CASES.items()]


def _assert_matches(result: Array, reference: torch.Tensor, name: str) -> None:
    with tolerance(atol=1e-4, rtol=1e-4):
        assert_close(result=result, reference=to_jax(reference), operation_name=name)


def _hidden_states(grids: tuple[ImageGrid, ...]) -> Array:
    """Realistic layer inputs: patch embeddings plus position embeddings, as the encoder computes them."""
    result = lalamo_encoder()(random_patches(grids), grids, return_activation_trace=True, keychain=keychain())
    assert result.activation_trace is not None
    first_layer, *_ = result.activation_trace.layer_results
    assert first_layer.activation_trace is not None
    return first_layer.activation_trace.inputs


@pytest.mark.parametrize("grids", GRID_PARAMS)
def test_rope(grids: tuple[ImageGrid, ...]) -> None:
    encoder = lalamo_encoder()
    embeddings = encoder.config.rope_config.compute_positional_embeddings(grids, encoder.config.spatial_merge_size)
    reference_cosines, reference_sines = hf_rotary_embeddings(tiny_hf_vision_model(), grids)
    _assert_matches(embeddings.cosines, reference_cosines, "RoPE cosines")
    _assert_matches(embeddings.sines, reference_sines, "RoPE sines")


@pytest.mark.parametrize("grids", GRID_PARAMS)
def test_position_embedding(grids: tuple[ImageGrid, ...]) -> None:
    encoder = lalamo_encoder()
    result = encoder.position_embedding(grids, encoder.config.spatial_merge_size)
    reference = hf_position_embeddings(tiny_hf_vision_model(), grids)
    _assert_matches(result, reference, "Position embedding")


@pytest.mark.parametrize("grids", GRID_PARAMS)
def test_layer(grids: tuple[ImageGrid, ...]) -> None:
    encoder = lalamo_encoder()
    hidden_states = _hidden_states(grids)
    embeddings = encoder.config.rope_config.compute_positional_embeddings(grids, encoder.config.spatial_merge_size)
    result = encoder.layers[1](
        hidden_states,
        embeddings,
        grids,
        MixerForwardPassConfig(attention_implementation=AttentionImplementation.XLA),
        keychain=keychain(),
    )
    reference = hf_block(1)(
        to_torch(hidden_states),
        cu_seqlens=hf_cu_seqlens(grids),
        position_embeddings=hf_rotary_embeddings(tiny_hf_vision_model(), grids),
    )
    _assert_matches(result.outputs, reference, "Layer")
