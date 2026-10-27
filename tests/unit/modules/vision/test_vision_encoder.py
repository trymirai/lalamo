import jax
import jax.numpy as jnp
import pytest
import torch

from lalamo.exportable import Exportable
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.modules.vision import ImageGrid, VisionAttentionImplementation, VisionForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close, gpu_only, tolerance

from .common import (
    GRID_CASES,
    hf_grid_thw,
    keychain,
    lalamo_encoder,
    random_patches,
    tiny_hf_vision_model,
    to_jax,
    to_torch,
)

GRID_PARAMS = [pytest.param(grids, id=name) for name, grids in GRID_CASES.items()]


@pytest.mark.parametrize("grids", GRID_PARAMS)
def test_encoder_matches_huggingface(grids: tuple[ImageGrid, ...]) -> None:
    patches = random_patches(grids)
    result = lalamo_encoder()(patches, grids, keychain=keychain())
    with torch.no_grad():
        reference = tiny_hf_vision_model()(to_torch(patches), grid_thw=hf_grid_thw(grids)).pooler_output

    assert result.outputs.shape == (lalamo_encoder().config.num_output_tokens(grids), 96)
    with tolerance(atol=1e-4, rtol=1e-4):
        assert_close(result=result.outputs, reference=to_jax(reference), operation_name="Encoder output")


def test_chunked_attention_matches_unchunked() -> None:
    grids = GRID_CASES["larger-than-table"]
    patches = random_patches(grids)
    unchunked = lalamo_encoder()(
        patches,
        grids,
        VisionForwardPassConfig(attention_query_chunk_size=None),
        keychain=keychain(),
    )
    # 7 does not divide the 168-patch frame, so the padded last chunk is exercised as well.
    chunked = lalamo_encoder()(
        patches,
        grids,
        VisionForwardPassConfig(attention_query_chunk_size=7),
        keychain=keychain(),
    )
    with tolerance(atol=1e-5, rtol=1e-5):
        assert_close(result=chunked.outputs, reference=unchunked.outputs, operation_name="Chunked attention")


def test_images_are_encoded_independently() -> None:
    first, second = GRID_CASES["two-images"]
    patches = random_patches((first, second))
    together = lalamo_encoder()(patches, (first, second), keychain=keychain()).outputs
    alone = lalamo_encoder()(patches[: first.num_patches], (first,), keychain=keychain()).outputs
    with tolerance(atol=1e-5, rtol=1e-5):
        assert_close(result=together[: alone.shape[0]], reference=alone, operation_name="First image")


def test_export_round_trip() -> None:
    grids = GRID_CASES["single-wide"]
    patches = random_patches(grids)
    encoder = lalamo_encoder()
    restored = encoder.load_exported(Exportable.export(encoder))
    assert_close(
        result=restored(patches, grids, keychain=keychain()).outputs,
        reference=encoder(patches, grids, keychain=keychain()).outputs,
    )


def test_rejects_inconsistent_inputs() -> None:
    grids = GRID_CASES["single-square"]
    with pytest.raises(ValueError, match="patches"):
        lalamo_encoder()(random_patches(grids)[1:], grids, keychain=keychain())
    with pytest.raises(ValueError, match="divisible"):
        odd_grid = (ImageGrid(1, 3, 4),)
        lalamo_encoder()(random_patches(odd_grid), odd_grid, keychain=keychain())


@gpu_only
def test_cudnn_attention_matches_xla() -> None:
    gpu_sharding = ShardingConfig.replicated(jax.devices("gpu")[:1])
    cpu_encoder = lalamo_encoder()
    template = cpu_encoder.config.init(EmptyInitializer(default_dtype=jnp.bfloat16, sharding_config=gpu_sharding))
    encoder = template.load_exported(Exportable.export(cpu_encoder))
    grids = GRID_CASES["two-images"]
    patches = jax.device_put(random_patches(grids), gpu_sharding.make_sharding((None, None)))
    gpu_keychain = Keychain.init(0, sharding_config=gpu_sharding)

    def encode(implementation: VisionAttentionImplementation) -> jax.Array:
        config = VisionForwardPassConfig(attention_implementation=implementation)
        return encoder(patches, grids, config, keychain=gpu_keychain).outputs

    # Both run in bfloat16; the tolerance covers bfloat16 rounding, not algorithmic differences.
    with tolerance(atol=5e-2, rtol=5e-2):
        assert_close(
            result=encode(VisionAttentionImplementation.CUDNN),
            reference=encode(VisionAttentionImplementation.XLA),
            operation_name="cuDNN vs XLA attention",
            fraction_of_allowed_violations=0.01,
        )
