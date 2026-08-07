import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed.hybrid import HybridSpec
from lalamo.compressed.low_rank_preview_readout import LowRankPreviewReadoutMatrix, LowRankPreviewReadoutSpec
from lalamo.module import Keychain
from lalamo.modules.utils import call_vmapped
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionSpec, Layout
from tests.common import assert_close, gpu_only

pytestmark = [gpu_only, pytest.mark.slow]


def test_candidate_readout_matches_ordered_block_tail_and_tie_reference() -> None:
    (device, *_) = jax.devices("gpu")
    if not device.device_kind.startswith("NVIDIA") or float(getattr(device, "compute_capability", 0)) < 9:
        pytest.skip("requires an NVIDIA GPU with compute capability 9 or newer")
    sharding_config = ShardingConfig.replicated((device,))

    preview_scores = jnp.array([9, 8, 7, 6, 5, 4, 3, 2, 2], dtype=jnp.bfloat16)
    preview_weights = jnp.zeros((1_041, 64), dtype=jnp.bfloat16)
    preview_weights = preview_weights.at[:9, 0].set(preview_scores)
    preview_weights = preview_weights.at[1_024:1_033, 0].set(preview_scores)
    readout_weights = jax.random.normal(jax.random.key(20), (1_041, 512), dtype=jnp.float32)
    readout_weights = (readout_weights * 0.05).astype(jnp.bfloat16)
    preview_token_ids = jnp.arange(1_041, dtype=jnp.int32)[::-1]
    matrix = LowRankPreviewReadoutMatrix(
        spec=LowRankPreviewReadoutSpec(rank=64, selection_block_size=1_024, exact_tokens_per_block=8),
        sharding_config=sharding_config,
        is_sharded=False,
        hidden_projection=FullPrecisionSpec().compress(
            jnp.zeros((64, 512), dtype=jnp.bfloat16).at[0, 0].set(1),
            sharding_config=sharding_config,
            is_sharded=False,
        ),
        preview=HybridSpec(
            quantization_spec=FullPrecisionSpec(),
            adapter_spec=None,
            incoherence_block_size=None,
        ).compress(
            preview_weights,
            sharding_config=sharding_config,
            is_sharded=False,
        ),
        weights=FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(
            readout_weights.T,
            sharding_config=sharding_config,
            is_sharded=False,
        ),
        preview_token_ids=jax.device_put(
            preview_token_ids,
            sharding_config.make_sharding((None,)),
        ),
    )
    vectors = jax.random.normal(jax.random.key(22), (8, 512), dtype=jnp.float32)
    vectors = (vectors * 0.1).astype(jnp.bfloat16)
    vectors = vectors.at[:, 0].set(jnp.arange(1, 9, dtype=jnp.bfloat16))
    vectors = jax.device_put(vectors, sharding_config.make_sharding((None, None)))

    result = call_vmapped(
        lambda vector, *, keychain: matrix.candidate_logits(vector, keychain=keychain),
        vectors,
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )
    physical_rows = jnp.concatenate((jnp.arange(8), 1_024 + jnp.arange(8)))
    reference_token_ids = preview_token_ids[physical_rows]
    reference_logits = jax.vmap(lambda vector: readout_weights[reference_token_ids] @ vector)(vectors)

    assert jnp.array_equal(
        jax.device_get(result.token_ids),
        jax.device_get(jnp.broadcast_to(reference_token_ids, result.token_ids.shape)),
    )
    assert_close(result=result.logits, reference=reference_logits, atol=2e-2, rtol=3e-2)
