import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules.flow import Context, FlowForwardPassConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.ngram import NGramConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_ngram_chunked_padding_and_eos_history(backend: str) -> None:
    sharding = ShardingConfig.replicated(jax.devices(backend)[:1])
    config = NGramConfig(
        vocab_size=32,
        seed=1234,
        eos_token_id=31,
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size_base=37,
        vocab_size_divisor=8,
        embedding_dim=16,
        kernel_size=4,
        norm_config=NormalizationConfig(1e-6, 1.0, UpcastMode.FULL_LAYER, subtract_mean=False),
        layer_index=0,
        linear_config=LinearConfig(),
        streams=4,
    )
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
    ):
        injection = config.init(RandomInitializer(jnp.float32, sharding, key=jax.random.key(7)), 8)
        tokens = jnp.array([[1, 2, 31, 4, 5, 6, 7, 8, 9, 31, 11, 12, 13], [7, 31, 9, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        streams = jax.random.normal(jax.random.key(8), (2, 13, 32))
        lengths = jnp.array([13, 10])
        keychain = Keychain.init(9, sharding_config=sharding)
        positions = jnp.broadcast_to(jnp.arange(13, dtype=jnp.int32), tokens.shape)

        def context(start: int, end: int) -> Context:
            return Context(
                tokens[:, start:end],
                positions[:, start:end],
                jnp.clip(lengths - start, 0, end - start),
                None,
                FlowForwardPassConfig.for_tracer_tests(),
                sharding,
            )

        full, full_state = injection.forward(streams, state=None, ctx=context(0, 13), keychain=keychain)
        state = None
        pieces = []
        for start, end in ((0, 1), (1, 4), (4, 6), (6, 13)):
            outputs, state = injection.forward(
                streams[:, start:end],
                state=state,
                ctx=context(start, end),
                keychain=keychain,
            )
            pieces.append(outputs)
        for row, length in enumerate(lengths.tolist()):
            np.testing.assert_allclose(
                jnp.concatenate(pieces, axis=1)[row, :length], full[row, :length], atol=3e-6, rtol=3e-5
            )
        for actual, expected in zip(jax.tree.leaves(state), jax.tree.leaves(full_state), strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        # The first token after EOS hashes as a new sequence, even when EOS was in a previous chunk.
        np.testing.assert_array_equal(injection.hash_ids(tokens[:1, 3:4]), injection.hash_ids(tokens[:1])[:, 3:4])
