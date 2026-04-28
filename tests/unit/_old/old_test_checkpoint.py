import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lalamo.checkpoint_manager import CheckpointManager
from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig


def _tiny_decoder_config() -> DecoderConfig:
    precision = jnp.float32
    norm = NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    linear = FullPrecisionLinearConfig(precision=precision)
    layer = TransformerLayerConfig(
        pre_mixer_norm_config=norm,
        mixer_config=AttentionConfig(
            qkv_projection_config=linear,
            out_projection_config=linear,
            query_norm_config=None,
            key_norm_config=None,
            num_heads=1,
            num_groups=1,
            head_dim=8,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
        ),
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm,
        mlp_config=DenseMLPConfig(
            linear_config=linear,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
        ),
        post_mlp_norm_config=None,
    )
    return DecoderConfig(
        embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
        transformer_config=TransformerConfig(
            global_rope_config=UnscaledRoPEConfig(
                base=10000.0,
                max_sequence_length=16,
            ),
            local_rope_config=None,
            layer_configs=(layer,),
            output_norm_config=norm,
            model_dim=8,
            hidden_dim=16,
            context_length=16,
        ),
        vocab_size=32,
    )


@pytest.mark.fast
def test_checkpoint_save_restore() -> None:
    config = _tiny_decoder_config()
    model = config.init(RandomInitializer(precision=jnp.float32, key=jax.random.key(0)))

    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(directory=Path(tmp))
        manager.save("test", model)
        restored = manager.restore("test")

    original_arrays = eqx.filter(model, eqx.is_array)
    restored_arrays = eqx.filter(restored, eqx.is_array)

    assert bool(eqx.tree_equal(original_arrays, restored_arrays))


@pytest.mark.fast
def test_checkpoint_restore_preserves_saved_dtypes() -> None:
    config = _tiny_decoder_config()
    model = config.init(RandomInitializer(precision=jnp.bfloat16, key=jax.random.key(0)))

    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(directory=Path(tmp))
        manager.save("test", model)
        restored = manager.restore("test")

    original_arrays = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    restored_arrays = jax.tree.leaves(eqx.filter(restored, eqx.is_array))

    assert [array.dtype for array in restored_arrays] == [array.dtype for array in original_arrays]
