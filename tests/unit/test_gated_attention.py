import jax
import jax.numpy as jnp

from lalamo.modules import AttentionConfig, FullPrecisionLinearConfig


def test_gated_attention_export_import_roundtrip() -> None:
    config = AttentionConfig(
        qkv_projection_config=FullPrecisionLinearConfig(precision=jnp.float32),
        out_projection_config=FullPrecisionLinearConfig(precision=jnp.float32),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=8,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        gate_projection_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    model_dim = 16
    attn = config.random_init(model_dim=model_dim, key=jax.random.key(0))
    target = config.random_init(model_dim=model_dim, key=jax.random.key(1))
    imported = target.import_weights(attn.export_weights())
    assert imported.gate_projection is not None
    assert jnp.array_equal(attn.gate_projection.weights, imported.gate_projection.weights)
