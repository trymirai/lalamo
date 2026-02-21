import subprocess
import sys
import textwrap

import pytest


SHARDING_TEST_SCRIPT = textwrap.dedent("""\
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import NamedSharding

from lalamo.modules import (
    GELU,
    AttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    Sharding,
    TransformerLayerConfig,
    UpcastMode,
    get_default_sharding_config,
    init_parallelism,
)

TENSOR_AXIS_SIZE = 2
init_parallelism(tensor_parallelism=TENSOR_AXIS_SIZE)


def make_transformer_layer():
    precision = jnp.float32
    linear_cfg = FullPrecisionLinearConfig(precision=precision)
    norm_cfg = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    attention_cfg = AttentionConfig(
        qkv_projection_config=linear_cfg,
        out_projection_config=linear_cfg,
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
        use_rope=False,
    )
    mlp_cfg = DenseMLPConfig(
        linear_config=linear_cfg,
        activation=GELU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    layer_cfg = TransformerLayerConfig(
        pre_mixer_norm_config=norm_cfg,
        mixer_config=attention_cfg,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm_cfg,
        mlp_config=mlp_cfg,
        post_mlp_norm_config=None,
    )
    return layer_cfg.random_init(model_dim=16, hidden_dim=32, key=jax.random.PRNGKey(0))


def is_sharded_along(array, axis_name):
    if not hasattr(array, "sharding"):
        return False
    sharding = array.sharding
    if not isinstance(sharding, NamedSharding):
        return False
    return any(p == axis_name for p in sharding.spec)


def test_weights_are_sharded():
    transformer = make_transformer_layer()
    for path, leaf in jax.tree_util.tree_leaves_with_path(transformer):
        if not isinstance(leaf, jax.Array):
            continue
        path_str = jax.tree_util.keystr(path)
        if leaf.ndim >= 2 and "weights" in path_str.lower():
            assert is_sharded_along(leaf, "tensor"), (
                f"Weight {path_str} shape={leaf.shape} not sharded along tensor"
            )
    print("PASS: test_weights_are_sharded")


def test_forward_pass():
    transformer = make_transformer_layer()
    batch, seq_len, model_dim = 2, 4, 16
    x = jax.random.normal(jax.random.key(1), (batch, seq_len, model_dim), dtype=jnp.float32)
    result = eqx.filter_jit(transformer)(x, positional_embeddings=None)
    assert result.outputs.shape == (batch, seq_len, model_dim)
    assert not jnp.any(jnp.isnan(result.outputs))
    print("PASS: test_forward_pass")


def test_import_preserves_sharding():
    transformer = make_transformer_layer()
    exported = transformer.export_weights()
    reimported = transformer.import_weights(exported)
    for path, leaf in jax.tree_util.tree_leaves_with_path(reimported):
        if not isinstance(leaf, jax.Array):
            continue
        path_str = jax.tree_util.keystr(path)
        if leaf.ndim >= 2 and "weights" in path_str.lower():
            assert is_sharded_along(leaf, "tensor"), (
                f"Reimported {path_str} shape={leaf.shape} not sharded along tensor"
            )
    print("PASS: test_import_preserves_sharding")


def test_invalid_parallelism_raises_concise_value_error():
    expected = (
        "Wrong sharding axes: use positive data/tensor parallelism with "
        "data_parallelism * tensor_parallelism == number of devices."
    )
    invalid_configs = [
        dict(tensor_parallelism=0),
        dict(tensor_parallelism=3),
        dict(data_parallelism=3),
        dict(data_parallelism=2, tensor_parallelism=3),
    ]
    for kwargs in invalid_configs:
        try:
            Sharding(**kwargs).resolve()
        except ValueError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
    print("PASS: test_invalid_parallelism_raises_concise_value_error")


def test_custom_axis_names_are_detected():
    init_parallelism(tensor_parallelism=2, data_parallelism=4, data_axis_name="dp", tensor_axis_name="tp")
    sharding_config = get_default_sharding_config()
    assert sharding_config is not None
    assert sharding_config.data_axis_name == "dp"
    assert sharding_config.tensor_axis_name == "tp"
    print("PASS: test_custom_axis_names_are_detected")


test_weights_are_sharded()
test_forward_pass()
test_import_preserves_sharding()
test_invalid_parallelism_raises_concise_value_error()
test_custom_axis_names_are_detected()
print("ALL TESTS PASSED")
""")


def test_sharding():
    result = subprocess.run(
        [sys.executable, "-c", SHARDING_TEST_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"Sharding test failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "ALL TESTS PASSED" in result.stdout, f"Unexpected output:\n{result.stdout}"
