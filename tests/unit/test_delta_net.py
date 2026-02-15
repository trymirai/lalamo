from typing import Any

import jax
import jax.numpy as jnp

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_delta_net_attention
from lalamo.modules import (
    DeltaNetAttentionConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    SeparableCausalConvConfig,
    UpcastMode,
)
from lalamo.modules.torch_interop import torch_to_jax
from tests.common import assert_close


def _make_hf_delta_net() -> tuple[Any, Any]:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

    config = Qwen3NextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=64,
        use_cache=False,
    )
    return Qwen3NextGatedDeltaNet(config, layer_idx=0), config


def _make_lalamo_delta_net(hf_config: Any):
    precision = jnp.float32
    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=hf_config.rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    linear_config = FullPrecisionLinearConfig(precision=precision)
    config = DeltaNetAttentionConfig(
        in_proj_config=linear_config,
        conv_config=SeparableCausalConvConfig(precision=precision, has_biases=False),
        out_proj_config=linear_config,
        norm_config=norm_config,
        num_heads=hf_config.linear_num_value_heads,
        num_groups=hf_config.linear_num_key_heads,
        head_dim=hf_config.linear_key_head_dim,
        value_head_dim=hf_config.linear_value_head_dim,
        kernel_size=hf_config.linear_conv_kernel_dim,
    )
    return config.random_init(model_dim=hf_config.hidden_size, key=jax.random.PRNGKey(0))


def test_delta_net_attention_matches_hf() -> None:
    import torch

    torch.manual_seed(0)
    hf_module, hf_config = _make_hf_delta_net()
    hf_module = hf_module.eval()
    lalamo_module = _make_lalamo_delta_net(hf_config)

    weights = {
        ParameterPath("in_proj_qkvz") / "weight": torch_to_jax(hf_module.in_proj_qkvz.weight),
        ParameterPath("in_proj_ba") / "weight": torch_to_jax(hf_module.in_proj_ba.weight),
        ParameterPath("conv1d") / "weight": torch_to_jax(hf_module.conv1d.weight),
        ParameterPath("out_proj") / "weight": torch_to_jax(hf_module.out_proj.weight),
        ParameterPath("norm") / "weight": torch_to_jax(hf_module.norm.weight),
        ParameterPath("dt_bias"): torch_to_jax(hf_module.dt_bias),
        ParameterPath("A_log"): torch_to_jax(hf_module.A_log),
    }

    lalamo_module = load_delta_net_attention(
        lalamo_module,
        weights,
        ParameterPath(""),
        permute_conv=False,
    )

    inputs = torch.randn(1, 7, hf_config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        hf_out = hf_module(inputs, cache_params=None, cache_position=None, attention_mask=None)

    lalamo_out = lalamo_module(
        torch_to_jax(inputs[0]),
        positional_embeddings=None,
        state=None,
        return_updated_state=False,
        length_without_padding=None,
    ).outputs

    assert_close(
        result=lalamo_out,
        reference=torch_to_jax(hf_out[0]),
        fraction_of_allowed_violations=0.01,
    )
