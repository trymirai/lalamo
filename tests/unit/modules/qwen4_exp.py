import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from jaxtyping import Array, Float
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import (
    load_delta_net_attention,
    load_gated_residual_connection,
    load_mlp,
)
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules import (
    DeltaNetConfig,
    DenseMLPConfig,
    GatedResidualConfig,
    LinearConfig,
    MixtureOfExpertsConfig,
    NormalizationConfig,
    SiLU,
    SoftmaxRouting,
    SublayerConfig,
    TransformerLayer,
    TransformerLayerConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.utils.torch_interop import torch_to_jax
from tests.helpers import make_test_sharding_config

__all__ = [
    "HIDDEN_SIZE",
    "STREAM_COUNT",
    "as_stream",
    "float_weights",
    "load_decoder_layer",
    "make_gated_residual_config",
    "make_hf_config",
    "make_hyper_input",
    "make_initializer",
    "make_keychain",
    "merge_streams",
    "randomized",
]

HIDDEN_SIZE = 32
STREAM_COUNT = 4
EXPERT_HIDDEN_DIM = 16


def make_hf_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=HIDDEN_SIZE,
        hc_count=STREAM_COUNT,
        hc_lowrank=16,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=EXPERT_HIDDEN_DIM,
        shared_expert_intermediate_size=EXPERT_HIDDEN_DIM,
        rms_norm_eps=1e-6,
        vocab_size=128,
    )


def randomized[ModuleT: torch.nn.Module](module: ModuleT) -> ModuleT:
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return module


def float_weights(module: torch.nn.Module) -> dict[str, Array]:
    return {name: torch_to_jax(tensor) for name, tensor in module.state_dict().items() if tensor.is_floating_point()}


def make_initializer() -> EmptyInitializer:
    return EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config())


def make_keychain() -> Keychain:
    return Keychain.init(0, sharding_config=make_test_sharding_config())


def make_norm_config() -> NormalizationConfig:
    return NormalizationConfig(
        epsilon=1e-6,
        scale_offset=1.0,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )


def make_gated_residual_config() -> GatedResidualConfig:
    return GatedResidualConfig(
        stream_count=STREAM_COUNT,
        low_rank_dim=16,
        norm_config=make_norm_config(),
        linear_config=LinearConfig(),
    )


def make_hyper_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 8, STREAM_COUNT * HIDDEN_SIZE)


def as_stream(hyper_input: torch.Tensor) -> Float[Array, "batch tokens streams channels"]:
    return jax.device_put(
        rearrange(
            torch_to_jax(hyper_input),
            "batch tokens (streams channels) -> batch tokens streams channels",
            streams=STREAM_COUNT,
        ),
        make_test_sharding_config().resolve_sharding((LogicalAxis.BATCH, None, None, None)),
    )


def merge_streams(
    stream: Float[Array, "batch tokens streams channels"],
) -> Float[Array, "batch tokens streams_channels"]:
    return rearrange(stream, "batch tokens streams channels -> batch tokens (streams channels)")


def load_decoder_layer(hf_config: Qwen4ExpTextConfig, weights: dict[str, Array]) -> TransformerLayer:
    linear_config = LinearConfig()
    delta_net_config = DeltaNetConfig(
        in_proj_config=linear_config,
        conv_config=SeparableCausalConvConfig(has_biases=False),
        out_proj_config=linear_config,
        norm_config=NormalizationConfig(
            epsilon=hf_config.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        ),
        num_heads=hf_config.linear_num_value_heads,
        num_groups=hf_config.linear_num_key_heads,
        head_dim=hf_config.linear_key_head_dim,
        value_head_dim=hf_config.linear_value_head_dim,
        kernel_size=hf_config.linear_conv_kernel_dim,
    )
    moe_config = MixtureOfExpertsConfig(
        num_routed_experts=hf_config.num_experts,
        num_active_routed_experts=hf_config.num_experts_per_tok,
        routing_function=SoftmaxRouting(),
        router_config=linear_config,
        router_has_biases=False,
        expert_config=DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        ),
        gate_config=linear_config,
        num_shared_experts=1,
        expert_hidden_dim=EXPERT_HIDDEN_DIM,
    )
    initializer = make_initializer()
    residual_config = make_gated_residual_config()
    layer_config = TransformerLayerConfig(
        mixer=SublayerConfig(
            pre_norm_config=None,
            conv_config=None,
            body_config=delta_net_config,
            post_norm_config=None,
        ),
        mlp=SublayerConfig(
            pre_norm_config=None,
            conv_config=None,
            body_config=moe_config,
            post_norm_config=None,
        ),
    )
    return load_as_at(
        lambda m: (m.mixer.connection, m.mixer.body, m.mlp.connection, m.mlp.body),
        layer_config.init(initializer, residual_config, HIDDEN_SIZE, EXPERT_HIDDEN_DIM),
        (
            load_gated_residual_connection(
                residual_config.init_connection(initializer, HIDDEN_SIZE),
                weights,
                ParameterPath("attn_hyper_connection"),
            ),
            load_delta_net_attention(
                delta_net_config.init(initializer, HIDDEN_SIZE),
                weights,
                ParameterPath("linear_attn"),
                permute_conv=False,
            ),
            load_gated_residual_connection(
                residual_config.init_connection(initializer, HIDDEN_SIZE),
                weights,
                ParameterPath("mlp_hyper_connection"),
            ),
            load_mlp(
                moe_config.init(initializer, HIDDEN_SIZE, EXPERT_HIDDEN_DIM),
                weights,
                ParameterPath("mlp"),
                "up_proj",
                "gate_proj",
                "down_proj",
            ),
        ),
    )
