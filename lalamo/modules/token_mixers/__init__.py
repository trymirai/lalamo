from lalamo.modules.common import register_config_union
from lalamo.modules.forward_pass_config import (
    AttentionForwardPassConfig,
    AttentionImplementation,
    MixerForwardPassConfig,
)

from .attention import Attention, AttentionConfig, AttentionResult
from .common import TokenMixerBase, TokenMixerResult
from .convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .delta_net_attention import DeltaNetAttention, DeltaNetAttentionConfig, DeltaNetAttentionResult
from .mamba import Mamba2, Mamba2Config, Mamba2Result
from .short_conv import ShortConv, ShortConvConfig, ShortConvResult
from .state import (
    DynamicKVCacheLayer,
    KVCacheLayer,
    ShortConvStateLayer,
    SSMStateLayer,
    State,
    StateLayerBase,
    StaticKVCacheLayer,
)

TokenMixerConfig = AttentionConfig | DeltaNetAttentionConfig | Mamba2Config | ShortConvConfig

register_config_union(TokenMixerConfig)

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionForwardPassConfig",
    "AttentionImplementation",
    "AttentionResult",
    "DeltaNetAttention",
    "DeltaNetAttentionConfig",
    "DeltaNetAttentionResult",
    "DynamicKVCacheLayer",
    "KVCacheLayer",
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
    "MixerForwardPassConfig",
    "SSMStateLayer",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
    "ShortConv",
    "ShortConvConfig",
    "ShortConvResult",
    "ShortConvStateLayer",
    "State",
    "StateLayerBase",
    "StaticKVCacheLayer",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]
