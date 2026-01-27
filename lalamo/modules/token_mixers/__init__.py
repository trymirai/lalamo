from lalamo.modules.common import register_config_union

from .attention import Attention, AttentionConfig, AttentionResult
from .common import TokenMixerBase, TokenMixerResult
from .delta_net_attention import DeltaNetAttention, DeltaNetAttentionConfig, DeltaNetAttentionResult
from .convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .mamba import Mamba2, Mamba2Config, Mamba2Result
from .short_conv import ShortConv, ShortConvConfig, ShortConvResult
from .state import (
    DynamicKVCacheLayer,
    KVCacheLayer,
    Mamba2StateLayer,
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
    "AttentionResult",
    "DeltaNetAttention",
    "DeltaNetAttentionConfig",
    "DeltaNetAttentionResult",
    "DynamicKVCacheLayer",
    "KVCacheLayer",
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
    "Mamba2StateLayer",
    "SeparableCausalConv",
    "SeparableCausalConvConfig",
    "ShortConv",
    "ShortConvConfig",
    "ShortConvResult",
    "ShortConvStateLayer",
    "SSMStateLayer",
    "State",
    "StateLayerBase",
    "StaticKVCacheLayer",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]
