from lalamo.modules.common import register_config_union

from .attention import Attention, AttentionConfig, AttentionResult
from .common import TokenMixerBase, TokenMixerResult
from .mamba import Mamba2, Mamba2Config, Mamba2Result, SeparableCausalConv, SeparableCausalConvConfig
from .short_conv import ShortConv, ShortConvConfig, ShortConvResult
from .state import (
    DynamicKVCacheLayer,
    KVCacheLayer,
    Mamba2StateLayer,
    ShortConvStateLayer,
    State,
    StateLayerBase,
    StaticKVCacheLayer,
)

TokenMixerConfig = AttentionConfig | Mamba2Config | ShortConvConfig

register_config_union(TokenMixerConfig)

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
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
    "State",
    "StateLayerBase",
    "StaticKVCacheLayer",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]
