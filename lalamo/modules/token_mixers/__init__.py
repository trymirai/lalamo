from lalamo.modules.common import register_config_union

from .attention import Attention, AttentionConfig, AttentionResult
from .common import TokenMixerBase, TokenMixerResult
from .mamba import Mamba2, Mamba2Config, Mamba2Result

TokenMixerConfig = AttentionConfig | Mamba2Config

register_config_union(TokenMixerConfig)

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
    "Mamba2",
    "Mamba2Config",
    "Mamba2Result",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]
