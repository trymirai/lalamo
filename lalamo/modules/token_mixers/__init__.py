from lalamo.modules.common import DummyUnionMember, register_config_union

from .attention import Attention, AttentionConfig, AttentionResult
from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult

TokenMixerConfig = TokenMixerConfigBase | DummyUnionMember

register_config_union(TokenMixerConfig)

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
]
