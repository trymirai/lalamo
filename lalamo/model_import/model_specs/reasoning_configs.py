from frozendict import frozendict

from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort

__all__ = ["BOOLEAN_REASONING_DEFAULT_OFF_CONFIG", "BOOLEAN_REASONING_DEFAULT_ON_CONFIG"]

BOOLEAN_REASONING_DEFAULT_ON_CONFIG = ReasoningConfig(
    default_reasoning_effort=ReasoningEffort.MEDIUM,
    reasoning_effort_to_template_fields=frozendict(
        {
            ReasoningEffort.MEDIUM: frozendict(enable_thinking=True),
            ReasoningEffort.NO_REASONING: frozendict(enable_thinking=False),
        }
    ),
)

BOOLEAN_REASONING_DEFAULT_OFF_CONFIG = ReasoningConfig(
    default_reasoning_effort=ReasoningEffort.NO_REASONING,
    reasoning_effort_to_template_fields=frozendict(
        {
            ReasoningEffort.MEDIUM: frozendict(enable_thinking=True),
            ReasoningEffort.NO_REASONING: frozendict(enable_thinking=False),
        }
    ),
)
