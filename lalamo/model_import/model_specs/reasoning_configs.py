from frozendict import frozendict

from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort

__all__ = ["BOOLEAN_REASONING_DEFAULT_OFF_CONFIG", "BOOLEAN_REASONING_DEFAULT_ON_CONFIG"]

BOOLEAN_REASONING_DEFAULT_ON_CONFIG = ReasoningConfig(
    default_reasoning_effort=ReasoningEffort.MEDIUM,
    field_name="enable_thinking",
    reasoning_effort_to_field_value=frozendict(
        {
            ReasoningEffort.MEDIUM: True,
            ReasoningEffort.NO_REASONING: False,
        }
    ),
)

BOOLEAN_REASONING_DEFAULT_OFF_CONFIG = ReasoningConfig(
    default_reasoning_effort=ReasoningEffort.NO_REASONING,
    field_name="enable_thinking",
    reasoning_effort_to_field_value=frozendict(
        {
            ReasoningEffort.MEDIUM: True,
            ReasoningEffort.NO_REASONING: False,
        }
    ),
)
