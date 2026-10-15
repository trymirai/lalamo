from frozendict import frozendict

from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort
from lalamo.models.language_model import GenerationConfig

__all__ = ["MUSE_GLIMMER_MODELS"]

MUSE_GLIMMER_OUTPUT_PARSER_REGEX = (
    r"(?s)\s*(?:to=self<\|message\|>(?P<chain_of_thought>.*?)"
    r"<\|eom\|><\|start\|>assistant )?to=user<\|message\|>"
    r"(?P<response>.*?)(?:<\|eot\|>|<\|end_of_text\|>)\Z"
)

MUSE_GLIMMER_MODELS = [
    LanguageModelSpec(
        vendor="Meta",
        family="Muse-Glimmer",
        name="Muse-Glimmer-30B",
        size="30B",
        origin=HuggingFaceOrigin(repo="meta-models/Muse-Glimmer-30B"),
        config_type=HFMuseGlimmerConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
            generation_params_overrides=GenerationConfig(
                temperature=1.0,
                top_k=64,
                top_p=0.95,
            ),
        ),
        output_parser_regex=MUSE_GLIMMER_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="<|eom|><|start|>assistant to=user<|message|>",
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.HIGH,
            field_name="reasoning_strength",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.MEDIUM: "medium",
                    ReasoningEffort.HIGH: "high",
                    ReasoningEffort.XHIGH: "xhigh",
                }
            ),
        ),
    ),
]
