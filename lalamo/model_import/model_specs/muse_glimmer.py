from frozendict import frozendict

from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import ReasoningConfig, ReasoningEffort, ToolCallFormat
from lalamo.models.language_model import GenerationConfig

__all__ = ["MUSE_GLIMMER_MODELS"]

MUSE_GLIMMER_OUTPUT_PARSER_REGEX = (
    r"(?s)\s*(?:to=self<\|message\|>(?P<chain_of_thought>.*?)"
    r"(?:<\|eom\|><\|start\|>assistant |\Z))?"
    r"(?:to=user<\|message\|>(?P<response>.*?))?"
    r"(?:<\|eot\|>|<\|end_of_text\|>)?\Z"
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
        tool_call_format=ToolCallFormat.MUSE_ATEM,
        end_of_thinking_tag="<|eom|><|start|>assistant to=user<|message|>",
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.HIGH,
            reasoning_effort_to_template_fields=frozendict(
                {
                    ReasoningEffort.LOW: frozendict(reasoning_strength="low"),
                    ReasoningEffort.MEDIUM: frozendict(reasoning_strength="medium"),
                    ReasoningEffort.HIGH: frozendict(reasoning_strength="high"),
                    ReasoningEffort.XHIGH: frozendict(reasoning_strength="xhigh"),
                }
            ),
        ),
    ),
]
