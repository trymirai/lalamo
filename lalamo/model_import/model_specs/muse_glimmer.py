import re
from typing import ClassVar

from frozendict import frozendict

from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parsers import parse_xml_tool_calls
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import (
    AssistantMessage,
    ReasoningConfig,
    ReasoningEffort,
    ResponseParser,
    ToolCall,
    ToolSchema,
)
from lalamo.models.language_model import GenerationConfig

__all__ = ["MUSE_GLIMMER_MODELS"]


class MuseResponseParser(ResponseParser):
    tool_call_tags: ClassVar[tuple[str, str] | None] = ("<atem:function_calls>", "</atem:function_calls>")

    turn = re.compile(
        r"\s*to=([^\s<]+)<\|message\|>(.*?)(?:<\|eom\|><\|start\|>assistant|<\|eot\|>|<\|end_of_text\|>|\Z)",
        re.DOTALL,
    )

    @classmethod
    def parse(cls, response: str, *, prompt: str = "", tools: tuple[ToolSchema, ...] = ()) -> AssistantMessage:  # noqa: ARG003
        reasoning = content = ""
        calls: tuple[ToolCall, ...] = ()
        position = 0
        while turn := cls.turn.match(response, position):
            recipient, body = turn.groups()
            position = turn.end()
            if recipient == "self":
                reasoning += body
                continue
            if recipient != "user" and tools:
                message = super().parse(body, tools=tools)
                if message.tool_calls and not message.response.strip():
                    calls += message.tool_calls
                    continue
            content += body
        if response[position:].strip():
            content += response[position:]
        return AssistantMessage(reasoning or None, content, calls)

    @classmethod
    def parse_tool_calls(cls, body: str, tools: tuple[ToolSchema, ...]) -> tuple[ToolCall, ...]:
        return parse_xml_tool_calls(
            body,
            r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>',
            r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>',
            tools,
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
        response_parser=MuseResponseParser,
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
