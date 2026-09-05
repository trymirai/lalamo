import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import cached_property
from re import Pattern
from typing import NotRequired, TypedDict

from frozendict import frozendict
from jinja2 import Template
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from lalamo.token_codec import TokenCodec, TokenCodecConfig

__all__ = [
    "AssistantMessage",
    "ChatCodec",
    "ChatCodecConfig",
    "ContentBlock",
    "Image",
    "Message",
    "ReasoningConfig",
    "ReasoningEffort",
    "SystemMessage",
    "ToolSchema",
    "UserMessage",
]


type ToolSchema = None  # WIP
type Image = None  # WIP


class ReasoningEffort(StrEnum):
    XHIGH = "xhigh"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_REASONING = "no_reasoning"


@dataclass(frozen=True)
class ReasoningConfig:
    default_reasoning_effort: ReasoningEffort
    field_name: str
    # Jinja's `is true`/`is false` tests reject string spellings, while other template fields expect strings.
    reasoning_effort_to_field_value: frozendict[ReasoningEffort, str | bool]

    def __post_init__(self) -> None:
        if self.default_reasoning_effort not in self.reasoning_effort_to_field_value:
            raise ValueError("The default reasoning effort must have a field value.")

    def field_value(self, effort: ReasoningEffort | None) -> str | bool:
        if effort is None:
            effort = self.default_reasoning_effort
        if effort not in self.reasoning_effort_to_field_value:
            raise ValueError(
                f"Reasoning effort {effort.value!r} is not supported by this model; "
                f"supported efforts: {self.reasoning_effort_to_field_value}."
            )
        return self.reasoning_effort_to_field_value[effort]


def _strftime_now(format_string: str) -> str:
    return datetime.now().strftime(format_string)  # noqa: DTZ005


class HuggingFaceMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[dict]]
    reasoning_content: NotRequired[str]


class HuggingFaceRequest(TypedDict):
    add_generation_prompt: bool
    bos_token: str | None
    eos_token: str | None
    messages: list[HuggingFaceMessage]
    tools: NotRequired[dict]


@dataclass(frozen=True)
class Message:
    pass


type ContentBlock = str | Image


@dataclass(frozen=True)
class UserMessage(Message):
    content: tuple[ContentBlock, ...] | ContentBlock


@dataclass(frozen=True)
class SystemMessage(UserMessage):
    content: tuple[ContentBlock, ...] | ContentBlock


@dataclass(frozen=True)
class AssistantMessage(Message):
    chain_of_thought: str | None
    response: str


@dataclass(frozen=True)
class ChatCodecConfig(TokenCodecConfig):
    prompt_template: str
    output_parser_regex: str | None
    system_role_name: str
    user_role_name: str
    assistant_role_name: str
    eos_token: str | None
    bos_token: str | None
    end_of_thinking_tag: str | None = None
    default_system_prompt: str | None = None
    reasoning_config: ReasoningConfig | None = None

    def init(self, tokenizer: Tokenizer) -> "ChatCodec":
        return ChatCodec(
            config=self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class ChatCodec(TokenCodec[Iterable[Message], AssistantMessage, ChatCodecConfig]):
    @cached_property
    def prompt_template(self) -> Template:
        return Template(self.config.prompt_template)

    @cached_property
    def output_parser_regex(self) -> Pattern | None:
        if self.config.output_parser_regex is None:
            return None
        return re.compile(self.config.output_parser_regex)

    def message_to_dict(self, message: Message) -> HuggingFaceMessage:
        match message:
            case SystemMessage(content=content):
                assert isinstance(content, str)
                return HuggingFaceMessage(role=self.config.system_role_name, content=content)
            case UserMessage(content=content):
                assert isinstance(content, str)
                return HuggingFaceMessage(role=self.config.user_role_name, content=content)
            case AssistantMessage(chain_of_thought=chain_of_thought, response=response):
                result = HuggingFaceMessage(role=self.config.assistant_role_name, content=response)
                if chain_of_thought:
                    result["reasoning_content"] = chain_of_thought
                return result
        raise ValueError(f"Unsupported message type: {type(message)}")

    def request_to_dict(
        self,
        messages: Iterable[Message],
        tools: Iterable[ToolSchema] | None = None,
    ) -> HuggingFaceRequest:
        converted_messages = [self.message_to_dict(message) for message in messages]
        if self.config.default_system_prompt is not None:  # noqa: SIM102
            if not converted_messages or converted_messages[0]["role"] != self.config.system_role_name:
                converted_messages = [
                    HuggingFaceMessage(role=self.config.system_role_name, content=self.config.default_system_prompt),
                    *converted_messages,
                ]
        result = HuggingFaceRequest(
            add_generation_prompt=True,
            messages=converted_messages,
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
        )
        if tools is not None:
            raise NotImplementedError("Tools are not supported yet.")
        return result

    def render_request(
        self,
        messages: Iterable[Message],
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        template_context: dict[str, object] = {
            **self.request_to_dict(messages),
            "strftime_now": _strftime_now,
        }

        reasoning_config = self.config.reasoning_config
        if reasoning_config is None and reasoning_effort is not None:
            raise ValueError("This model does not support configurable reasoning effort.")
        if reasoning_config is not None:
            template_context[reasoning_config.field_name] = reasoning_config.field_value(reasoning_effort)

        return self.prompt_template.render(template_context)

    def encode_request(
        self,
        request: Iterable[Message],
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return self.encode_text(self.render_request(request, reasoning_effort=reasoning_effort))

    def parse_response(self, response: str) -> AssistantMessage:
        if self.output_parser_regex is None:
            return AssistantMessage(chain_of_thought=None, response=response)
        match = self.output_parser_regex.match(response)
        if match is None:
            return AssistantMessage(chain_of_thought=None, response=response)
        groups = match.groupdict()
        for key in groups:
            if groups[key] is None and AssistantMessage.__dataclass_fields__[key].type is str:
                groups[key] = ""
        return AssistantMessage(**groups)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_tokens(self, tokens: list[int], *, hide_invalid_utf_chars: bool = False) -> str:
        if hide_invalid_utf_chars:
            return DecodeStream(skip_special_tokens=False).step(self.tokenizer, tokens) or ""
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def decode_stream(self, reasoning_effort: ReasoningEffort | None = None) -> "ChatDecodeStream":
        reasoning_config = self.config.reasoning_config
        response_started = self.output_parser_regex is None
        if reasoning_config is not None:
            effective_effort = reasoning_effort or reasoning_config.default_reasoning_effort
            response_started = response_started or effective_effort is ReasoningEffort.NO_REASONING
        return ChatDecodeStream(self, response_started=response_started)

    def decode_response(self, response: list[int]) -> AssistantMessage:
        return self.parse_response(self.decode_tokens(response))

    def __post_init__(self) -> None:
        if self.output_parser_regex is not None:
            all_fields = AssistantMessage.__dataclass_fields__
            # NOTE: str type annotations are assumed to be required
            required_fields = {
                k: v for k, v in all_fields.items() if isinstance(v.type, str) or v.type != (v.type | None)
            }
            named_groups = self.output_parser_regex.groupindex
            invalid_groups = set(named_groups) - set(all_fields)
            if invalid_groups:
                raise ValueError(f"Unsupported output fields: {list(invalid_groups)}")
            for group_name in required_fields:
                if group_name not in named_groups:
                    raise ValueError(f"Missing required output field: {group_name}")


@dataclass
class ChatDecodeStream:
    codec: ChatCodec
    response_started: bool
    decoder: DecodeStream = field(default_factory=DecodeStream)
    raw_response: str = ""
    reasoning: str = ""
    response: str = ""

    def step(self, token_id: int) -> tuple[str, str]:
        """Returns the newly visible reasoning and response text."""
        piece = self.decoder.step(self.codec.tokenizer, token_id) or ""
        self.raw_response += piece
        reasoning = ""
        if not self.response_started:
            end_of_thinking_tag = self.codec.config.end_of_thinking_tag
            if end_of_thinking_tag is None:
                return "", ""
            tag_position = self.raw_response.find(end_of_thinking_tag)
            if tag_position < 0:
                # Hold back a suffix that may still grow into the end-of-thinking tag.
                held_back = max(
                    (
                        n
                        for n in range(1, len(end_of_thinking_tag))
                        if self.raw_response.endswith(end_of_thinking_tag[:n])
                    ),
                    default=0,
                )
                return self._take_reasoning(len(self.raw_response) - held_back), ""
            self.response_started = True
            reasoning = self._take_reasoning(tag_position)
        # Until the first visible character the parser may still strip leading whitespace or opening tags.
        if not self.response:
            piece = self.codec.parse_response(self.raw_response).response
        self.response += piece
        return reasoning, piece

    def _take_reasoning(self, end: int) -> str:
        piece = self.raw_response[len(self.reasoning) : end]
        self.reasoning += piece
        return piece

    def finish(self) -> AssistantMessage:
        if not self.response_started:
            return AssistantMessage(chain_of_thought=self.raw_response, response="")
        return self.codec.parse_response(self.raw_response)
