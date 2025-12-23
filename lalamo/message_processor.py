import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from re import Pattern
from typing import NotRequired, TypedDict

from jinja2 import Template
from tokenizers import Tokenizer

__all__ = [
    "AssistantMessage",
    "ContentBlock",
    "Image",
    "Message",
    "MessageProcessor",
    "MessageProcessorConfig",
    "SystemMessage",
    "ToolSchema",
    "UserMessage",
]

type ToolSchema = None  # WIP
type Image = None  # WIP


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
    messages: list[HuggingFaceMessage]
    enable_thinking: NotRequired[bool]
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
class MessageProcessorConfig:
    prompt_template: str
    output_parser_regex: str | None
    system_role_name: str
    user_role_name: str
    assistant_role_name: str
    bos_token: str | None

    def init(self, tokenizer: Tokenizer) -> "MessageProcessor":
        return MessageProcessor(
            config=self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class MessageProcessor:
    config: MessageProcessorConfig
    tokenizer: Tokenizer

    @cached_property
    def prompt_template(self) -> Template:
        return Template(self.config.prompt_template)

    @cached_property
    def output_parser_regex(self) -> Pattern | None:
        if self.config.output_parser_regex is None:
            return None
        return re.compile(self.config.output_parser_regex)

    @property
    def system_role_name(self) -> str:
        return self.config.system_role_name

    @property
    def user_role_name(self) -> str:
        return self.config.user_role_name

    @property
    def assistant_role_name(self) -> str:
        return self.config.assistant_role_name

    @property
    def bos_token(self) -> str | None:
        return self.config.bos_token

    def message_to_dict(self, message: Message) -> HuggingFaceMessage:
        match message:
            case UserMessage(content=content):
                assert isinstance(content, str)
                return HuggingFaceMessage(role=self.user_role_name, content=content)
            case SystemMessage(content=content):
                assert isinstance(content, str)
                return HuggingFaceMessage(role=self.system_role_name, content=content)
            case AssistantMessage(chain_of_thought=chain_of_thought, response=response):
                result = HuggingFaceMessage(role=self.assistant_role_name, content=response)
                if chain_of_thought:
                    result["reasoning_content"] = chain_of_thought
                return result
        raise ValueError(f"Unsupported message type: {type(message)}")

    def request_to_dict(
        self,
        messages: Iterable[Message],
        tools: Iterable[ToolSchema] | None = None,
        enable_thinking: bool | None = None,
    ) -> HuggingFaceRequest:
        converted_messages = [self.message_to_dict(message) for message in messages]
        result = HuggingFaceRequest(add_generation_prompt=True, messages=converted_messages, bos_token=self.bos_token)
        if enable_thinking is not None:
            result["enable_thinking"] = enable_thinking
        if tools is not None:
            raise NotImplementedError("Tools are not supported yet.")
        return result

    def render_request(self, messages: Iterable[Message]) -> str:
        request_dict = self.request_to_dict(messages)
        return self.prompt_template.render({**request_dict, "strftime_now": _strftime_now})

    def parse_response(self, response: str) -> AssistantMessage:
        if self.output_parser_regex is None:
            return AssistantMessage(chain_of_thought=None, response=response)
        match = self.output_parser_regex.match(response)
        if match is None:
            raise ValueError(f"Invalid response format: {response}")
        return AssistantMessage(**match.groupdict())

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def tokenize_request(self, messages: Iterable[Message]) -> list[int]:
        rendered = self.render_request(messages)
        return self.tokenize_text(rendered)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def __post_init__(self) -> None:
        if self.output_parser_regex is not None:
            all_fields = AssistantMessage.__dataclass_fields__
            # NOTE: str type annotations are assumed to be required
            required_fields = {
                k: v for k, v in all_fields.items() if isinstance(v.type, str) or v.type == (v.type | None)
            }
            named_groups = self.output_parser_regex.groupindex
            invalid_groups = set(named_groups) - set(all_fields)
            if invalid_groups:
                raise ValueError(f"Unsupported output fields: {list(invalid_groups)}")
            for group_name in required_fields:
                if group_name not in named_groups:
                    raise ValueError(f"Missing required output field: {group_name}")
