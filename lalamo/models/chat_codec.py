import codecs
import itertools
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import cached_property, partial
from re import Pattern
from typing import Literal, NotRequired, TypedDict, cast, get_origin

import cattrs
from cattrs.cols import homogenous_tuple_structure_factory, mapping_structure_factory
from cattrs.dispatch import StructureHook
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override
from cattrs.preconf.json import make_converter
from cattrs.strategies import configure_tagged_union
from frozendict import frozendict
from jinja2 import Environment, Template
from tokenizers import Tokenizer

from lalamo.token_codec import TokenCodec, TokenCodecConfig
from lalamo.utils.json import JSON

__all__ = [
    "AssistantMessage",
    "ChatCodec",
    "ChatCodecConfig",
    "Message",
    "ReasoningConfig",
    "ReasoningEffort",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "ToolSchema",
    "UserMessage",
    "message_converter",
    "parse_hf_message",
]


type ToolSchema = dict[str, JSON]


class FunctionCall(TypedDict):
    name: str
    arguments: dict[str, JSON]


class ToolCall(TypedDict):
    type: Literal["function"]
    function: FunctionCall
    id: NotRequired[str]
    index: NotRequired[int]


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
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]
    thinking: NotRequired[str]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]


class HuggingFaceRequest(TypedDict):
    add_generation_prompt: bool
    bos_token: str | None
    eos_token: str | None
    messages: list[HuggingFaceMessage]
    tools: NotRequired[list[ToolSchema]]


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class SystemMessage(UserMessage):
    pass


@dataclass(frozen=True)
class ToolMessage:
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class AssistantMessage:
    chain_of_thought: str | None = None
    response: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


type Message = UserMessage | SystemMessage | AssistantMessage | ToolMessage


message_converter = make_converter(
    forbid_extra_keys=True, omit_if_default=True, unstruct_collection_overrides={tuple: list}
)
message_converter.register_structure_hook(JSON, lambda value, _: value)
message_converter.register_unstructure_hook(JSON, lambda value: value)


@message_converter.register_structure_hook
def _structure_text(value: JSON, _: type[str]) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected text.")
    return value


@message_converter.register_structure_hook_factory(lambda cls: get_origin(cls) in (dict, tuple))
def _structure_json_collection(cls: object) -> StructureHook:
    # Arrow JSON columns and function arguments can arrive as encoded collections.
    # Runtime type expressions are not classes; beartype cannot annotate GenericAlias.
    if get_origin(cls) is dict:
        structure: StructureHook = mapping_structure_factory(cast("type", cls), message_converter)
    else:
        structure = homogenous_tuple_structure_factory(cast("type", cls), message_converter)

    def decode(value: JSON, _: object) -> dict | tuple:
        if isinstance(value, str):
            value = json.loads(value)
        return structure(value, cls)

    return decode


_assistant_fields = {
    "response": override(rename="content", omit_if_default=False),
    "chain_of_thought": override(rename="reasoning_content"),
}
message_converter.register_structure_hook(
    AssistantMessage, make_dict_structure_fn(AssistantMessage, message_converter, **_assistant_fields)
)
message_converter.register_unstructure_hook(
    AssistantMessage,
    make_dict_unstructure_fn(AssistantMessage, message_converter, _cattrs_omit_if_default=True, **_assistant_fields),
)
configure_tagged_union(
    Message,
    message_converter,
    tag_name="role",
    tag_generator=lambda cls: cls.__name__.removesuffix("Message").lower(),
)
_structure_message = message_converter.get_structure_hook(Message)


def parse_hf_message(obj: dict) -> Message:
    obj = {key: value for key, value in obj.items() if value is not None}
    obj["role"] = {"human": "user", "developer": "system"}.get(obj["role"], obj["role"])
    for alias in ("reasoning", "thinking"):
        if alias in obj and "reasoning_content" not in obj:
            obj["reasoning_content"] = obj.pop(alias)
    return _structure_message(obj, Message)


message_converter.register_structure_hook(Message, lambda obj, _: parse_hf_message(obj))


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
    def _byte_token_ids(self) -> dict[int, int]:
        return {
            token_id: byte_value
            for byte_value in range(256)
            if (token_id := self.tokenizer.token_to_id(f"<0x{byte_value:02X}>")) is not None
        }

    @cached_property
    def prompt_template(self) -> Template:
        environment = Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        # Hugging Face templates emit JSON, without Jinja's HTML escaping or key sorting.
        environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)
        return environment.from_string(self.config.prompt_template)

    @cached_property
    def output_parser_regex(self) -> Pattern | None:
        if self.config.output_parser_regex is None:
            return None
        return re.compile(self.config.output_parser_regex)

    def message_to_dict(self, message: Message) -> HuggingFaceMessage:
        result: HuggingFaceMessage = message_converter.unstructure(message, Message)
        result["role"] = getattr(self.config, f"{result['role']}_role_name", result["role"])
        return result

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
            result["tools"] = list(tools)
        return result

    def render_request(
        self,
        messages: Iterable[Message],
        *,
        tools: Iterable[ToolSchema] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        template_context: dict[str, object] = {
            **self.request_to_dict(messages, tools),
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
        tools: Iterable[ToolSchema] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return self.encode_text(self.render_request(request, tools=tools, reasoning_effort=reasoning_effort))

    def parse_response(self, response: str) -> AssistantMessage:
        if self.output_parser_regex is None:
            return AssistantMessage(response=response)
        match = self.output_parser_regex.match(response)
        if match is None:
            return AssistantMessage(response=response)
        return cattrs.structure(
            {name: value for name, value in match.groupdict().items() if value is not None}, AssistantMessage
        )

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_tokens(self, tokens: list[int], *, hide_invalid_utf_chars: bool = False) -> str:
        errors = "ignore" if hide_invalid_utf_chars else "replace"
        return "".join(codecs.iterdecode(self._token_groups(tokens), "utf-8", errors=errors))

    def _token_groups(self, tokens: list[int]) -> Iterable[bytes]:
        byte_token_ids = self._byte_token_ids
        for is_byte, group in itertools.groupby(tokens, key=lambda tid: tid in byte_token_ids):
            if is_byte:
                yield bytes(byte_token_ids[tid] for tid in group)
            else:
                yield self.tokenizer.decode(list(group), skip_special_tokens=False).encode("utf-8")

    def decode_response(self, response: list[int]) -> AssistantMessage:
        return self.parse_response(self.decode_tokens(response))

    def __post_init__(self) -> None:
        if self.output_parser_regex is not None:
            text_fields = {
                name: field
                for name, field in AssistantMessage.__dataclass_fields__.items()
                if field.type in (str, str | None)
            }
            named_groups = self.output_parser_regex.groupindex
            invalid_groups = set(named_groups) - text_fields.keys()
            if invalid_groups:
                raise ValueError(f"Unsupported output fields: {list(invalid_groups)}")
            for name, field in text_fields.items():
                if field.type is str and name not in named_groups:
                    raise ValueError(f"Missing required output field: {name}")
