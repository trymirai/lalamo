import ast
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
from re._parser import LITERAL, SubPattern, parse  # type: ignore[missing-import]
from typing import Any, Literal, NoReturn, NotRequired, TypedDict, cast, get_origin

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
    "ToolCallFormat",
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


_MUSE_TURN_ENDS = ("<|eom|><|start|>assistant", "<|eot|>", "<|end_of_text|>")
# Byte-level BPE vocabularies spell printable bytes as themselves and the other bytes as characters from U+0100.
_PRINTABLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
_BYTE_LEVEL_CHARACTERS = {chr(byte): byte for byte in _PRINTABLE_BYTES} | {
    chr(256 + index): byte for index, byte in enumerate(sorted(set(range(256)) - set(_PRINTABLE_BYTES)))
}
_MUSE_TURN = re.compile(
    r"\s*to=([^\s<]+)<\|message\|>(.*?)(?:" + "|".join(map(re.escape, _MUSE_TURN_ENDS)) + r"|\Z)", re.DOTALL
)


class ToolCallFormat(StrEnum):
    QWEN_XML = "qwen_xml"
    LIQUID = "liquid"
    MUSE_ATEM = "muse_atem"

    @property
    def tags(self) -> tuple[str, str]:
        match self:
            case ToolCallFormat.QWEN_XML:
                return "<tool_call>", "</tool_call>"
            case ToolCallFormat.LIQUID:
                return "<|tool_call_start|>", "<|tool_call_end|>"
            case ToolCallFormat.MUSE_ATEM:
                return "<atem:function_calls>", "</atem:function_calls>"

    def parse_calls(self, body: str, tools: tuple["ToolSchema", ...]) -> tuple[ToolCall, ...]:
        """Parses the calls between this format's tags, raising ValueError or SyntaxError if they are malformed."""
        functions = [cast("dict[str, Any]", tool["function"]) for tool in tools]
        match self:
            case ToolCallFormat.QWEN_XML:
                return _xml_tool_calls(
                    body,
                    r"<function=([^>\n]+)>(.*?)</function>",
                    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
                    functions,
                )
            case ToolCallFormat.LIQUID:
                return _liquid_tool_calls(body, functions)
            case ToolCallFormat.MUSE_ATEM:
                return _xml_tool_calls(
                    body,
                    r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>',
                    r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>',
                    functions,
                )


def _xml_tool_calls(
    body: str, function_pattern: str, parameter_pattern: str, functions: list[dict[str, Any]]
) -> tuple[ToolCall, ...]:
    if re.fullmatch(rf"(?:\s*{function_pattern})+\s*", body, re.DOTALL) is None:
        raise ValueError("Malformed tool call.")
    # These formats write strings verbatim, so only parameters declared without a string type are decoded.
    declared_types = {
        (function["name"], name): schema.get("type") if isinstance(schema, dict) else None
        for function in functions
        for name, schema in ((function.get("parameters") or {}).get("properties") or {}).items()
    }

    def decode(function: str, parameter: str, value: str) -> JSON:
        declared = declared_types.get((function, parameter))
        if not declared or "string" in ([declared] if isinstance(declared, str) else declared):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"True": True, "False": False, "None": None}.get(value.strip(), value)

    return tuple(
        ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments={
                    parameter: decode(name, parameter, value)
                    for parameter, value in re.findall(parameter_pattern, arguments, re.DOTALL)
                },
            ),
        )
        for name, arguments in re.findall(function_pattern, body, re.DOTALL)
    )


class _JsonLiterals(ast.NodeTransformer):
    """Liquid writes nested values as JSON, whose literals read as Python names."""

    def visit_Name(self, node: ast.Name) -> ast.Constant:
        literals = {"true": True, "false": False, "null": None}
        if node.id not in literals:
            raise ValueError(f"Unexpected name {node.id!r} in a tool argument.")
        return ast.Constant(literals[node.id])


def _liquid_tool_calls(body: str, functions: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    # Liquid calls are Python expressions, which cannot contain the hyphens OpenAI allows in names. Outside string
    # literals, such names are replaced by non-ASCII identifiers, which cannot collide with ASCII OpenAI names.
    aliases = {function["name"]: f"tool_\u03b1{index}" for index, function in enumerate(functions)}
    hyphenated = "|".join(re.escape(name) for name in aliases if not name.isidentifier())
    if hyphenated:
        body = re.sub(
            rf"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?<![\w.-])({hyphenated})(?=\s*\()""",
            lambda match: match[1] or aliases[match[2]],
            body,
        )
    names = {alias: name for name, alias in aliases.items()}
    expression = ast.parse(body.strip(), mode="eval").body
    parsed = []
    for call in expression.elts if isinstance(expression, ast.List) else [expression]:
        if (
            not isinstance(call, ast.Call)
            or call.args
            or any(keyword.arg is None for keyword in call.keywords)
            or len({keyword.arg for keyword in call.keywords}) != len(call.keywords)
        ):
            raise ValueError("Malformed tool call.")
        name = ast.unparse(call.func)
        arguments = {
            cast("str", keyword.arg): ast.literal_eval(_JsonLiterals().visit(keyword.value))
            for keyword in call.keywords
        }
        parsed.append(
            ToolCall(type="function", function=FunctionCall(name=names.get(name, name), arguments=arguments))
        )
    return tuple(parsed)


def _regex_literal_runs(pattern: SubPattern) -> Iterable[str]:
    # Python's regex parser is private; it is used only to find literal delimiters in output parser regexes.
    literals = ""
    for operation, argument in pattern:
        if operation is LITERAL:
            literals += chr(argument)
            continue
        if literals:
            yield literals
            literals = ""
        for child in argument if isinstance(argument, tuple) else (argument,):
            if isinstance(child, SubPattern):
                yield from _regex_literal_runs(child)
            elif isinstance(child, list):
                for branch in child:
                    if isinstance(branch, SubPattern):
                        yield from _regex_literal_runs(branch)
    if literals:
        yield literals


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


def _raise_template_error(message: str) -> NoReturn:
    raise ValueError(message)


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
    tools: list[ToolSchema] | None


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
    tool_call_format: ToolCallFormat | None = None

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
        return HuggingFaceRequest(
            add_generation_prompt=True,
            messages=converted_messages,
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
            tools=None if tools is None else list(tools),
        )

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
            "raise_exception": _raise_template_error,
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

    @cached_property
    def output_markers(self) -> tuple[str, ...]:
        if self.config.output_parser_regex is None:
            return ()
        return tuple(
            literal for literal in _regex_literal_runs(parse(self.config.output_parser_regex)) if "<" in literal
        )

    def parse_response(self, response: str, *, prompt: str = "", tools: Iterable[ToolSchema] = ()) -> AssistantMessage:
        tools = tuple(tools)
        tool_call_format = self.config.tool_call_format
        if tool_call_format is ToolCallFormat.MUSE_ATEM:
            return self._parse_muse_turns(response, tools)
        regex = self.output_parser_regex
        chain_of_thought = None
        if regex is not None:
            # The rendered prompt may already open the assistant's reasoning channel.
            tails = (prompt[prompt.rfind(marker) :] for marker in self.output_markers if marker in prompt)
            prefix = max(
                (
                    tail
                    for tail in tails
                    if (match := regex.fullmatch(tail)) is not None
                    and not any((text or "").strip() for text in match.groupdict().values())
                ),
                key=len,
                default="",
            )
            text = prefix + response
            match = regex.fullmatch(text)
            if match is not None:
                channels = {}
                for name in ("chain_of_thought", "response"):
                    if name not in regex.groupindex:
                        continue
                    start, end = match.span(name)
                    if start >= 0 and end > len(prefix):
                        channels[name] = text[max(start, len(prefix)) : end]
                chain_of_thought = channels.get("chain_of_thought")
                response = channels.get("response") or ""
        if not tools or tool_call_format is None:
            return AssistantMessage(chain_of_thought, response)
        response, tool_calls = self._split_tool_blocks(response, tools)
        return AssistantMessage(chain_of_thought, response, tool_calls)

    def _split_tool_blocks(self, text: str, tools: tuple[ToolSchema, ...]) -> tuple[str, tuple[ToolCall, ...]]:
        """Removes the well-formed tool-call blocks from `text`; malformed ones remain text."""
        assert self.config.tool_call_format is not None
        opening_tag, closing_tag = self.config.tool_call_format.tags
        segments = re.split(f"({re.escape(opening_tag)}.*?{re.escape(closing_tag)})", text, flags=re.DOTALL)
        content = ""
        tool_calls: tuple[ToolCall, ...] = ()
        for index, segment in enumerate(segments):
            if index % 2:
                try:
                    tool_calls += self.config.tool_call_format.parse_calls(
                        segment[len(opening_tag) : -len(closing_tag)], tools
                    )
                    continue
                except (SyntaxError, ValueError):
                    pass
            content += segment
        return content, tool_calls

    def _parse_muse_turns(self, generated: str, tools: tuple[ToolSchema, ...]) -> AssistantMessage:
        """Muse addresses each assistant turn to `self`, `user`, or a tool, which receives native calls."""
        reasoning = response = ""
        tool_calls: tuple[ToolCall, ...] = ()
        position = 0
        while turn := _MUSE_TURN.match(generated, position):
            recipient, body = turn.groups()
            position = turn.end()
            if recipient == "self":
                reasoning += body
            elif recipient == "user" or not tools:
                response += body
            else:
                remainder, calls = self._split_tool_blocks(body, tools)
                if calls and not remainder.strip():
                    tool_calls += calls
                else:
                    response += body
        if generated[position:].strip():
            response += generated[position:]
        return AssistantMessage(reasoning or None, response, tool_calls)

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

    def decode_token_bytes(self, token_id: int) -> bytes:
        """The bytes of one token, including partial UTF-8 sequences that decoding to text would replace."""
        if token_id in self._byte_token_ids:
            return bytes((self._byte_token_ids[token_id],))
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        token = self.tokenizer.id_to_token(token_id)
        if "\ufffd" in text and token is not None and all(char in _BYTE_LEVEL_CHARACTERS for char in token):
            return bytes(_BYTE_LEVEL_CHARACTERS[char] for char in token)
        return text.encode()

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
