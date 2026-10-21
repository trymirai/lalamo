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
from typing import Literal, NoReturn, NotRequired, TypedDict, cast, get_origin

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
    "THINKING_RESPONSE_FORMAT",
    "AssistantMessage",
    "ChatCodec",
    "ChatCodecConfig",
    "FunctionCall",
    "Message",
    "PythonToolCalls",
    "ReasoningConfig",
    "ReasoningEffort",
    "ResponseFormat",
    "SystemMessage",
    "TaggedToolCalls",
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


@dataclass(frozen=True)
class AssistantMessage:
    chain_of_thought: str | None = None
    response: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class PythonToolCalls:
    tags: tuple[str, str]


@dataclass(frozen=True)
class TaggedToolCalls:
    tags: tuple[str, str]
    function_pattern: str
    parameter_pattern: str

    def __post_init__(self) -> None:
        for pattern in (self.function_pattern, self.parameter_pattern):
            if re.compile(pattern, re.DOTALL).groups != 2:
                raise ValueError("Tagged tool patterns must capture a name and a body.")


@dataclass(frozen=True)
class ResponseFormat:
    reasoning_pattern: str | None = None
    prefixes: tuple[str, ...] = ()
    turn_pattern: str | None = None
    tool_calls: PythonToolCalls | TaggedToolCalls | None = None

    def __post_init__(self) -> None:
        if self.turn_pattern is not None and (self.reasoning_pattern is not None or self.prefixes):
            raise ValueError("Turn framing cannot be combined with a reasoning split or prompt prefixes.")
        if self.prefixes and self.reasoning_pattern is None:
            raise ValueError("Prompt prefixes require a reasoning pattern.")
        for pattern in (self.reasoning_pattern, self.turn_pattern):
            if pattern is None:
                continue
            groups = re.compile(pattern, re.DOTALL).groupindex.keys()
            expected = {"chain_of_thought", "response"}
            if pattern == self.turn_pattern:
                expected.add("tool_calls")
                if self.tool_calls is not None and "tool_calls" not in groups:
                    raise ValueError("A turn format with tool support must capture tool_calls.")
            if not groups or not groups <= expected:
                raise ValueError(f"Response patterns must use named captures from {expected}.")
            if self.prefixes and "chain_of_thought" not in groups:
                raise ValueError("Prompt prefixes require a chain_of_thought capture.")


THINKING_RESPONSE_FORMAT = ResponseFormat(
    reasoning_pattern=(
        r"(?:</think>|(?:(?:<think>|(?=.*</think>))(?P<chain_of_thought>.*?)(?:</think>\s*|\Z)))?"
        r"(?P<response>.*)"
    ),
    prefixes=("<think>", "</think>"),
)


class _JsonLiterals(ast.NodeTransformer):
    # Python-style calls can contain JSON literals in nested arguments.
    def visit_Name(self, node: ast.Name) -> ast.Constant:
        literals = {"true": True, "false": False, "null": None}
        if node.id not in literals:
            raise ValueError(f"Unexpected name {node.id!r} in a tool argument.")
        return ast.Constant(literals[node.id])


def _parse_python_tool_calls(body: str, tools: tuple[ToolSchema, ...]) -> tuple[ToolCall, ...]:
    names = (cast("str", cast("dict[str, JSON]", tool["function"])["name"]) for tool in tools)
    # Python identifiers cannot contain the hyphens OpenAI allows in tool names. Replace only call positions,
    # outside string literals, with non-ASCII identifiers that cannot collide with ASCII OpenAI names.
    aliases = {name: f"tool_\u03b1{index}" for index, name in enumerate(names) if not name.isidentifier()}
    if aliases:
        hyphenated = "|".join(re.escape(name) for name in aliases)
        body = re.sub(
            rf"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?<![\w.-])({hyphenated})(?=\s*\()""",
            lambda match: match[1] or aliases[match[2]],
            body,
        )
    names_by_alias = {alias: name for name, alias in aliases.items()}
    expression = ast.parse(body.strip(), mode="eval").body
    expressions = [expression]
    if isinstance(expression, ast.List):
        expressions = expression.elts
    if not expressions:
        raise ValueError("Malformed tool call.")
    calls = []
    for call in expressions:
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.args
            or any(keyword.arg is None for keyword in call.keywords)
            or len({keyword.arg for keyword in call.keywords}) != len(call.keywords)
        ):
            raise ValueError("Malformed tool call.")
        for node in ast.walk(call):
            if isinstance(node, (ast.Set, ast.Tuple)) or (
                isinstance(node, ast.Dict)
                and any(not isinstance(key, ast.Constant) or not isinstance(key.value, str) for key in node.keys)
            ):
                raise ValueError("Tool arguments must be JSON values.")
        arguments = {
            cast("str", keyword.arg): ast.literal_eval(_JsonLiterals().visit(keyword.value))
            for keyword in call.keywords
        }
        try:
            json.dumps(arguments, allow_nan=False)
        except TypeError as error:
            raise ValueError("Tool arguments must be JSON values.") from error
        calls.append(
            ToolCall(
                type="function",
                function=FunctionCall(name=names_by_alias.get(call.func.id, call.func.id), arguments=arguments),
            )
        )
    return tuple(calls)


def _parse_tagged_tool_calls(
    body: str, tool_format: TaggedToolCalls, tools: tuple[ToolSchema, ...]
) -> tuple[ToolCall, ...]:
    function_pattern = tool_format.function_pattern
    parameter_pattern = tool_format.parameter_pattern
    if re.fullmatch(rf"(?:\s*{function_pattern})+\s*", body, re.DOTALL) is None:
        raise ValueError("Malformed tool call.")
    declared_types: dict[tuple[str, str], JSON] = {}
    for tool in tools:
        function = cast("dict[str, JSON]", tool["function"])
        parameters = cast("dict[str, JSON]", function.get("parameters") or {})
        properties = cast("dict[str, JSON]", parameters.get("properties") or {})
        declared_types.update(
            ((cast("str", function["name"]), name), schema.get("type"))
            for name, schema in properties.items()
            if isinstance(schema, dict)
        )
    calls: list[ToolCall] = []
    for name, parameters_body in re.findall(function_pattern, body, re.DOTALL):
        if re.fullmatch(rf"(?:\s*{parameter_pattern})*\s*", parameters_body, re.DOTALL) is None:
            raise ValueError("Malformed tool parameters.")
        arguments: dict[str, JSON] = {}
        for parameter, value in re.findall(parameter_pattern, parameters_body, re.DOTALL):
            if parameter in arguments:
                raise ValueError("Duplicate tool parameter.")
            declared = declared_types.get((name, parameter))
            argument: JSON = value
            # Tagged formats emit strings verbatim; decode only parameters declared without a string type.
            if declared and declared != "string" and not (isinstance(declared, list) and "string" in declared):
                try:
                    argument = json.loads(value)
                except json.JSONDecodeError:
                    argument = {"True": True, "False": False, "None": None}.get(value.strip(), value)
            arguments[parameter] = argument
        calls.append(ToolCall(type="function", function=FunctionCall(name=name, arguments=arguments)))
    return tuple(calls)


def _extract_tool_calls(
    response: str, tool_format: PythonToolCalls | TaggedToolCalls | None, tools: tuple[ToolSchema, ...]
) -> tuple[str, tuple[ToolCall, ...]]:
    if tool_format is None or not tools:
        return response, ()
    calls: list[ToolCall] = []

    def extract(match: re.Match[str]) -> str:
        try:
            if isinstance(tool_format, TaggedToolCalls):
                calls.extend(_parse_tagged_tool_calls(match[1], tool_format, tools))
            else:
                calls.extend(_parse_python_tool_calls(match[1], tools))
        except (SyntaxError, ValueError):
            return match[0]
        return ""

    opening, closing = tool_format.tags
    return (
        re.sub(re.escape(opening) + "(.*?)" + re.escape(closing), extract, response, flags=re.DOTALL),
        tuple(calls),
    )


# Byte-level BPE vocabularies spell printable bytes as themselves and the other bytes as characters from U+0100.
_PRINTABLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
_BYTE_LEVEL_CHARACTERS = {chr(byte): byte for byte in _PRINTABLE_BYTES} | {
    chr(256 + index): byte for index, byte in enumerate(sorted(set(range(256)) - set(_PRINTABLE_BYTES)))
}


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
    response_format: ResponseFormat | None
    system_role_name: str
    user_role_name: str
    assistant_role_name: str
    eos_token: str | None
    bos_token: str | None
    default_system_prompt: str | None = None
    reasoning_config: ReasoningConfig | None = None

    def init(self, tokenizer: Tokenizer) -> "ChatCodec":
        return ChatCodec(
            config=self,
            tokenizer=tokenizer,
        )


@dataclass(frozen=True)
class ChatCodec(TokenCodec[ChatCodecConfig]):
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

    def render_request(
        self,
        messages: Iterable[Message],
        *,
        tools: Iterable[ToolSchema] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        converted_messages: list[HuggingFaceMessage] = []
        for message in messages:
            converted: HuggingFaceMessage = message_converter.unstructure(message, Message)
            converted["role"] = getattr(self.config, f"{converted['role']}_role_name", converted["role"])
            converted_messages.append(converted)
        if self.config.default_system_prompt is not None:  # noqa: SIM102
            if not converted_messages or converted_messages[0]["role"] != self.config.system_role_name:
                converted_messages = [
                    HuggingFaceMessage(role=self.config.system_role_name, content=self.config.default_system_prompt),
                    *converted_messages,
                ]
        template_context: dict[str, object] = {
            "add_generation_prompt": True,
            "messages": converted_messages,
            "bos_token": self.config.bos_token,
            "eos_token": self.config.eos_token,
            "tools": None if tools is None else list(tools),
            "strftime_now": _strftime_now,
            "raise_exception": _raise_template_error,
        }

        reasoning_config = self.config.reasoning_config
        if reasoning_config is None and reasoning_effort is not None:
            raise ValueError("This model does not support configurable reasoning effort.")
        if reasoning_config is not None:
            template_context[reasoning_config.field_name] = reasoning_config.field_value(reasoning_effort)

        return self.prompt_template.render(template_context)

    def parse_response(self, response: str, *, prompt: str, tools: Iterable[ToolSchema] = ()) -> AssistantMessage:
        response_format = self.config.response_format
        if response_format is None:
            return AssistantMessage(response=response)
        tools = tuple(tools)
        if response_format.turn_pattern is not None:
            reasoning = content = ""
            calls: tuple[ToolCall, ...] = ()
            position = 0
            pattern = re.compile(response_format.turn_pattern, re.DOTALL)
            while turn := pattern.match(response, position):
                if turn.end() == position:
                    raise ValueError("A turn pattern must consume text.")
                fields = turn.groupdict()
                position = turn.end()
                reasoning += fields.get("chain_of_thought") or ""
                content += fields.get("response") or ""
                if (body := fields.get("tool_calls")) is not None:
                    remaining, tool_calls = _extract_tool_calls(body, response_format.tool_calls, tools)
                    if tool_calls and not remaining.strip():
                        calls += tool_calls
                    else:
                        content += body
            if response[position:].strip():
                content += response[position:]
            return AssistantMessage(reasoning or None, content, calls)
        reasoning = None
        if response_format.reasoning_pattern is not None:
            prefix = next(
                (marker for marker in response_format.prefixes if prompt.rstrip().endswith(marker.rstrip())), ""
            )
            if prefix and response.startswith(prefix):
                prefix_match = re.fullmatch(response_format.reasoning_pattern, prefix, re.DOTALL)
                if prefix_match is not None and prefix_match["chain_of_thought"] is not None:
                    prefix = ""
            match = re.fullmatch(response_format.reasoning_pattern, prefix + response, re.DOTALL)
            if match is not None:
                fields = match.groupdict()
                reasoning = fields.get("chain_of_thought") or None
                response = fields.get("response") or ""
        response, calls = _extract_tool_calls(response, response_format.tool_calls, tools)
        return AssistantMessage(reasoning, response, calls)

    def decode_tokens(self, tokens: list[int]) -> str:
        return "".join(codecs.iterdecode(self._token_groups(tokens), "utf-8", errors="replace"))

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
