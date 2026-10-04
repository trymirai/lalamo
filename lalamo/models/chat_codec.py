import ast
import codecs
import itertools
import json
import math
import re
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from functools import cached_property, partial
from re import Pattern
from re._parser import LITERAL, SubPattern, parse  # type: ignore[missing-import]
from typing import TYPE_CHECKING, Literal, NoReturn, NotRequired, TypedDict, cast, get_origin
from urllib.parse import unquote

from cattrs.cols import homogenous_tuple_structure_factory, mapping_structure_factory
from cattrs.dispatch import StructureHook
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override
from cattrs.preconf.json import make_converter
from cattrs.strategies import configure_tagged_union
from frozendict import frozendict
from jinja2 import Environment, Template
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from lalamo.token_codec import TokenCodec, TokenCodecConfig
from lalamo.utils.json import JSON

if TYPE_CHECKING:
    import xgrammar

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


_MUSE_ASSISTANT_HEADER = "<|start|>assistant"
_MUSE_RECIPIENT_HEADER = re.compile(r"\s*to=([^\s<]+)<\|message\|>")
_MUSE_TERMINATORS = ("<|eom|>", "<|eot|>", "<|end_of_text|>")
_MAX_FORMATTING_WHITESPACE = 16
_PYTHON_LAYOUT = r"(?:[ \t\f\r\n]|#[^\r\n]*(?:\r\n?|\n|$)|\\(?:\r\n?|\n))*"
_LIQUID_NAME = r"""[^\s()\[\]{},=#\\<>|"']+"""


@dataclass(frozen=True)
class _GeneratedToolCall:
    name: str
    arguments: str
    source_end: int | None = None

    def at_offset(self, offset: int) -> "_GeneratedToolCall":
        if self.source_end is None:
            return self
        return replace(self, source_end=offset + self.source_end)

    def to_tool_call(self) -> ToolCall:
        arguments = json.loads(
            self.arguments,
            parse_float=_finite_tool_number,
            parse_constant=_finite_tool_number,
            object_pairs_hook=_distinct_tool_object,
        )
        if not isinstance(arguments, dict):
            raise TypeError("Tool-call arguments must be a JSON object.")
        return ToolCall(type="function", function=FunctionCall(name=self.name, arguments=arguments))


def _distinct_tool_object(pairs: list[tuple[str, JSON]]) -> dict[str, JSON]:
    arguments: dict[str, JSON] = {}
    for key, value in pairs:
        if key in arguments:
            raise ValueError("Duplicate tool-call parameter.")
        arguments[key] = value
    return arguments


class ToolCallFormat(StrEnum):
    QWEN_XML = "qwen_xml"
    LIQUID = "liquid"
    MUSE_ATEM = "muse_atem"

    @property
    def opening_tag(self) -> str:
        match self:
            case ToolCallFormat.QWEN_XML:
                return "<tool_call>"
            case ToolCallFormat.LIQUID:
                return "<|tool_call_start|>"
            case ToolCallFormat.MUSE_ATEM:
                return "<atem:function_calls>"

    @property
    def closing_tag(self) -> str:
        match self:
            case ToolCallFormat.QWEN_XML:
                return "</tool_call>"
            case ToolCallFormat.LIQUID:
                return "<|tool_call_end|>"
            case ToolCallFormat.MUSE_ATEM:
                return "</atem:function_calls>"

    def parse_calls(
        self, body: str, tools: tuple[ToolSchema, ...], *, parallel_tool_calls: bool | None = None
    ) -> tuple[tuple[_GeneratedToolCall, ...], int | None]:
        if self is ToolCallFormat.LIQUID:
            calls: list[_GeneratedToolCall] = []
            opening = re.match(_PYTHON_LAYOUT + r"\[", body)
            if opening is None:
                return (), None
            position = opening.end()
            while position < len(body):
                closing = re.match(
                    _PYTHON_LAYOUT + r"]" + _PYTHON_LAYOUT + rf"(?={re.escape(self.closing_tag)}|$)",
                    body[position:],
                )
                if closing is not None:
                    return tuple(calls), position + closing.end()
                function = re.match(_PYTHON_LAYOUT + f"({_LIQUID_NAME})" + _PYTHON_LAYOUT + r"\(", body[position:])
                if function is None:
                    break
                name = function[1]
                position += function.end()
                arguments, consumed = _liquid_arguments(body[position:], _tool_function(name, tools))
                source_end = None
                if consumed is not None:
                    source_end = position + consumed
                calls.append(_GeneratedToolCall(name, arguments, source_end))
                if consumed is None:
                    return tuple(calls), None
                position += consumed
                if parallel_tool_calls is False:
                    return tuple(calls), position
                comma = re.match(_PYTHON_LAYOUT + r",", body[position:])
                if comma is not None:
                    position += comma.end()
                elif not re.match(_PYTHON_LAYOUT + r"]", body[position:]):
                    return tuple(calls), None
            return tuple(calls), None

        if self is ToolCallFormat.QWEN_XML:
            function_pattern = r"\s*<function=([^>\s]+)>"
            function_end = "</function>"
            parameter_pattern = r"\s*<parameter=([^>\s]+)>"
            parameter_end = "</parameter>"
        else:
            function_pattern = r'\s*<atem:invoke name="([^"\s]+)">'
            function_end = "</atem:invoke>"
            parameter_pattern = r'\s*<atem:parameter name="([^"\s]+)">'
            parameter_end = "</atem:parameter>"
        calls = []
        position = 0
        while position < len(body):
            ending = re.match(r"\s*" + re.escape(self.closing_tag), body[position:])
            if ending is not None:
                return tuple(calls), position + ending.start() + len(ending[0]) - len(self.closing_tag)
            function = re.match(function_pattern, body[position:])
            if function is None:
                if calls and not body[position:].strip():
                    return tuple(calls), len(body)
                return tuple(calls), None
            name = function[1]
            definition = _tool_function(name, tools)
            position += function.end()
            strict = definition is not None and definition.get("strict") is True
            end = -1 if strict else body.find(function_end, position)
            source_end: int | None = None
            if end >= 0:
                source_end = end + len(function_end)
            parameters = body[position:end] if end >= 0 else body[position:]
            if not strict and end < 0:
                parameters = _hold_marker_prefix(parameters, (function_end,))
            arguments = "{"
            parameter_position = 0
            while parameters[parameter_position:].strip():
                if strict:
                    closing = re.match(r"\s*" + re.escape(function_end), parameters[parameter_position:])
                    if closing is not None:
                        end = position + parameter_position + closing.end() - len(function_end)
                        source_end = end + len(function_end)
                        break
                parameter = re.match(parameter_pattern, parameters[parameter_position:])
                if parameter is None:
                    source_end = None
                    arguments += _hold_marker_prefix(
                        parameters[parameter_position:], ("<parameter=", '<atem:parameter name="')
                    )
                    break
                parameter_name = parameter[1]
                parameter_position += parameter.end()
                value_end = parameters.find(parameter_end, parameter_position)
                value = parameters[parameter_position:value_end] if value_end >= 0 else parameters[parameter_position:]
                if strict:
                    value_end = -1
                    value = parameters[parameter_position:]
                    lexical_end = _json_value_end(value)
                    if lexical_end is not None:
                        tail = value[lexical_end:].lstrip()
                        if tail.startswith(parameter_end):
                            value_end = parameter_position + len(value) - len(tail)
                            value = parameters[parameter_position:value_end]
                        elif parameter_end.startswith(tail):
                            value = value[:lexical_end]
                elif value_end < 0:
                    value = _hold_marker_prefix(value, (parameter_end,))
                if self is ToolCallFormat.QWEN_XML and not strict:
                    value = value.removeprefix("\n")
                    if value_end >= 0:
                        value = value.removesuffix("\n")
                if arguments != "{":
                    arguments += ","
                arguments += (
                    json.dumps(parameter_name)
                    + ":"
                    + _tool_parameter(value, name, parameter_name, tools, complete=value_end >= 0)
                )
                if value_end < 0:
                    source_end = None
                    break
                parameter_position = value_end + len(parameter_end)
            if end >= 0:
                arguments += "}"
            calls.append(_GeneratedToolCall(name, arguments, source_end))
            if parallel_tool_calls is False and source_end is not None:
                return tuple(calls), source_end
            if source_end is None:
                return tuple(calls), None
            position = source_end
        if not calls:
            return (), None
        return tuple(calls), position


def _python_scalar(text: str, *, complete: bool) -> str:
    for python, encoded in (("True", "true"), ("False", "false"), ("None", "null")):
        if text == python or (not complete and python.startswith(text)):
            return encoded[: len(text)]
    return text


def _python_string(text: str) -> tuple[str, int]:
    prefix = 0
    if text[0] in "rRuU":
        prefix = 1
    delimiter = text[prefix]
    if text.startswith(delimiter * 3, prefix):
        delimiter *= 3
    position = prefix + len(delimiter)
    while position < len(text):
        if text.startswith(delimiter, position):
            end = position + len(delimiter)
            try:
                value = ast.literal_eval(text[:end])
            except (SyntaxError, ValueError):
                return text[:end], end
            return json.dumps(value), end
        if text[position] == "\\":
            position += 1
        position += 1
    # A partial Python string is still output, including an unfinished escape; never supply its closing quote.
    value = text[prefix + len(delimiter) :]
    if prefix and text[0] in "rR":
        return json.dumps(value)[:-1], len(text)
    encoded = '"'
    position = 0
    escape_pattern = r'\\(?:[abfnrtv\\"\x27\n]|[0-7]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{[^}]+\})'
    while position < len(value):
        if value[position] != "\\":
            encoded += json.dumps(value[position])[1:-1]
            position += 1
            continue
        escape = re.match(escape_pattern, value[position:])
        if escape is None:
            encoded += value[position:]
            break
        try:
            decoded = ast.literal_eval('"' + escape[0] + '"')
        except (SyntaxError, ValueError):
            encoded += escape[0]
        else:
            encoded += json.dumps(decoded)[1:-1]
        position += escape.end()
    return encoded, len(text)


def _liquid_arguments(text: str, function: dict[str, JSON] | None = None) -> tuple[str, int | None]:
    arguments = "{"
    position = 0
    delimiters: list[tuple[str, int]] = []
    parameter = True
    while position < len(text):
        character = text[position]
        if character in " \t\f\r\n":
            if not parameter:
                arguments += character
            position += 1
            continue
        if character == "#" or text.startswith(("\\\n", "\\\r"), position):
            layout = re.match(_PYTHON_LAYOUT, text[position:])
            assert layout is not None
            position += layout.end()
            if not parameter:
                arguments += "\n"
            continue
        if character == ")" and not delimiters:
            return arguments.rstrip().removesuffix(",") + "}", position + 1
        if parameter and not delimiters:
            parameter_name = re.match(f"({_LIQUID_NAME})" + _PYTHON_LAYOUT + r"=", text[position:])
            if parameter_name is None:
                return arguments + text[position:], None
            arguments += json.dumps(parameter_name[1]) + ":"
            position += parameter_name.end()
            parameter = False
            continue
        if function is not None and function.get("strict") is True and not parameter and not delimiters:
            value_end = _json_value_end(text[position:])
            if value_end is not None:
                arguments += text[position : position + value_end]
                position += value_end
                continue
            if character in '"[{':
                return arguments + text[position:], None
        if character in "\"'" or (character in "rRuU" and text[position + 1 : position + 2] in ("'", '"')):
            encoded, consumed = _python_string(text[position:])
            position += consumed
            while adjacency := re.match(_PYTHON_LAYOUT + r"(?=[rRuU]?[\"'])", text[position:]):
                position += adjacency.end()
                following, consumed = _python_string(text[position:])
                encoded = encoded.removesuffix('"') + following[1:]
                position += consumed
            arguments += encoded
            continue
        if character in "[{(":
            delimiters.append(({"[": "]", "{": "}", "(": ")"}[character], position))
            if character == "(":
                arguments += " "
                position += 1
                continue
        elif character in "]})" and delimiters and character == delimiters[-1][0]:
            _, start = delimiters.pop()
            if arguments.rstrip().endswith(",") and arguments.rstrip()[:-1].rstrip().endswith(("[", "{")):
                return arguments + text[position:], None
            arguments = arguments.rstrip().removesuffix(",")
            if character == ")":
                if not text[start + 1 : position].strip():
                    return arguments + text[position:], None
                arguments += " "
                position += 1
                continue
        elif character == "," and delimiters and delimiters[-1][0] == ")":
            return arguments + text[position:], None
        elif character == "," and not delimiters:
            parameter = True
        sign = re.match(r"[-+][ \t\f\r\n]*(?=[0-9.(])", text[position:])
        if sign is not None:
            arguments += sign[0][0]
            position += sign.end()
            continue
        atom = re.match(r"[^\s,\[\]{}():\"'=#\\]+", text[position:])
        if atom is not None:
            token = _python_scalar(atom[0], complete=position + atom.end() < len(text))
            if re.fullmatch(r"-?(?:[0-9][0-9a-fA-F_xXoObB.]*|\.[0-9]+)(?:[eE][-+]?[0-9_]*)?", token):
                with suppress(SyntaxError, ValueError):
                    token = json.dumps(ast.literal_eval(token))
                    preceding = arguments.rstrip()
                    if preceding.endswith("+") and preceding[:-1].rstrip().endswith((":", "[", ",", "{")):
                        arguments = preceding[:-1]
                    elif preceding.endswith("-"):
                        arguments = preceding
            arguments += token
            position += atom.end()
        else:
            arguments += character
            position += 1
    return arguments, None


def _tool_function(name: str, tools: Iterable[ToolSchema]) -> dict[str, JSON] | None:
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return function
    return None


def _tool_parameter(
    value: str, function: str, parameter: str, tools: tuple[ToolSchema, ...], *, complete: bool
) -> str:
    types: frozenset[str] | None = None
    definition = _tool_function(function, tools)
    if definition is not None:
        if definition.get("strict") is True:
            return value
        parameters = definition.get("parameters")
        if isinstance(parameters, dict):
            types = _parameter_types(parameters, parameters).get((parameter,))
    if types is None or "string" in types:
        encoded = json.dumps(value)
        if not complete:
            return encoded[:-1]
        return encoded
    # Qwen3.5 emits Python spellings for scalar booleans/null; newer Qwen and Muse emit JSON spellings.
    return _python_scalar(value.strip(), complete=complete)


def _finite_tool_number(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("Tool-call numbers must be finite.")
    return number


def _parameter_types(
    schema: JSON, document: dict[str, JSON], seen: frozenset[str] = frozenset()
) -> dict[tuple[()] | tuple[str], frozenset[str] | None]:
    # None is unconstrained; an empty set is a contradictory type intersection.
    if schema is False:
        return {(): frozenset()}
    if not isinstance(schema, dict):
        return {(): None}
    declared = schema.get("type")
    types: frozenset[str] | None = None
    if isinstance(declared, str):
        types = frozenset((declared,))
    elif isinstance(declared, list):
        types = frozenset(kind for kind in declared if isinstance(kind, str))
    # JSON Schema's number type includes integers.
    if types is not None and "number" in types:
        types |= {"integer"}
    direct: dict[tuple[()] | tuple[str], frozenset[str] | None] = {(): types}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, parameter_schema in properties.items():
            direct[(name,)] = _parameter_types(parameter_schema, document, seen)[()]
    constraints = [direct]
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#"):
        with suppress(UnicodeDecodeError):
            reference = "#" + unquote(reference[1:], errors="strict")
    if isinstance(reference, str) and (reference == "#" or reference.startswith("#/")) and reference not in seen:
        target: JSON = document
        for segment in reference[1:].split("/")[1:]:
            part = segment.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict):
                target = target.get(part)
            elif isinstance(target, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
                try:
                    target = target[int(part)]
                except (IndexError, ValueError):
                    target = None
                    break
            else:
                target = None
                break
        constraints.append(_parameter_types(target, document, seen | {reference}))
    for combination in ("anyOf", "oneOf"):
        alternatives = schema.get(combination)
        if isinstance(alternatives, list):
            options = [
                inferred
                for option in alternatives
                if (inferred := _parameter_types(option, document, seen))[()] != frozenset()
            ]
            # Property constraints only apply to object instances; root types describe every instance.
            object_options = [
                option for option in options if (root_types := option[()]) is None or "object" in root_types
            ]
            combined: dict[tuple[()] | tuple[str], frozenset[str] | None] = {(): frozenset()}
            for path in {()} | set().union(*(option.keys() for option in object_options)):
                applicable = options
                if path:
                    applicable = object_options
                possible = [option.get(path) for option in applicable]
                combined[path] = (
                    None
                    if any(types is None for types in possible)
                    else frozenset().union(*(types for types in possible if types is not None))
                )
            constraints.append(combined)
    conjunctions = schema.get("allOf")
    if isinstance(conjunctions, list):
        constraints.extend(_parameter_types(option, document, seen) for option in conjunctions)
    resolved: dict[tuple[()] | tuple[str], frozenset[str] | None] = {}
    for path in set().union(*(constraint.keys() for constraint in constraints)):
        restricted = [types for constraint in constraints if (types := constraint.get(path)) is not None]
        if restricted:
            first, *remaining = restricted
            resolved[path] = first.intersection(*remaining)
        else:
            resolved[path] = None
    return resolved


def _hold_marker_prefix(text: str, markers: Iterable[str]) -> str:
    held = max(
        (
            length
            for marker in markers
            for length in range(1, min(len(marker), len(text)) + 1)
            if text.endswith(marker[:length])
        ),
        default=0,
    )
    return text[: len(text) - held]


def _json_value_end(text: str) -> int | None:
    start = len(text) - len(text.lstrip())
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return end


def _json_body_end(text: str) -> int | None:
    start = len(text) - len(text.lstrip())
    body = text[start:]
    if not body or not (
        body[0] in '{["-0123456789'
        or any(literal.startswith(body) or body.startswith(literal) for literal in ("true", "false", "null"))
    ):
        return None
    end = _json_value_end(text)
    if end is None:
        return len(text)
    return end + len(text[end:]) - len(text[end:].lstrip())


class ReasoningEffort(StrEnum):
    XHIGH = "xhigh"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_REASONING = "no_reasoning"


@dataclass(frozen=True)
class ReasoningConfig:
    default_reasoning_effort: ReasoningEffort
    # Jinja's `is true`/`is false` tests reject string spellings, while other template fields expect strings.
    reasoning_effort_to_template_fields: frozendict[ReasoningEffort, frozendict[str, str | bool]]

    def __post_init__(self) -> None:
        if self.default_reasoning_effort not in self.reasoning_effort_to_template_fields:
            raise ValueError("The default reasoning effort must have template fields.")

    def template_fields(self, effort: ReasoningEffort | None) -> frozendict[str, str | bool]:
        if effort is None:
            effort = self.default_reasoning_effort
        if effort not in self.reasoning_effort_to_template_fields:
            raise ValueError(
                f"Reasoning effort {effort.value!r} is not supported by this model; "
                f"supported efforts: {self.reasoning_effort_to_template_fields}."
            )
        return self.reasoning_effort_to_template_fields[effort]

    def effort_for_thinking(self, *, enabled: bool) -> ReasoningEffort:
        if not enabled:
            return ReasoningEffort.NO_REASONING
        if self.default_reasoning_effort is not ReasoningEffort.NO_REASONING:
            return self.default_reasoning_effort
        enabled_efforts = tuple(
            effort for effort in self.reasoning_effort_to_template_fields if effort is not ReasoningEffort.NO_REASONING
        )
        if len(enabled_efforts) != 1:
            raise ValueError("This model requires an explicit reasoning effort to enable thinking.")
        (effort,) = enabled_efforts
        return effort


def _strftime_now(format_string: str) -> str:
    return datetime.now().strftime(format_string)  # noqa: DTZ005


def _raise_template_error(message: str) -> NoReturn:
    raise ValueError(message)


def _regex_literal_runs(pattern: SubPattern) -> Iterable[str]:
    # Python's regex parser is private; use it only here to keep streaming delimiters owned by the regex.
    literals = ""
    for operation, argument in pattern:
        if operation is LITERAL:
            literals += chr(argument)
            continue
        if literals:
            yield literals
            literals = ""
        if isinstance(argument, tuple):
            children = argument
        else:
            children = (argument,)
        for child in children:
            if isinstance(child, SubPattern):
                yield from _regex_literal_runs(child)
            elif isinstance(child, list):
                for branch in child:
                    if isinstance(branch, SubPattern):
                        yield from _regex_literal_runs(branch)
    if literals:
        yield literals


class HuggingFaceMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]
    thinking: NotRequired[str]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]


def _participant_header(message: HuggingFaceMessage) -> str:
    name = message.get("name")
    if name is None or message["role"] == "tool":
        return ""
    return " name=" + json.dumps(name, ensure_ascii=False)


class HuggingFaceRequest(TypedDict):
    add_generation_prompt: bool
    bos_token: str | None
    eos_token: str | None
    messages: list[HuggingFaceMessage]
    tools: list[ToolSchema] | None


@dataclass(frozen=True)
class UserMessage:
    content: str
    name: str | None = None


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
    name: str | None = None


@dataclass(frozen=True)
class _ParsedResponse:
    chain_of_thought: str | None = None
    response: str = ""
    tool_calls: tuple[_GeneratedToolCall, ...] = ()
    response_spans: tuple[tuple[int, int], ...] = ()
    unclosed_tool_start: int | None = None

    def to_message(self) -> AssistantMessage:
        if self.unclosed_tool_start is not None:
            raise ValueError(f"Incomplete or invalid tool-call block at character {self.unclosed_tool_start}.")
        return AssistantMessage(
            self.chain_of_thought, self.response, tuple(call.to_tool_call() for call in self.tool_calls)
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        spans: Iterable[tuple[int, int]],
        *,
        response: str = "",
        chain_of_thought: str | None = None,
        tool_calls: tuple[_GeneratedToolCall, ...] = (),
        unclosed_tool_start: int | None = None,
    ) -> "_ParsedResponse":
        return cls(
            chain_of_thought,
            response,
            tool_calls,
            tuple((len(text[:start].encode()), len(text[:end].encode())) for start, end in spans if end > start),
            unclosed_tool_start,
        )

    def token_position(self, start: int, end: int) -> int | None:
        """Maps a sampled token's raw bytes to the first visible response byte they overlap."""
        position = 0
        for span_start, span_end in self.response_spans:
            overlap = max(start, span_start)
            if overlap < min(end, span_end):
                return position + overlap - span_start
            position += span_end - span_start
        return None


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
class _TokenizerDecoder:
    type: str
    decoders: "tuple[_TokenizerDecoder, ...]" = ()
    replacement: str | None = None

    def find(self, decoder_type: str) -> "_TokenizerDecoder | None":
        if self.type == decoder_type:
            return self
        for decoder in self.decoders:
            if found := decoder.find(decoder_type):
                return found
        return None


@dataclass(frozen=True)
class ChatCodec(TokenCodec[Iterable[Message], AssistantMessage, ChatCodecConfig]):
    @cached_property
    def _tokenizer_decoder(self) -> _TokenizerDecoder | None:
        if self.tokenizer.decoder is None:
            return None
        # Tokenizers exposes Sequence children only through its serialized decoder state.
        return make_converter().loads(self.tokenizer.decoder.__getstate__(), _TokenizerDecoder)

    @cached_property
    def _byte_token_ids(self) -> dict[int, int]:
        if self._tokenizer_decoder is None or self._tokenizer_decoder.find("ByteFallback") is None:
            return {}
        return {
            token_id: byte_value
            for byte_value in range(256)
            if (token_id := self.tokenizer.token_to_id(f"<0x{byte_value:02X}>")) is not None
        }

    @cached_property
    def _byte_level_char_bytes(self) -> dict[str, int] | None:
        if self._tokenizer_decoder is None or self._tokenizer_decoder.find("ByteLevel") is None:
            return None
        # ByteLevel maps printable bytes to themselves and the rest to characters above U+00FF.
        printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
        remaining = [byte for byte in range(256) if byte not in printable]
        return {chr(byte): byte for byte in printable} | {
            chr(256 + offset): byte for offset, byte in enumerate(remaining)
        }

    @cached_property
    def prompt_template(self) -> Template:
        environment = Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        # Hugging Face templates emit JSON, without Jinja's HTML escaping or key sorting.
        environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)
        template = self.config.prompt_template
        tool_format = self.config.tool_call_format
        # These native templates omit OpenAI participant names, including tool-only assistant messages.
        if tool_format in (ToolCallFormat.QWEN_XML, ToolCallFormat.LIQUID):
            for quote in ("'", '"'):
                template = template.replace(
                    f"{quote}<|im_start|>{quote} + message.role + {quote}\\n",
                    f"{quote}<|im_start|>{quote} + message.role + participant_header(message) + {quote}\\n",
                )
            if tool_format is ToolCallFormat.QWEN_XML:
                template = template.replace(
                    "'<|im_start|>system\\n'",
                    "'<|im_start|>system' + (participant_header(messages[0]) "
                    "if messages[0].role == 'system' else '') + '\\n'",
                )
            else:
                # Liquid removes the first system message before emitting its combined system/tool header.
                template = (
                    template.replace(
                        "{%- set messages = messages[1:] -%}",
                        "{%- set participant_system = messages[0] -%}{%- set messages = messages[1:] -%}",
                    )
                    .replace(
                        "{%- if ns.system_prompt -%}",
                        "{%- if ns.system_prompt or "
                        "(participant_system is defined and participant_header(participant_system)) -%}",
                    )
                    .replace(
                        '"<|im_start|>system\\n" + ns.system_prompt',
                        '"<|im_start|>system" + (participant_header(participant_system) '
                        'if participant_system is defined else "") + "\\n" + ns.system_prompt',
                    )
                )
        elif tool_format is ToolCallFormat.MUSE_ATEM:
            for role in ("system", "user"):
                template = template.replace(
                    f"'<|start|>{role}<|message|>'",
                    f"'<|start|>{role}' + participant_header(message) + '<|message|>'",
                )
            template = (
                template.replace(
                    "'<|start|>assistant to=self<|message|>'",
                    "'<|start|>assistant' + participant_header(message) + ' to=self<|message|>'",
                )
                .replace(
                    "'<|start|>assistant to=' + tc.function.name",
                    "'<|start|>assistant' + participant_header(message) + ' to=' + tc.function.name",
                )
                .replace(
                    "{{- '<|start|>assistant' -}}\n            {%- if recipient -%}",
                    "{{- '<|start|>assistant' + participant_header(message) -}}\n            {%- if recipient -%}",
                )
                .replace(
                    "{%- set rns = namespace(name=tcid if tcid else '') -%}\n            {%- for m in messages -%}",
                    "{%- set rns = namespace(name=tcid if tcid else '') -%}\n"
                    "            {%- for m in messages[:loop.index0] -%}",
                )
                .replace(
                    "{%- if message.get('tool_calls') -%}\n            {%- for tc in message['tool_calls'] -%}",
                    "{%- if message.get('tool_calls') -%}\n"
                    "            {%- if message.get('content') -%}\n"
                    "                {{- '<|start|>assistant' + participant_header(message)"
                    " + ' to=user<|message|>' -}}\n"
                    "                {{- render_content(message['content']) + '<|eom|>' -}}\n"
                    "            {%- endif -%}\n"
                    "            {%- for tc in message['tool_calls'] -%}",
                )
            )
        if tool_format is ToolCallFormat.LIQUID:
            template = template.replace(
                "format_arg_value(arg_value)]",
                "tool_argument_value(arg_value, func_name, tools, format_arg_value(arg_value))]",
            )
        elif tool_format is ToolCallFormat.QWEN_XML:
            for expression in (
                "args_value | tojson | safe if args_value is mapping or "
                "(args_value is sequence and args_value is not string) else args_value | string",
                "args_value | string if args_value is string else args_value | tojson | safe",
            ):
                template = template.replace(
                    "set args_value = " + expression,
                    "set args_value = tool_argument_value(args_value, tool_call.name, tools, (" + expression + "))",
                )
        elif tool_format is ToolCallFormat.MUSE_ATEM:
            template = template.replace(
                "{%- if v is boolean -%}",
                "{%- set rendered_argument -%}{%- if v is boolean -%}",
            ).replace(
                "        {{- '</atem:parameter>\\n' -}}",
                "        {%- endset -%}"
                "{{- tool_argument_value(v, tc.function.name, tools, rendered_argument) -}}"
                "{{- '</atem:parameter>\\n' -}}",
            )
        return environment.from_string(
            template,
            globals={"participant_header": _participant_header, "tool_argument_value": self._render_tool_argument},
        )

    def _render_tool_argument(self, value: JSON, name: str, tools: Iterable[ToolSchema] | None, rendered: str) -> str:
        definition = _tool_function(name, tools or ())
        if definition is None or definition.get("strict") is not True:
            return rendered
        if self.config.tool_call_format is ToolCallFormat.LIQUID and (value is None or isinstance(value, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    @cached_property
    def output_parser_regex(self) -> Pattern | None:
        if self.config.output_parser_regex is None:
            return None
        return re.compile(self.config.output_parser_regex)

    @cached_property
    def output_markers(self) -> tuple[str, ...]:
        if self.config.output_parser_regex is None:
            return ()
        return tuple(
            literal for literal in _regex_literal_runs(parse(self.config.output_parser_regex)) if "<" in literal
        )

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
        if tools is not None:
            tools = tuple(tools)
        if tools and self.config.tool_call_format is None:
            raise ValueError("This model does not support tool calling.")
        if tools and self.config.tool_call_format in (ToolCallFormat.QWEN_XML, ToolCallFormat.MUSE_ATEM):
            for tool in tools:
                definition = tool.get("function")
                if not isinstance(definition, dict) or definition.get("strict") is True:
                    continue
                parameters = definition.get("parameters")
                if not isinstance(parameters, dict):
                    continue
                for path, types in _parameter_types(parameters, parameters).items():
                    if path and types is not None and "string" in types and types - {"string"}:
                        raise ValueError(
                            f"Tool parameter {path[0]!r} mixes string and nonstring types; this model's native "
                            "unquoted string format cannot distinguish those types."
                        )
        if tools and self.config.tool_call_format is ToolCallFormat.LIQUID:
            for tool in tools:
                definition = tool.get("function")
                if not isinstance(definition, dict):
                    continue
                name = definition.get("name")
                parameters = definition.get("parameters")
                names = (name,)
                if isinstance(parameters, dict):
                    names = (name, *(path[0] for path in _parameter_types(parameters, parameters) if path))
                if any(not isinstance(name, str) or re.fullmatch(_LIQUID_NAME, name) is None for name in names):
                    raise ValueError("Tool name cannot be represented in Liquid's native format.")
        for tool in tools or ():
            definition = tool.get("function")
            if isinstance(definition, dict) and definition.get("strict") is True:
                self._strict_tool_parameters(definition)
        template_context: dict[str, object] = {
            **self.request_to_dict(messages, tools),
            "strftime_now": _strftime_now,
            "raise_exception": _raise_template_error,
        }

        reasoning_config = self.config.reasoning_config
        if reasoning_config is None and reasoning_effort is not None:
            raise ValueError("This model does not support configurable reasoning effort.")
        if reasoning_config is not None:
            template_context.update(reasoning_config.template_fields(reasoning_effort))

        return self.prompt_template.render(template_context)

    def encode_request(
        self,
        request: Iterable[Message],
        *,
        tools: Iterable[ToolSchema] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return self.encode_text(self.render_request(request, tools=tools, reasoning_effort=reasoning_effort))

    def parse_response(self, response: str, *, tools: Iterable[ToolSchema] | None = None) -> AssistantMessage:
        return self._parse_response(response, tools=tuple(tools or ()), final=True).to_message()

    def _parse_response(
        self,
        response: str,
        *,
        tools: tuple[ToolSchema, ...],
        final: bool,
        prefix: str = "",
        parallel_tool_calls: bool | None = None,
        response_schema: dict[str, JSON] | None = None,
    ) -> _ParsedResponse:
        tool_format = None
        if tools:
            tool_format = self.config.tool_call_format
        if self.config.tool_call_format is ToolCallFormat.MUSE_ATEM and (tools or response_schema is not None):
            return self._parse_muse_response(
                response,
                tools=tools,
                final=final,
                prefix=prefix.rstrip(),
                parallel_tool_calls=parallel_tool_calls,
                response_schema=response_schema,
            )
        text = prefix + response
        parse_text = text
        native_blocks: list[tuple[int, int | None, tuple[_GeneratedToolCall, ...]]] = []
        if tool_format is not None and any(
            isinstance(function := tool.get("function"), dict) and function.get("strict") is True for tool in tools
        ):
            position = 0
            if response_schema is not None:
                match = self.output_parser_regex.fullmatch(text) if self.output_parser_regex is not None else None
                json_start = 0
                if _json_body_end(text) is None and match is not None:
                    json_start = match.start("response")
                if json_start >= 0 and (json_end := _json_body_end(text[json_start:])) is not None:
                    position = json_start + json_end
                    parse_text = text[:json_start] + "x" * json_end + text[position:]
            while position < len(text):
                if response_schema is not None and (json_end := _json_body_end(text[position:])) is not None:
                    end = position + json_end
                    parse_text = parse_text[:position] + "x" * json_end + parse_text[end:]
                    position = end
                start = text.find(tool_format.opening_tag, position)
                if start < 0:
                    break
                body_start = start + len(tool_format.opening_tag)
                parsed, native_end = tool_format.parse_calls(
                    text[body_start:], tools, parallel_tool_calls=parallel_tool_calls
                )
                end = None
                if native_end is not None and text.startswith(tool_format.closing_tag, body_start + native_end):
                    end = body_start + native_end + len(tool_format.closing_tag)
                native_blocks.append((start, end, parsed))
                position = len(text) if end is None else end
                # The canonical native scan makes argument markers opaque to channel framing, at the same offsets.
                parse_text = parse_text[:start] + "x" * (position - start) + parse_text[position:]
                if parallel_tool_calls is False and parsed:
                    break
        groups: dict[str, str] = {"response": response}
        response_start = 0
        match = None
        json_body = False
        if self.output_parser_regex is not None and not (
            response_schema is not None and not prefix and _json_body_end(text) is not None
        ):
            match = self.output_parser_regex.fullmatch(parse_text)
            if match is None and not final:
                return _ParsedResponse()
            if match is not None:
                held_back = 0
                json_body = response_schema is not None and _json_body_end(match["response"] or "") is not None
                if not final and not json_body:
                    held_back = len(text) - len(_hold_marker_prefix(parse_text, self.output_markers))
                    while preceding_marker := max(
                        (
                            len(marker)
                            for marker in self.output_markers
                            if parse_text.endswith(marker, 0, len(text) - held_back)
                        ),
                        default=0,
                    ):
                        held_back += preceding_marker
                groups = {}
                for name in match.re.groupindex:
                    start, end = match.span(name)
                    if start >= 0 and end > len(prefix):
                        groups[name] = text[max(start, len(prefix)) : min(end, len(text) - held_back)]
                        if name == "response":
                            response_start = max(start, len(prefix)) - len(prefix)
        calls: list[_GeneratedToolCall] = []
        spans: list[tuple[int, int]] = []
        unclosed_tool_start = None
        if tool_format is not None:
            visible = ""
            remaining = groups.get("response", "")
            position = response_start
            while remaining:
                if response_schema is not None and (json_end := _json_body_end(remaining)) is not None:
                    visible += remaining[:json_end]
                    spans.append((position, position + json_end))
                    position += json_end
                    remaining = remaining[json_end:]
                if tool_format.opening_tag not in remaining:
                    break
                before, body = remaining.split(tool_format.opening_tag, 1)
                visible += before
                spans.append((position, position + len(before)))
                position += len(before) + len(tool_format.opening_tag)
                native_block = next(
                    (
                        block
                        for block in native_blocks
                        if block[0] == position + len(prefix) - len(tool_format.opening_tag)
                    ),
                    None,
                )
                if native_block is not None:
                    _, end, parsed = native_block
                    calls.extend(call.at_offset(position) for call in parsed)
                    if end is None:
                        unclosed_tool_start = position - len(tool_format.opening_tag)
                        remaining = ""
                        break
                    remaining = body[end - position - len(prefix) :]
                    position = end - len(prefix)
                    if parallel_tool_calls is False and parsed:
                        remaining = ""
                        break
                    continue
                if tool_format.closing_tag not in body:
                    unclosed_tool_start = position - len(tool_format.opening_tag)
                    parsed, _ = tool_format.parse_calls(
                        _hold_marker_prefix(body, (tool_format.closing_tag,)),
                        tools,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                    calls.extend(call.at_offset(position) for call in parsed)
                    remaining = ""
                    break
                body, remaining = body.split(tool_format.closing_tag, 1)
                parsed, native_end = tool_format.parse_calls(body, tools, parallel_tool_calls=parallel_tool_calls)
                calls.extend(call.at_offset(position) for call in parsed)
                if native_end is None:
                    unclosed_tool_start = position - len(tool_format.opening_tag)
                if parallel_tool_calls is False and parsed:
                    remaining = ""
                    break
                position += len(body) + len(tool_format.closing_tag)
            if not final:
                remaining = _hold_marker_prefix(remaining, (tool_format.opening_tag,))
            groups["response"] = visible + remaining
            spans.append((position, position + len(remaining)))
        else:
            spans.append((response_start, response_start + len(groups.get("response", ""))))
        first_visible_start, first_visible_end = spans[0]
        if (
            not final
            and not json_body
            and not native_blocks
            and (not calls or response[first_visible_start:first_visible_end].strip())
            and groups.get("response")
            and match is not None
            and self.output_parser_regex is not None
        ):
            # A leading native call resolves the channel; preceding prose can still be unmarked reasoning.
            for marker in self.output_markers:
                continued = self.output_parser_regex.fullmatch(parse_text + marker)
                if continued is not None and continued.start("response") > match.start("response"):
                    return _ParsedResponse(chain_of_thought=groups.get("chain_of_thought"))
        return _ParsedResponse.from_text(
            response, spans, **groups, tool_calls=tuple(calls), unclosed_tool_start=unclosed_tool_start
        )

    def _parse_muse_response(
        self,
        response: str,
        *,
        tools: tuple[ToolSchema, ...],
        final: bool,
        prefix: str,
        parallel_tool_calls: bool | None = None,
        response_schema: dict[str, JSON] | None = None,
    ) -> _ParsedResponse:
        reasoning = ""
        text = ""
        calls: list[_GeneratedToolCall] = []
        spans: list[tuple[int, int]] = []
        raw = prefix + response
        unclosed_tool_start = None
        remaining = raw.removeprefix(_MUSE_ASSISTANT_HEADER)
        terminators = _MUSE_TERMINATORS
        while remaining:
            header_start = len(raw) - len(remaining) - len(prefix)
            header = _MUSE_RECIPIENT_HEADER.match(remaining)
            if header is None:
                if final and remaining.strip():
                    if (
                        not remaining.lstrip().startswith("to=")
                        and not "to=".startswith(remaining.lstrip())
                        and not calls
                        and not reasoning
                        and not text
                    ):
                        return _ParsedResponse.from_text(response, ((0, len(response)),), response=raw)
                    unclosed_tool_start = len(raw) - len(remaining) - len(prefix)
                break
            recipient = header[1]
            remaining = remaining[header.end() :]
            body_start = len(raw) - len(remaining) - len(prefix)
            tool_turn = recipient not in ("self", "user")
            if not tool_turn and _tool_function(recipient, tools) is not None:
                opening = ToolCallFormat.MUSE_ATEM.opening_tag
                body = remaining.lstrip()
                if not final and opening.startswith(body):
                    break
                if body.startswith(opening):
                    body = body[len(opening) :].lstrip()
                    invoke = f'<atem:invoke name="{recipient}">'
                    if not final and invoke.startswith(body):
                        break
                    tool_turn = body.startswith(invoke)
            json_end = None
            if recipient == "user" and response_schema is not None:
                json_end = _json_body_end(remaining)
            native = None
            native_start = len(remaining) - len(remaining.lstrip()) + len(ToolCallFormat.MUSE_ATEM.opening_tag)
            if (
                tool_turn
                and any(
                    isinstance(function := tool.get("function"), dict) and function.get("strict") is True
                    for tool in tools
                )
                and remaining.lstrip().startswith(ToolCallFormat.MUSE_ATEM.opening_tag)
            ):
                native = ToolCallFormat.MUSE_ATEM.parse_calls(
                    remaining[native_start:], tools, parallel_tool_calls=parallel_tool_calls
                )
                _, native_end = native
                json_end = len(remaining)
                if native_end is not None and remaining.startswith(
                    ToolCallFormat.MUSE_ATEM.closing_tag, native_start + native_end
                ):
                    json_end = native_start + native_end + len(ToolCallFormat.MUSE_ATEM.closing_tag)
            ends = [
                (position, marker)
                for marker in terminators
                if (position := remaining.find(marker, json_end or 0)) >= 0
            ]
            if ends:
                position, marker = min(ends)
                body, remaining = remaining[:position], remaining[position + len(marker) :]
                remaining = remaining.removeprefix(_MUSE_ASSISTANT_HEADER)
            else:
                body, remaining = remaining, ""
                if not final:
                    body_end = json_end or 0
                    body = body[:body_end] + _hold_marker_prefix(body[body_end:], terminators)
            if recipient == "self" and not tool_turn:
                reasoning += body
            elif recipient == "user" and not tool_turn:
                text += body
                spans.append((max(0, body_start), body_start + len(body)))
            else:
                tool_format = ToolCallFormat.MUSE_ATEM
                leading = len(body) - len(body.lstrip())
                body = body.lstrip()
                if not body.startswith(tool_format.opening_tag):
                    unclosed_tool_start = header_start
                    break
                body = body[len(tool_format.opening_tag) :]
                if native is not None:
                    parsed, native_end = native
                    if (
                        native_end is None
                        or not body.startswith(tool_format.closing_tag, native_end)
                        or body[native_end + len(tool_format.closing_tag) :].strip()
                    ):
                        unclosed_tool_start = header_start
                else:
                    if tool_format.closing_tag in body:
                        body, suffix = body.split(tool_format.closing_tag, 1)
                        if suffix.strip():
                            unclosed_tool_start = header_start
                    else:
                        unclosed_tool_start = header_start
                        body = _hold_marker_prefix(body, (tool_format.closing_tag,))
                    parsed, native_end = tool_format.parse_calls(body, tools, parallel_tool_calls=parallel_tool_calls)
                if native_end is None or any(call.name != recipient for call in parsed):
                    unclosed_tool_start = header_start
                calls.extend(call.at_offset(body_start + leading + len(tool_format.opening_tag)) for call in parsed)
                if parallel_tool_calls is False and parsed:
                    break
        return _ParsedResponse.from_text(
            response,
            spans,
            chain_of_thought=reasoning or None,
            response=text,
            tool_calls=tuple(calls),
            unclosed_tool_start=unclosed_tool_start,
        )

    def json_response_grammar(self, schema: dict[str, JSON], *, prefix: str) -> "xgrammar.Grammar":
        import xgrammar  # noqa: PLC0415 - ordinary chat decoding does not load the grammar backend.

        from lalamo.models.json_schema import xgrammar_schema  # noqa: PLC0415

        rules = [rf"ws ::= [ \t\r\n]{{0,{_MAX_FORMATTING_WHITESPACE}}}"]
        continuation = "ws"
        ending = 'root ::= ""'
        if self.config.tool_call_format is ToolCallFormat.MUSE_ATEM:
            excludes = ", ".join(json.dumps(marker) for marker in _MUSE_TERMINATORS)
            rules += [
                f"thinking ::= TagDispatch(excludes=({excludes}), loop_after_dispatch=false)",
                f"assistant ::= {json.dumps(_MUSE_ASSISTANT_HEADER)} ws",
                'self ::= ws "to=self<|message|>" thinking "<|eom|>" assistant',
                'user ::= ws "to=user<|message|>" ws',
            ]
            continuation = "self* user"
            recipient = _MUSE_RECIPIENT_HEADER.fullmatch(prefix.rstrip())
            if recipient is not None:
                if recipient[1] == "self":
                    continuation = 'thinking "<|eom|>" assistant self* user'
                elif recipient[1] == "user":
                    continuation = "ws"
                else:
                    raise ValueError("A JSON response cannot start in a prefilled tool recipient.")
            finals = " | ".join(json.dumps(marker) for marker in _MUSE_TERMINATORS if marker != "<|eom|>")
            ending = f"root ::= ws ({finals})? ws"
        elif "</think>" in self.output_markers:
            rules += ['thinking ::= TagDispatch(excludes=("</think>"), loop_after_dispatch=false)']
            if prefix and "</think>" not in prefix:
                continuation = 'thinking "</think>" ws'
            elif not prefix:
                continuation = '("<think>" thinking "</think>")? ws'
        rules += [f"root ::= {json.dumps(prefix)} {continuation}"]
        try:
            body_grammar = xgrammar.Grammar.from_json_schema(
                xgrammar_schema(schema), strict_mode=False, max_whitespace_cnt=_MAX_FORMATTING_WHITESPACE
            )
        except RuntimeError as error:
            raise ValueError("Cannot compile response JSON schema.") from error
        return xgrammar.Grammar.concat(
            xgrammar.Grammar.from_ebnf("\n".join(rules)),
            body_grammar,
            xgrammar.Grammar.from_ebnf("\n".join([rules[0], ending])),
        )

    def tool_call_grammar(
        self,
        tools: Iterable[ToolSchema],
        *,
        prefix: str,
        require_call: bool = True,
        response_schema: dict[str, JSON] | None = None,
    ) -> str:
        tools = tuple(tools)
        tool_format = self.config.tool_call_format
        if tool_format is None:
            raise ValueError("This model does not support tool calling.")
        names: list[str] = []
        for tool in tools:
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(name := function.get("name"), str):
                raise TypeError("Forced tool calling requires named function tools.")
            names.append(name)
        if not names:
            raise ValueError("Forced tool calling requires at least one function tool.")

        prelude = '"to=self<|message|>"'
        if response_schema is None and not require_call:
            prelude = '("to=self<|message|>" | "to=user<|message|>")'
        rules = [rf"ws ::= [ \t\r\n]{{0,{_MAX_FORMATTING_WHITESPACE}}}"]
        quoted_prefix = json.dumps(prefix)
        opening = json.dumps(tool_format.opening_tag)
        closing = json.dumps(tool_format.closing_tag)
        if tool_format is ToolCallFormat.LIQUID:
            functions = " | ".join(json.dumps(name) for name in names)
            entries = 'function (ws "," ws function)* ws ","?'
            if not require_call:
                entries = f"({entries})?"
            argument_names = []
            for tool in tools:
                function = tool["function"]
                assert isinstance(function, dict)
                parameters = function.get("parameters")
                if isinstance(parameters, dict):
                    argument_names.extend(path[0] for path in _parameter_types(parameters, parameters) if path)
            rules += [
                "identifier ::= [a-zA-Z_0-9-]+" + "".join(" | " + json.dumps(name) for name in argument_names),
                f'function ::= ({functions}) ws "(" ws (argument ("," ws argument)* ","? ws)? ")"',
                r'argument ::= identifier ws "=" ws value ws',
                f'call ::= {opening} ws "[" ws {entries} ws "]" ws {closing}',
                r'value ::= number | string | "True" | "False" | "None" | list | dict | "(" ws value ws ")"',
                r'number ::= [-+]? ws ("0"+ | [1-9] [0-9]* | ([0-9]+ "." [0-9]* | "." [0-9]+) exponent?'
                r" | [0-9]+ exponent)",
                r"exponent ::= [eE] [-+]? [0-9]+",
                r'list ::= "[" ws (value ws ("," ws value ws)* ","?)? "]"',
                r'dict ::= "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value ws)* ","?)? "}"',
                r'''string ::= "\"" double-char* "\"" | "'" single-char* "'"''',
                r"""double-char ::= [^"\\\r\n] | "\\" escape""",
                r"""single-char ::= [^'\\\r\n] | "\\" escape""",
                r"""escape ::= ["'\\abfnrtv] | [0-7]{1,3} | "x" [0-9a-fA-F]{2}"""
                r' | "u" [0-9a-fA-F]{4} | "U" [0-9a-fA-F]{8}',
            ]
        else:
            if tool_format is ToolCallFormat.QWEN_XML:
                functions = " | ".join(json.dumps(f"<function={name}>") for name in names)
                rules += [
                    f'function ::= ({functions}) ws parameter* "</function>" ws',
                    r'parameter ::= "<parameter=" [^<> \t\r\n]+ ">" parameter-text "</parameter>" ws',
                    f"call ::= {opening} ws function+ {closing}",
                ]
                excluded = ("</parameter>", "</function>", tool_format.closing_tag)
            else:
                for index, name in enumerate(names):
                    rules += [
                        f"function-{index} ::= {json.dumps(f'<atem:invoke name="{name}">')} "
                        'ws parameter* "</atem:invoke>" ws',
                        f"body-{index} ::= {opening} ws function-{index}+ {closing}",
                        f"call-{index} ::= ws {json.dumps(f'to={name}<|message|>')} ws body-{index}",
                    ]
                functions = " | ".join(f"call-{index}" for index in range(len(names)))
                rules += [
                    f"call ::= {functions}",
                    r'parameter ::= "<atem:parameter name=\"" [^<>" \t\r\n]+ "\">" '
                    r'parameter-text "</atem:parameter>" ws',
                    f"assistant ::= {json.dumps(_MUSE_ASSISTANT_HEADER)} ws",
                    f'prelude ::= ws {prelude} turn-text "<|eom|>" assistant',
                    'calls ::= call (ws "<|eom|>" assistant call)* ws "<|eom|>"? ws',
                ]
                excluded = ("</atem:parameter>", "</atem:invoke>", tool_format.closing_tag, *_MUSE_TERMINATORS)
            parameter_excludes = ", ".join(json.dumps(marker) for marker in excluded)
            rules += [f"parameter-text ::= TagDispatch(excludes=({parameter_excludes}), loop_after_dispatch=false)"]

        if tool_format is ToolCallFormat.MUSE_ATEM:
            turn_excludes = ", ".join(json.dumps(marker) for marker in _MUSE_TERMINATORS)
            rules += [f"turn-text ::= TagDispatch(excludes=({turn_excludes}), loop_after_dispatch=false)"]
            recipient = _MUSE_RECIPIENT_HEADER.fullmatch(prefix.rstrip())
            continuation = "prelude* calls"
            following = "call"
            ending = '"<|eom|>"'
            if not require_call:
                rules += [f"turn ::= ws {prelude} turn-text | call"]
                following = "turn"
                ending = "(" + " | ".join(json.dumps(marker) for marker in _MUSE_TERMINATORS) + ")"
            tail = f'(ws "<|eom|>" assistant {following})* ws {ending}? ws'
            if not require_call:
                continuation = f"(turn {tail})?"
            if recipient is not None:
                if recipient[1] in ("self", "user"):
                    body = "turn-text"
                    if recipient[1] == "user" and (require_call or response_schema is not None):
                        body = "ws"
                    continuation = f'{body} "<|eom|>" assistant prelude* calls'
                    if not require_call:
                        continuation = f"{body} {tail}"
                elif recipient[1] in names:
                    index = names.index(recipient[1])
                    continuation = f"ws body-{index} {tail}"
                else:
                    raise ValueError("The prefilled Muse recipient is not an allowed tool.")
        else:
            excluded = [tool_format.opening_tag]
            if self.config.eos_token is not None:
                excluded.append(self.config.eos_token)
            if "</think>" in self.output_markers:
                excluded += ["<think>", "</think>"]
            preamble_excludes = ", ".join(json.dumps(marker) for marker in excluded)
            preamble = f"preamble ::= TagDispatch(excludes=({preamble_excludes}), loop_after_dispatch=false)"
            if require_call or response_schema is not None:
                preamble = "preamble ::= ws"
            rules += [preamble]
            if require_call:
                rules += ["calls ::= call (ws call)* ws"]
            else:
                rules += ["calls ::= (call preamble)*"]
            continuation = "preamble calls"
            if "</think>" in self.output_markers:
                rules += [f"thinking-text ::= TagDispatch(excludes=({preamble_excludes}), loop_after_dispatch=false)"]
                if prefix and "</think>" not in prefix:
                    continuation = 'thinking-text "</think>" preamble calls'
                elif not prefix:
                    continuation = '("<think>" thinking-text "</think>")? preamble calls'
        rules += [f"root ::= {quoted_prefix} {continuation}"]
        if any(
            isinstance(function := tool.get("function"), dict) and function.get("strict") is True for tool in tools
        ):
            return self._strict_tool_call_grammar(
                tools, prefix=prefix, require_call=require_call, response_schema=response_schema, rules=rules
            )
        return "\n".join(rules)

    def _strict_tool_parameters(self, definition: dict[str, JSON]) -> dict[str, JSON]:
        from lalamo.models.json_schema import validate_json_schema  # noqa: PLC0415

        parameters: JSON = definition.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        if not isinstance(parameters, dict):
            raise TypeError("Strict tool parameters must be an object schema.")
        validate_json_schema(parameters, strict=True)
        properties = parameters.get("properties", {})
        assert isinstance(properties, dict)
        names = tuple(properties)
        if self.config.tool_call_format is ToolCallFormat.LIQUID:
            name = definition.get("name")
            if not isinstance(name, str):
                raise TypeError("Strict tool functions must have a name.")
            names = (name, *names)
        for key in names:
            if self.config.tool_call_format is ToolCallFormat.LIQUID:
                valid = re.fullmatch(_LIQUID_NAME, key) is not None
            elif self.config.tool_call_format is ToolCallFormat.QWEN_XML:
                valid = re.fullmatch(r"[^>\s]+", key) is not None
            else:
                valid = re.fullmatch(r'[^"\s]+', key) is not None
            if not valid:
                raise ValueError("Strict tool property name cannot be represented in this model's native format.")
        return parameters

    def _strict_tool_call_grammar(
        self,
        tools: tuple[ToolSchema, ...],
        *,
        prefix: str,
        require_call: bool,
        response_schema: dict[str, JSON] | None,
        rules: list[str],
    ) -> str:
        import xgrammar  # noqa: PLC0415
        from jsonschema import Draft202012Validator  # noqa: PLC0415
        from xgrammar.structural_tag import GrammarFormat, RepeatFormat, StructuralTag  # noqa: PLC0415

        from lalamo.models.json_schema import xgrammar_schema  # noqa: PLC0415

        def literal(text: str) -> xgrammar.Grammar:
            return xgrammar.Grammar.from_ebnf("root ::= " + json.dumps(text))

        def native_rule(rule: str) -> xgrammar.Grammar:
            return xgrammar.Grammar.from_ebnf("\n".join(rules), root_rule_name=rule)

        def repeat(grammar: xgrammar.Grammar, minimum: int = 0, maximum: int = -1) -> xgrammar.Grammar:
            return xgrammar.Grammar.from_structural_tag(
                StructuralTag(
                    format=RepeatFormat(min=minimum, max=maximum, content=GrammarFormat(grammar=str(grammar)))
                )
            )

        tool_format = self.config.tool_call_format
        assert tool_format is not None
        ws = native_rule("ws")
        functions: list[xgrammar.Grammar] = []
        calls: list[xgrammar.Grammar] = []
        names: list[str] = []
        for tool in tools:
            definition = tool["function"]
            assert isinstance(definition, dict)
            name = definition["name"]
            assert isinstance(name, str)
            names.append(name)
            if definition.get("strict") is True:
                parameters = self._strict_tool_parameters(definition)
                properties = parameters.get("properties", {})
                assert isinstance(properties, dict)
                validator = Draft202012Validator(parameters)
                arguments = [ws]
                for index, (key, node) in enumerate(properties.items()):
                    projected = {**parameters, "$ref": "#/properties/" + key.replace("~", "~0").replace("/", "~1")}
                    try:
                        value = xgrammar.Grammar.from_json_schema(
                            xgrammar_schema(projected),
                            strict_mode=False,
                            max_whitespace_cnt=_MAX_FORMATTING_WHITESPACE,
                        )
                    except RuntimeError as error:
                        raise ValueError("Cannot compile strict tool JSON schema.") from error
                    if tool_format is ToolCallFormat.LIQUID:
                        aliases = [
                            str(item) for item in (True, False, None) if validator.evolve(schema=node).is_valid(item)
                        ]
                        if aliases:
                            value = xgrammar.Grammar.union(
                                value,
                                xgrammar.Grammar.from_ebnf(
                                    "root ::= " + " | ".join(json.dumps(alias) for alias in aliases)
                                ),
                            )
                        if index:
                            arguments += [literal(","), ws]
                        arguments += [literal(key), ws, literal("="), ws, value, ws]
                    else:
                        if tool_format is ToolCallFormat.QWEN_XML:
                            opening, closing = "<parameter=" + key + ">", "</parameter>"
                        else:
                            opening, closing = '<atem:parameter name="' + key + '">', "</atem:parameter>"
                        arguments += [literal(opening), ws, value, ws, literal(closing), ws]
                body = xgrammar.Grammar.concat(*arguments)
            else:
                expression = "ws parameter*"
                if tool_format is ToolCallFormat.LIQUID:
                    expression = 'ws (argument ("," ws argument)* ","? ws)?'
                body = xgrammar.Grammar.from_ebnf(
                    "\n".join([*rules, "native-body ::= " + expression]), root_rule_name="native-body"
                )
            if tool_format is ToolCallFormat.LIQUID:
                opening, closing = name + "(", ")"
            elif tool_format is ToolCallFormat.QWEN_XML:
                opening, closing = "<function=" + name + ">", "</function>"
            else:
                opening, closing = '<atem:invoke name="' + name + '">', "</atem:invoke>"
            header = [literal(opening)]
            if tool_format is ToolCallFormat.LIQUID:
                header = [literal(name), ws, literal("(")]
            function = xgrammar.Grammar.concat(*header, body, literal(closing), ws)
            functions.append(function)
            if tool_format is ToolCallFormat.MUSE_ATEM:
                calls.append(
                    xgrammar.Grammar.concat(
                        ws,
                        literal("to=" + name + "<|message|>"),
                        ws,
                        literal(tool_format.opening_tag),
                        ws,
                        repeat(function, 1),
                        literal(tool_format.closing_tag),
                    )
                )
        if tool_format is not ToolCallFormat.MUSE_ATEM:
            function = xgrammar.Grammar.union(*functions)
            entries = repeat(function, 1)
            if tool_format is ToolCallFormat.LIQUID:
                entries = xgrammar.Grammar.concat(
                    function,
                    repeat(xgrammar.Grammar.concat(literal(","), ws, function)),
                    repeat(literal(","), 0, 1),
                    ws,
                )
                if not require_call:
                    entries = repeat(entries, 0, 1)
                entries = xgrammar.Grammar.concat(literal("["), ws, entries, literal("]"), ws)
            call = xgrammar.Grammar.concat(
                literal(tool_format.opening_tag), ws, entries, literal(tool_format.closing_tag)
            )
            preamble = native_rule("preamble")
            continuation = []
            if "</think>" in self.output_markers:
                if prefix and "</think>" not in prefix:
                    continuation += [native_rule("thinking-text"), literal("</think>")]
                elif not prefix:
                    continuation += [
                        repeat(
                            xgrammar.Grammar.concat(
                                literal("<think>"), native_rule("thinking-text"), literal("</think>")
                            ),
                            0,
                            1,
                        )
                    ]
            continuation += [preamble]
            if require_call:
                continuation += [repeat(xgrammar.Grammar.concat(call, ws), 1)]
            else:
                continuation += [repeat(xgrammar.Grammar.concat(call, preamble))]
        else:
            call = xgrammar.Grammar.union(*calls)
            assistant, prelude, body = native_rule("assistant"), native_rule("prelude"), native_rule("turn-text")
            turn, endings = call, literal("<|eom|>")
            if not require_call:
                recipients = ["self", "user"] if response_schema is None else ["self"]
                turn = xgrammar.Grammar.union(
                    call,
                    *[
                        xgrammar.Grammar.concat(ws, literal("to=" + recipient + "<|message|>"), body)
                        for recipient in recipients
                    ],
                )
                endings = xgrammar.Grammar.union(*(literal(marker) for marker in _MUSE_TERMINATORS))
            tail = [
                repeat(xgrammar.Grammar.concat(ws, literal("<|eom|>"), assistant, turn)),
                ws,
                repeat(endings, 0, 1),
                ws,
            ]
            continuation = [repeat(prelude), call, *tail]
            if not require_call:
                continuation = [repeat(xgrammar.Grammar.concat(turn, *tail), 0, 1)]
            recipient = _MUSE_RECIPIENT_HEADER.fullmatch(prefix.rstrip())
            if recipient is not None:
                if recipient[1] in ("self", "user"):
                    if recipient[1] == "user" and (require_call or response_schema is not None):
                        body = ws
                    continuation = [body, *tail]
                    if require_call:
                        continuation = [body, literal("<|eom|>"), assistant, repeat(prelude), call, *tail]
                else:
                    continuation = [
                        ws,
                        literal(tool_format.opening_tag),
                        ws,
                        repeat(functions[names.index(recipient[1])], 1),
                        literal(tool_format.closing_tag),
                        *tail,
                    ]
        return str(xgrammar.Grammar.concat(literal(prefix), *continuation))

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_tokens(self, tokens: list[int], *, hide_invalid_utf_chars: bool = False) -> str:
        errors = "ignore" if hide_invalid_utf_chars else "replace"
        return "".join(codecs.iterdecode(self._token_groups(tokens), "utf-8", errors=errors))

    def decode_token_bytes(self, token_id: int) -> bytes:
        byte = self._byte_token_ids.get(token_id)
        if byte is not None:
            return bytes((byte,))
        character_bytes = self._byte_level_char_bytes
        if character_bytes is not None:
            token = self.tokenizer.id_to_token(token_id)
            if token is not None and all(character in character_bytes for character in token):
                return bytes(character_bytes[character] for character in token)
        if self._tokenizer_decoder is not None:
            metaspace = self._tokenizer_decoder.find("Metaspace")
            token = self.tokenizer.id_to_token(token_id)
            if metaspace is not None and metaspace.replacement is not None and token is not None:
                return token.replace(metaspace.replacement, " ").encode()
        return self.tokenizer.decode([token_id], skip_special_tokens=False).encode("utf-8")

    def decode_token_spans(self, token_ids: Iterable[int]) -> tuple[tuple[int, int], ...]:
        """Token byte ranges after UTF-8 replacement, including all bytes of a replaced invalid sequence."""
        chunks = [self.decode_token_bytes(token_id) for token_id in token_ids]
        raw = b"".join(chunks)
        # Valid UTF-8 preserves byte offsets. Invalid sequences expand or contract to a three-byte U+FFFD.
        replacements = []
        position = 0
        while position < len(raw):
            try:
                raw[position:].decode()
                break
            except UnicodeDecodeError as error:
                start = position + error.start
                end = position + error.end
                replacements.append((start, end))
                position = end
        spans = []
        position = 0
        for chunk in chunks:
            start, end = position, position + len(chunk)
            position = end
            delta = 0
            rendered_start, rendered_end = start, end
            for invalid_start, invalid_end in replacements:
                if start >= invalid_end:
                    rendered_start += 3 - (invalid_end - invalid_start)
                elif start > invalid_start:
                    rendered_start = invalid_start + delta
                if end >= invalid_end:
                    rendered_end += 3 - (invalid_end - invalid_start)
                elif end > invalid_start:
                    rendered_end = invalid_start + delta + 3
                delta += 3 - (invalid_end - invalid_start)
            if not chunk:
                rendered_end = rendered_start
            spans.append((rendered_start, rendered_end))
        return tuple(spans)

    def _token_groups(self, tokens: list[int]) -> Iterable[bytes]:
        byte_token_ids = self._byte_token_ids
        for is_byte, group in itertools.groupby(tokens, key=lambda tid: tid in byte_token_ids):
            if is_byte:
                yield bytes(byte_token_ids[tid] for tid in group)
            else:
                yield self.tokenizer.decode(list(group), skip_special_tokens=False).encode("utf-8")

    def decode_stream(
        self,
        reasoning_effort: ReasoningEffort | None = None,
        *,
        prompt: str | None = None,
        tools: Iterable[ToolSchema] | None = None,
        stop_strings: tuple[str, ...] = (),
        parallel_tool_calls: bool | None = None,
        response_schema: dict[str, JSON] | None = None,
    ) -> "ChatDecodeStream":
        tools = tuple(tools or ())
        if prompt is None:
            prompt = self.render_request([UserMessage("")], tools=tools, reasoning_effort=reasoning_effort)
        prefix = ""
        if self.output_parser_regex is not None:
            for marker in self.output_markers:
                marker_position = prompt.rfind(marker)
                if marker_position < 0:
                    continue
                suffix = prompt[marker_position:]
                match = self.output_parser_regex.fullmatch(suffix)
                if (
                    match is not None
                    and all(not (value or "").strip() for value in match.groupdict().values())
                    and len(suffix) > len(prefix)
                ):
                    prefix = suffix
        if tools and self.config.tool_call_format is ToolCallFormat.MUSE_ATEM:
            suffix = prompt.rsplit(_MUSE_ASSISTANT_HEADER, 1)[-1]
            if _MUSE_RECIPIENT_HEADER.fullmatch(suffix.rstrip()) is not None:
                prefix = suffix
        return ChatDecodeStream(
            self,
            prefix=prefix,
            tools=tools,
            stop_strings=stop_strings,
            parallel_tool_calls=parallel_tool_calls,
            response_schema=response_schema,
        )

    def decode_response(self, response: list[int], *, tools: Iterable[ToolSchema] | None = None) -> AssistantMessage:
        return self.parse_response(self.decode_tokens(response), tools=tools)

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


@dataclass
class ChatDecodeStream:
    codec: ChatCodec
    prefix: str = ""
    tools: tuple[ToolSchema, ...] = ()
    stop_strings: tuple[str, ...] = ()
    parallel_tool_calls: bool | None = None
    response_schema: dict[str, JSON] | None = None
    decoder: DecodeStream = field(default_factory=DecodeStream)
    raw_response: str = ""
    parsed: _ParsedResponse = field(default_factory=_ParsedResponse)
    undecoded_token_ids: list[int] = field(default_factory=list)

    @property
    def reasoning(self) -> str:
        return self.parsed.chain_of_thought or ""

    @property
    def response(self) -> str:
        return self.parsed.response

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        parsed = self.parsed
        if parsed.unclosed_tool_start is not None:
            if parsed.unclosed_tool_start <= 0:
                return ()
            parsed = self.codec._parse_response(  # noqa: SLF001 - strict access derives the complete native prefix.
                self.raw_response[: parsed.unclosed_tool_start],
                tools=self.tools,
                final=True,
                prefix=self.prefix,
                response_schema=self.response_schema,
            )
        return parsed.to_message().tool_calls

    def step(self, token_id: int) -> tuple[str, str]:
        """Returns the newly visible reasoning and response text."""
        self.undecoded_token_ids.append(token_id)
        piece = self.decoder.step(self.codec.tokenizer, token_id)
        if piece is not None:
            self.raw_response += piece
            self.undecoded_token_ids.clear()
        parsed = self._message(final=False)
        reasoning = parsed.chain_of_thought or ""
        if not reasoning.startswith(self.reasoning) or not parsed.response.startswith(self.response):
            raise ValueError("The parsed output changed after text was streamed.")
        new_reasoning = reasoning[len(self.reasoning) :]
        new_response = parsed.response[len(self.response) :]
        self.parsed = parsed
        return new_reasoning, new_response

    @property
    def stop_position(self) -> int | None:
        positions = [position for stop in self.stop_strings if (position := self.raw_response.find(stop)) >= 0]
        if positions:
            return min(positions)
        return None

    @property
    def tool_call_position(self) -> int | None:
        if self.parallel_tool_calls is False and self.parsed.tool_calls:
            return self.parsed.tool_calls[0].source_end
        return None

    def _message(self, *, final: bool) -> _ParsedResponse:
        response = self.raw_response
        position = self.stop_position
        if position is not None:
            response = response[:position]
        elif not final:
            response = _hold_marker_prefix(response, self.stop_strings)
        return self.codec._parse_response(  # noqa: SLF001 - the stream and codec share this parsing boundary.
            response,
            tools=self.tools,
            final=final,
            prefix=self.prefix,
            parallel_tool_calls=self.parallel_tool_calls,
            response_schema=self.response_schema,
        )

    def finish(self) -> AssistantMessage:
        return self.finish_output().to_message()

    def finish_output(self) -> _ParsedResponse:
        # Tokenizers.DecodeStream has no flush method; decode a retained UTF-8 suffix with replacement at EOF.
        if self.undecoded_token_ids:
            self.raw_response += self.codec.decode_tokens(self.undecoded_token_ids)
            self.undecoded_token_ids.clear()
        self.parsed = self._message(final=True)
        return self.parsed
