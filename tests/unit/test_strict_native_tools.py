import json

import numpy as np
import pytest

from lalamo.models.chat_codec import ToolCallFormat, ToolSchema
from lalamo.utils.json import JSON
from tests.unit.test_generated_tool_output import native_call
from tests.unit.test_json_response import codec_for_pieces
from tests.unit.test_tool_choice_grammar import allowed, matcher, token_mask

pytestmark = pytest.mark.fast


def strict_tool(node: dict[str, JSON], name: str = "lookup") -> ToolSchema:
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": node},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def strict_call(tool_format: ToolCallFormat, value: str) -> str:
    return native_call(tool_format, "value", value, complete=True).removesuffix("<|eot|>")


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize(
    "node,value",
    [
        ({"type": ["string", "null"]}, None),
        ({"type": ["string", "null"]}, "null"),
        ({"type": "boolean"}, True),
        ({"type": "number", "minimum": -2.5, "maximum": 4.5}, 1.25),
        ({"type": "string", "minLength": 2, "maxLength": 3}, "café"[:3]),
        (
            {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean"},
                    "items": {"type": "array", "items": {"type": ["string", "null"]}},
                },
                "required": ["flag", "items"],
                "additionalProperties": False,
            },
            {"flag": False, "items": ["café", None]},
        ),
        (
            {
                "type": "string",
                "const": "</parameter></function></tool_call></think><|eot|>"
                "</atem:parameter></atem:invoke></atem:function_calls>",
            },
            "</parameter></function></tool_call></think><|eot|></atem:parameter></atem:invoke></atem:function_calls>",
        ),
    ],
)
def test_strict_schema_masks_invalid_values_and_roundtrips_native_json(
    tool_format: ToolCallFormat, node: dict[str, JSON], value: JSON
) -> None:
    tools = [strict_tool(node)]
    good = strict_call(tool_format, json.dumps(value, ensure_ascii=False))
    bad = strict_call(tool_format, json.dumps({"invalid": True}))
    codec = codec_for_pieces(tool_format, list(good))
    current = matcher(codec, tools, ["<EOS>", bad, good])
    mask = token_mask(current, 3)
    assert not allowed(mask, 0) and not allowed(mask, 1) and allowed(mask, 2)
    assert current.accept_token(2) and current.is_completed()
    assert allowed(token_mask(current, 3), 0)
    decoder = codec.decode_stream(prompt="", tools=tools)
    for piece in good:
        assert decoder.step(codec.tokenizer.token_to_id(piece))[1] == ""
    message = decoder.finish()
    assert message.response == "" and message.chain_of_thought is None
    assert [call["function"] for call in message.tool_calls] == [{"name": "lookup", "arguments": {"value": value}}]
    assert codec.parse_response(good, tools=tools) == message


@pytest.mark.parametrize(
    "tool_format,prefix",
    [
        *[
            (tool_format, prefix)
            for tool_format in (ToolCallFormat.QWEN_XML, ToolCallFormat.LIQUID)
            for prefix in ("", "<think>\n", "<think>\n\n</think>\n\n")
        ],
        (ToolCallFormat.MUSE_ATEM, ""),
    ],
)
def test_strict_native_arguments_preserve_thinking_literals_and_visible_byte_positions(
    tool_format: ToolCallFormat, prefix: str
) -> None:
    value = "</think> café ☃️ 東京"
    tools = [strict_tool({"type": "string"})]
    raw = strict_call(tool_format, json.dumps(value, ensure_ascii=False))
    head = ""
    tail = "visible café"
    if prefix and "</think>" not in prefix:
        head = "private</think>"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        head = "to=self<|message|>private<|eom|><|start|>assistant "
        tail = "<|eom|><|start|>assistant to=user<|message|>" + tail + "<|eot|>"
    text = head + raw + tail
    codec = codec_for_pieces(tool_format, list(text))
    decoder = codec.decode_stream(prompt=prefix, tools=tools)
    for piece in text:
        decoder.step(codec.tokenizer.token_to_id(piece))
    parsed = decoder.finish_output()
    assert parsed.response == "visible café"
    assert parsed.to_message().tool_calls[0]["function"]["arguments"] == {"value": value}
    assert b"".join(text.encode()[start:end] for start, end in parsed.response_spans) == "visible café".encode()
    if head:
        assert parsed.chain_of_thought == "private"


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("value", ['"partial\\', "1e-", '{"items":[true,', '"</think><|eot|>partial'])
def test_strict_native_length_and_raw_stop_do_not_repair_or_discard_argument_prefixes(
    tool_format: ToolCallFormat, value: str
) -> None:
    tools = [strict_tool({})]
    raw = native_call(tool_format, "value", value, complete=False)
    text = raw + "CUT" + strict_call(tool_format, "oops")
    codec = codec_for_pieces(tool_format, [raw, text])
    for token, stops in [(raw, ()), (text, ("CUT",))]:
        decoder = codec.decode_stream(prompt="", tools=tools, stop_strings=stops)
        decoder.step(codec.tokenizer.token_to_id(token))
        parsed = decoder.finish_output()
        assert parsed.response == "" and len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].arguments == '{"value":' + value
        assert parsed.tool_calls[0].source_end is None
        with pytest.raises(ValueError):
            parsed.to_message()


@pytest.mark.parametrize("native,expected", [("True", True), ("False", False), ("None", None), ('"\\/"', "/")])
def test_strict_liquid_uses_native_primitives_and_json_string_escapes(native: str, expected: JSON) -> None:
    tool_format = ToolCallFormat.LIQUID
    tools = [strict_tool({"type": ["boolean", "null", "string"]})]
    raw = strict_call(tool_format, native)
    codec = codec_for_pieces(tool_format, [raw])
    current = matcher(codec, tools, ["<EOS>", raw])
    assert current.accept_token(1) and current.is_completed()
    assert codec.parse_response(raw, tools=tools).tool_calls[0]["function"]["arguments"] == {"value": expected}


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_recursive_root_keeps_the_original_argument_object_in_native_projection(tool_format: ToolCallFormat) -> None:
    tools = [strict_tool({"anyOf": [{"type": "null"}, {"$ref": "#"}]})]
    value: JSON = {"value": {"value": None}}
    good = strict_call(tool_format, json.dumps(value))
    bad = strict_call(tool_format, json.dumps({"wrong": None}))
    codec = codec_for_pieces(tool_format, list(good))
    current = matcher(codec, tools, ["<EOS>", bad, good])
    mask = token_mask(current, 3)
    assert not allowed(mask, 0) and not allowed(mask, 1) and allowed(mask, 2)
    assert current.accept_token(2) and current.is_completed()
    assert codec.parse_response(good, tools=tools).tool_calls[0]["function"]["arguments"] == {"value": value}


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_strict_no_argument_tool_uses_the_omitted_parameter_default(tool_format: ToolCallFormat) -> None:
    tool: ToolSchema = {"type": "function", "function": {"name": "lookup", "strict": True}}
    raw = strict_call(tool_format, "0")
    if tool_format is ToolCallFormat.LIQUID:
        raw = raw.replace("value=0", "")
    elif tool_format is ToolCallFormat.QWEN_XML:
        raw = raw.replace("<parameter=value>0</parameter>", "")
    else:
        raw = raw.replace('<atem:parameter name="value">0</atem:parameter>', "")
    codec = codec_for_pieces(tool_format, [raw])
    current = matcher(codec, [tool], ["<EOS>", raw])
    assert current.accept_token(1) and current.is_completed()
    assert codec.parse_response(raw, tools=[tool]).tool_calls[0]["function"]["arguments"] == {}
    with pytest.raises(ValueError):
        codec.tool_call_grammar(
            [{"type": "function", "function": {"name": "lookup", "strict": True, "parameters": {}}}], prefix=""
        )


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_mixed_strict_and_ordinary_tools_keep_each_native_argument_contract(tool_format: ToolCallFormat) -> None:
    strict = strict_tool({"type": "string", "const": "</think><|eot|>"})
    ordinary: ToolSchema = {
        "type": "function",
        "function": {"name": "other", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}}},
    }
    first = strict_call(tool_format, json.dumps("</think><|eot|>"))
    second = native_call(tool_format, "count", "2", complete=True).replace("lookup", "other").removesuffix("<|eot|>")
    separator = ""
    if tool_format is ToolCallFormat.MUSE_ATEM:
        separator = "<|eom|><|start|>assistant "
    raw = first + separator + second
    codec = codec_for_pieces(tool_format, list(raw))
    tools = [strict, ordinary]
    current = matcher(codec, tools, ["<EOS>", raw])
    assert current.accept_token(1) and current.is_completed()
    decoder = codec.decode_stream(prompt="", tools=tools)
    for piece in raw:
        decoder.step(codec.tokenizer.token_to_id(piece))
    message = decoder.finish()
    assert [call["function"] for call in message.tool_calls] == [
        {"name": "lookup", "arguments": {"value": "</think><|eot|>"}},
        {"name": "other", "arguments": {"count": 2}},
    ]
    assert message.response == "" and message.chain_of_thought is None


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("position", ["before", "after"])
def test_strict_tools_and_public_json_do_not_capture_each_others_native_literals(
    tool_format: ToolCallFormat, position: str
) -> None:
    tools = [strict_tool({"type": "string"})]
    real = strict_call(tool_format, json.dumps("</think><|eot|>"))
    body = json.dumps({"quoted_call": real})
    public = body
    separator = ""
    if tool_format is ToolCallFormat.MUSE_ATEM:
        public = "to=user<|message|>" + body
        separator = "<|eom|><|start|>assistant "
    raw = real + separator + public
    if position == "after":
        raw = public + separator + real
    codec = codec_for_pieces(tool_format, list(raw))
    decoder = codec.decode_stream(prompt="", tools=tools, response_schema={})
    for piece in raw:
        decoder.step(codec.tokenizer.token_to_id(piece))
    parsed = decoder.finish_output()
    assert parsed.response == body and parsed.chain_of_thought is None
    assert [call.to_tool_call()["function"] for call in parsed.tool_calls] == [
        {"name": "lookup", "arguments": {"value": "</think><|eot|>"}}
    ]
    assert b"".join(raw.encode()[start:end] for start, end in parsed.response_spans) == body.encode()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_strict_call_cap_uses_the_real_function_end_after_json_delimiter_literals(tool_format: ToolCallFormat) -> None:
    tools = [strict_tool({"type": "string"})]
    first = strict_call(tool_format, json.dumps("</parameter></function></atem:invoke>"))
    second = strict_call(tool_format, '"second"')
    separator = ""
    function_end = "</function>"
    if tool_format is ToolCallFormat.LIQUID:
        function_end = ")"
    elif tool_format is ToolCallFormat.MUSE_ATEM:
        function_end = "</atem:invoke>"
        separator = "<|eom|><|start|>assistant "
    raw = first + separator + second
    codec = codec_for_pieces(tool_format, [raw])
    decoder = codec.decode_stream(prompt="", tools=tools, parallel_tool_calls=False)
    decoder.step(codec.tokenizer.token_to_id(raw))
    parsed = decoder.finish_output()
    assert decoder.tool_call_position is not None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].source_end == first.rfind(function_end) + len(function_end)
    assert parsed.tool_calls[0].to_tool_call()["function"]["arguments"] == {
        "value": "</parameter></function></atem:invoke>"
    }


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_strict_optional_calls_allow_text_and_eos_but_never_invalid_arguments(tool_format: ToolCallFormat) -> None:
    tools = [strict_tool({"type": "string", "minLength": 2})]
    good = strict_call(tool_format, '"okay"')
    bad = strict_call(tool_format, '"x"')
    plain = "plain answer "
    if tool_format is ToolCallFormat.MUSE_ATEM:
        plain = "to=user<|message|>plain answer<|eom|><|start|>assistant "
    codec = codec_for_pieces(tool_format, [])
    current = matcher(codec, tools, ["<EOS>", plain, good, bad], require_call=False)
    mask = token_mask(current, 4)
    assert allowed(mask, 0) and allowed(mask, 1) and allowed(mask, 2) and not allowed(mask, 3)
    assert current.accept_token(1) and current.accept_token(2) and current.is_completed()
    assert codec.parse_response(plain + good, tools=tools).response.strip() == "plain answer"


@pytest.mark.parametrize("name", ["class", "\u212a"])
def test_strict_liquid_function_names_preserve_literal_identity(name: str) -> None:
    codec = codec_for_pieces(ToolCallFormat.LIQUID, [])
    tools = [strict_tool({"type": "integer"}, name=name)]
    raw = strict_call(ToolCallFormat.LIQUID, "1").replace("lookup", name)
    current = matcher(codec, tools, ["<EOS>", raw])
    assert current.accept_token(1) and current.is_completed()
    assert codec.parse_response(raw, tools=tools).tool_calls[0]["function"] == {
        "name": name,
        "arguments": {"value": 1},
    }


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_strict_split_stop_clips_inside_a_json_string_before_later_native_bytes(tool_format: ToolCallFormat) -> None:
    tools = [strict_tool({"type": "string"})]
    raw = native_call(tool_format, "value", '"partCU', complete=False)
    later = 'T"' + strict_call(tool_format, '"unseen"')
    codec = codec_for_pieces(tool_format, [raw, later])
    decoder = codec.decode_stream(prompt="", tools=tools, stop_strings=("CUT",))
    decoder.step(codec.tokenizer.token_to_id(raw))
    assert decoder.parsed.tool_calls[0].arguments == '{"value":"part'
    decoder.step(codec.tokenizer.token_to_id(later))
    parsed = decoder.finish_output()
    assert len(parsed.tool_calls) == 1 and parsed.tool_calls[0].arguments == '{"value":"part'
    assert parsed.tool_calls[0].source_end is None


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_strict_incomplete_value_cannot_cap_at_a_later_function_closer(tool_format: ToolCallFormat) -> None:
    tools = [strict_tool({"type": "string"})]
    raw = strict_call(tool_format, '"missing quote')
    codec = codec_for_pieces(tool_format, [raw])
    decoder = codec.decode_stream(prompt="", tools=tools, parallel_tool_calls=False)
    decoder.step(codec.tokenizer.token_to_id(raw))
    parsed = decoder.finish_output()
    assert len(parsed.tool_calls) == 1 and parsed.tool_calls[0].arguments.startswith('{"value":"missing quote')
    assert decoder.tool_call_position is None
    with pytest.raises(ValueError):
        parsed.to_message()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("boundary", ["native_separator", "nested_json"])
def test_strict_tools_make_progress_under_formatting_whitespace_pressure(
    tool_format: ToolCallFormat, boundary: str
) -> None:
    text = "</parameter></function></think><|eot|> café" + " " * 40 + "東京"
    node: dict[str, JSON] = {"type": "string", "const": text}
    value: JSON = text
    first = json.dumps(text, ensure_ascii=False)
    rest = ""
    if boundary == "nested_json":
        node = {
            "type": "object",
            "properties": {"text": node, "done": {"type": "boolean", "const": True}},
            "required": ["text", "done"],
            "additionalProperties": False,
        }
        value = {"text": text, "done": True}
        first = '{"text":' + first
        rest = ',"done":true}'
    tools = [strict_tool(node)]
    definition = tools[0]["function"]
    assert isinstance(definition, dict)
    parameters = definition["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    properties["next"] = {"type": "integer", "const": 2}
    parameters["required"] = ["value", "next"]
    beginning = native_call(tool_format, "value", first, complete=False)
    if tool_format is ToolCallFormat.LIQUID:
        rest += ",next=2)]<|tool_call_end|>"
    elif tool_format is ToolCallFormat.QWEN_XML:
        rest += "</parameter><parameter=next>2</parameter></function></tool_call>"
    else:
        rest += '</atem:parameter><atem:parameter name="next">2</atem:parameter>'
        rest += "</atem:invoke></atem:function_calls>"
    vocabulary = ["<EOS>", " ", "\n", rest, beginning]
    codec = codec_for_pieces(tool_format, vocabulary[1:])
    current = matcher(codec, tools, vocabulary)
    assert allowed(token_mask(current, len(vocabulary)), 4) and current.accept_token(4)
    emitted = [beginning]
    logits = np.array([70.0, 100.0, 90.0, 80.0, -np.inf])
    for _ in range(128):
        mask = token_mask(current, len(vocabulary))
        permitted = np.array([allowed(mask, token) for token in range(len(vocabulary))])
        selected = int(np.argmax(np.where(permitted, logits, -np.inf)))
        assert current.accept_token(selected)
        if selected == 0:
            break
        emitted.append(vocabulary[selected])
    assert current.is_terminated()
    assert 1 < len(emitted) < 128 and rest in emitted
    decoder = codec.decode_stream(prompt="", tools=tools)
    for piece in emitted:
        decoder.step(codec.tokenizer.token_to_id(piece))
    result = decoder.finish()
    assert not result.response.strip() and result.chain_of_thought is None
    assert result.tool_calls[0]["function"]["arguments"] == {"value": value, "next": 2}
