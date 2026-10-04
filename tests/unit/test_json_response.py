import json

import numpy as np
import pytest
import xgrammar
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from lalamo.model_import.model_specs.muse_glimmer import MUSE_GLIMMER_OUTPUT_PARSER_REGEX
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.models.chat_codec import ChatCodec, ToolCallFormat
from lalamo.utils.json import JSON
from tests.unit.test_chat_codec import _chat_codec
from tests.unit.test_generated_tool_output import TOOL, native_call
from tests.unit.test_tool_choice_grammar import allowed, token_mask

pytestmark = pytest.mark.fast

_JSON_FRAMING = [
    (ToolCallFormat.QWEN_XML, "", ""),
    (ToolCallFormat.QWEN_XML, "<think>\n", "private</think>\n"),
    (ToolCallFormat.QWEN_XML, "<think>\n\n</think>\n\n", ""),
    (ToolCallFormat.LIQUID, "", "<think>private</think>\n"),
    (ToolCallFormat.LIQUID, "<think>\n", "private</think>\n"),
    (ToolCallFormat.LIQUID, "<think>\n\n</think>\n\n", ""),
    (ToolCallFormat.MUSE_ATEM, "", "to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>"),
    (ToolCallFormat.MUSE_ATEM, "to=self<|message|>\n", "private<|eom|><|start|>assistant to=user<|message|>"),
    (ToolCallFormat.MUSE_ATEM, "to=user<|message|>\n", ""),
]


def codec_for_pieces(tool_format: ToolCallFormat, pieces: list[str]) -> ChatCodec:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens(pieces)
    tokenizer.decoder = Fuse()
    parser = OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
    if tool_format is ToolCallFormat.MUSE_ATEM:
        parser = MUSE_GLIMMER_OUTPUT_PARSER_REGEX
    return _chat_codec(output_parser_regex=parser, tool_call_format=tool_format).config.init(tokenizer)


def native_public(tool_format: ToolCallFormat, body: str) -> str:
    if tool_format is ToolCallFormat.MUSE_ATEM:
        return "to=user<|message|>" + body
    return body


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("value", ["</think>", "<|eom|>", "<|eot|>", "<|end_of_text|>", "native_call", "café ☃️ 東京"])
def test_json_strings_remain_public_text_instead_of_native_controls(
    tool_format: ToolCallFormat, with_tools: bool, value: str
) -> None:
    if value == "native_call":
        value = native_call(tool_format, "count", "1", complete=True)
    body = json.dumps({"value": value}, ensure_ascii=False)
    raw = native_public(tool_format, body)
    pieces = list(raw)
    codec = codec_for_pieces(tool_format, pieces)
    decoder = codec.decode_stream(prompt="", tools=[TOOL] if with_tools else None, response_schema={})
    streamed = "".join(decoder.step(codec.tokenizer.token_to_id(piece))[1] for piece in pieces)
    parsed = decoder.finish_output()
    assert streamed == parsed.response == body
    assert parsed.tool_calls == () and parsed.chain_of_thought is None
    assert b"".join(raw.encode()[start:end] for start, end in parsed.response_spans) == body.encode()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("body", ['{"value":"<|eom', '"</think', "tr", "-1e", '[1,{"a":', "null"])
def test_json_prefix_at_budget_or_custom_stop_is_preserved_without_repair(
    tool_format: ToolCallFormat, body: str
) -> None:
    raw = native_public(tool_format, body)
    pieces = [raw + "CUT" + native_call(tool_format, "count", "oops", complete=True)]
    codec = codec_for_pieces(tool_format, [raw, *pieces])
    for token, stops in [(raw, ()), (pieces[0], ("CUT",))]:
        decoder = codec.decode_stream(prompt="", tools=[TOOL], stop_strings=stops, response_schema={})
        _, visible = decoder.step(codec.tokenizer.token_to_id(token))
        parsed = decoder.finish_output()
        assert visible == parsed.response == body and not parsed.tool_calls
        assert decoder.tool_calls == ()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("position", ["before", "after"])
def test_json_body_and_real_native_tool_call_retain_their_separate_meaning(
    tool_format: ToolCallFormat, position: str
) -> None:
    body = json.dumps({"quoted_call": native_call(tool_format, "count", "9", complete=True)})
    real = native_call(tool_format, "count", "1", complete=True)
    raw = body + real
    if position == "before":
        raw = real + body
    if tool_format is ToolCallFormat.MUSE_ATEM:
        raw = native_public(tool_format, body) + "<|eom|><|start|>assistant " + real
        if position == "before":
            raw = real.removesuffix("<|eot|>") + "<|eom|><|start|>assistant " + native_public(tool_format, body)
    codec = codec_for_pieces(tool_format, [raw])
    decoder = codec.decode_stream(prompt="", tools=[TOOL], response_schema={})
    decoder.step(codec.tokenizer.token_to_id(raw))
    parsed = decoder.finish_output()
    assert parsed.response == body
    assert [call.to_tool_call()["function"] for call in parsed.tool_calls] == [
        {"name": "lookup", "arguments": {"count": 1}}
    ]
    assert decoder.tool_calls == (parsed.tool_calls[0].to_tool_call(),)
    assert b"".join(raw.encode()[start:end] for start, end in parsed.response_spans) == body.encode()


@pytest.mark.parametrize("tool_format,prefix,head", _JSON_FRAMING)
@pytest.mark.parametrize("with_tools", [False, True])
def test_native_reasoning_prefill_keeps_json_close_markers_and_utf8_in_the_public_body(
    tool_format: ToolCallFormat, prefix: str, head: str, with_tools: bool
) -> None:
    body = json.dumps({"value": "</think> café ☃️ 東京"}, ensure_ascii=False)
    raw = head + body
    pieces = list(raw)
    codec = codec_for_pieces(tool_format, pieces)
    decoder = codec.decode_stream(prompt=prefix, tools=[TOOL] if with_tools else None, response_schema={})
    streamed = "".join(decoder.step(codec.tokenizer.token_to_id(piece))[1] for piece in pieces)
    parsed = decoder.finish_output()
    assert streamed == parsed.response == body and not parsed.tool_calls
    assert b"".join(raw.encode()[start:end] for start, end in parsed.response_spans) == body.encode()


@pytest.mark.parametrize("tool_format,prefix,head", _JSON_FRAMING)
def test_json_schema_grammar_preserves_native_reasoning_and_masks_invalid_body(
    tool_format: ToolCallFormat, prefix: str, head: str
) -> None:
    codec = codec_for_pieces(tool_format, [])
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    legal = head + '{"answer":1}'
    wrong = head + '{"answer":"wrong"}'
    vocabulary = ["<EOS>", wrong, legal, head + '{"answer":']
    info = xgrammar.TokenizerInfo(vocabulary, stop_token_ids=[0])
    compiled = xgrammar.GrammarCompiler(info).compile_grammar(codec.json_response_grammar(schema, prefix=prefix))
    matcher = xgrammar.GrammarMatcher(compiled)
    if prefix:
        assert matcher.accept_string(prefix)
    mask = np.empty((1, (len(vocabulary) + 31) // 32), dtype=np.int32)
    matcher.fill_next_token_bitmask(mask)
    assert int(mask[0, 0]) & 0b1111 == 0b1100
    assert matcher.accept_token(2) and matcher.is_completed()
    matcher.fill_next_token_bitmask(mask)
    assert int(mask[0, 0]) & 1


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize(
    "schema,body",
    [
        ({"type": "object"}, {"extra": {"nested": 1}}),
        ({"type": "object", "properties": {"optional": {"type": "string"}}}, {}),
        (
            {
                "type": "object",
                "properties": {"nested": {"type": "object", "properties": {"known": {"type": "integer"}}}},
            },
            {"nested": {"extra": 1}, "top_extra": 2},
        ),
    ],
)
def test_json_schema_owns_optional_fields_and_open_object_properties(
    tool_format: ToolCallFormat, schema: dict[str, JSON], body: dict[str, JSON]
) -> None:
    codec = codec_for_pieces(tool_format, [])
    info = xgrammar.TokenizerInfo(["<EOS>"], stop_token_ids=[0])
    compiled = xgrammar.GrammarCompiler(info).compile_grammar(codec.json_response_grammar(schema, prefix=""))
    matcher = xgrammar.GrammarMatcher(compiled)
    assert matcher.accept_string(native_public(tool_format, json.dumps(body))) and matcher.is_completed()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_json_response_makes_progress_under_formatting_whitespace_pressure(tool_format: ToolCallFormat) -> None:
    value = "café" + " " * 40 + "東京"
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": {"text": {"type": "string", "const": value}, "done": {"type": "boolean", "const": True}},
        "required": ["text", "done"],
        "additionalProperties": False,
    }
    beginning = native_public(tool_format, '{"text":' + json.dumps(value, ensure_ascii=False))
    rest = ',"done":true}'
    vocabulary = ["<EOS>", " ", "\n", rest, beginning]
    codec = codec_for_pieces(tool_format, vocabulary[1:])
    info = xgrammar.TokenizerInfo(vocabulary, stop_token_ids=[0])
    compiled = xgrammar.GrammarCompiler(info).compile_grammar(codec.json_response_grammar(schema, prefix=""))
    current = xgrammar.GrammarMatcher(compiled)
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
    decoder = codec.decode_stream(prompt="", response_schema=schema)
    streamed = "".join(decoder.step(codec.tokenizer.token_to_id(piece))[1] for piece in emitted)
    result = decoder.finish_output()
    assert streamed == result.response
    assert json.loads(result.response) == {"text": value, "done": True}
    assert result.tool_calls == () and result.chain_of_thought is None
