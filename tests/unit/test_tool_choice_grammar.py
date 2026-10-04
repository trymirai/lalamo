import ast
import keyword

import numpy as np
import pytest
import xgrammar as xgr
from numpy.typing import NDArray

from lalamo.inference.continuous_batching import FinishReason
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.models.chat_codec import ChatCodec, ToolCallFormat, ToolSchema
from lalamo.utils.json import JSON
from tests.unit.test_chat_codec import _chat_codec
from tests.unit.test_generated_tool_output import TOOL, native_call, request_output

pytestmark = pytest.mark.fast


def tool(name: str) -> ToolSchema:
    function = TOOL["function"]
    assert isinstance(function, dict)
    return {"type": "function", "function": {**function, "name": name}}


def call(tool_format: ToolCallFormat, name: str = "wanted", value: str = "1") -> str:
    return native_call(tool_format, "count", value, complete=True).replace("lookup", name).removesuffix("<|eot|>")


def matcher(
    codec: ChatCodec,
    tools: list[ToolSchema],
    vocabulary: list[str],
    *,
    prefix: str = "",
    require_call: bool = True,
    response_schema: dict[str, JSON] | None = None,
) -> xgr.GrammarMatcher:
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[0])
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        codec.tool_call_grammar(tools, prefix=prefix, require_call=require_call, response_schema=response_schema)
    )
    result = xgr.GrammarMatcher(compiled)
    if prefix:
        assert result.accept_string(prefix)
    return result


def token_mask(matcher: xgr.GrammarMatcher, vocabulary_size: int) -> NDArray[np.int32]:
    result = np.empty((1, (vocabulary_size + 31) // 32), dtype=np.int32)
    matcher.fill_next_token_bitmask(result)
    return result


def allowed(mask: NDArray[np.int32], token_id: int) -> bool:
    return bool(int(mask[0, token_id // 32]) & (1 << (token_id % 32)))


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("named", [False, True])
def test_forced_identity_masks_wrong_names_and_eos_under_greedy_pressure(
    tool_format: ToolCallFormat, named: bool
) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    tools = [tool("wanted"), tool("other")]
    wrong_name = "unlisted"
    if named:
        tools = tools[:1]
        wrong_name = "other"
    good = call(tool_format)
    wrong = call(tool_format, wrong_name)
    preamble = "plain answer "
    if tool_format is ToolCallFormat.MUSE_ATEM:
        preamble = "to=user<|message|>plain answer<|eom|><|start|>assistant"
    vocabulary = ["<EOS>", wrong, good, preamble]
    current = matcher(codec, tools, vocabulary)
    mask = token_mask(current, len(vocabulary))
    assert not allowed(mask, 0) and not allowed(mask, 1) and allowed(mask, 2) and not allowed(mask, 3)
    logits = np.array([100.0, 90.0, 80.0, 70.0])
    allowed_tokens = np.array([allowed(mask, index) for index in range(len(vocabulary))])
    selected = int(np.argmax(np.where(allowed_tokens, logits, -np.inf)))
    assert selected == 2 and current.accept_token(selected)
    assert current.is_completed() and allowed(token_mask(current, len(vocabulary)), 0)
    assert current.accept_token(0) and current.is_terminated()
    message = codec.parse_response(vocabulary[selected], tools=tools)
    assert [entry["function"] for entry in message.tool_calls] == [{"name": "wanted", "arguments": {"count": 1}}]

    current = matcher(codec, tools, vocabulary)
    assert not current.accept_token(3)


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_wrong_call_cannot_be_swallowed_as_preamble_before_the_selected_call(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    right = call(tool_format)
    wrong = call(tool_format, "other")
    separator = ""
    if tool_format is ToolCallFormat.MUSE_ATEM:
        separator = "<|eom|><|start|>assistant"
    current = matcher(codec, [tool("wanted")], ["<EOS>", wrong + separator + right, right])
    assert not allowed(token_mask(current, 3), 1)
    assert not current.accept_token(1)
    assert current.accept_token(2) and current.accept_token(0)


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_empty_native_call_envelopes_cannot_complete_a_required_choice(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    raw = {
        ToolCallFormat.QWEN_XML: "<tool_call></tool_call>",
        ToolCallFormat.LIQUID: "<|tool_call_start|>[]<|tool_call_end|>",
        ToolCallFormat.MUSE_ATEM: "to=wanted<|message|><atem:function_calls></atem:function_calls>",
    }[tool_format]
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert not allowed(token_mask(current, 2), 1)
    assert not current.accept_token(1)


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_required_choice_allows_both_declared_names_in_coherent_native_calls(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    first = call(tool_format)
    second = call(tool_format, "other")
    separator = ""
    if tool_format is ToolCallFormat.MUSE_ATEM:
        separator = "<|eom|><|start|>assistant"
    raw = first + separator + second
    current = matcher(codec, [tool("wanted"), tool("other")], ["<EOS>", raw])
    assert current.accept_token(1) and current.accept_token(0)
    assert [
        entry["function"]["name"]
        for entry in codec.parse_response(raw, tools=[tool("wanted"), tool("other")]).tool_calls
    ] == [
        "wanted",
        "other",
    ]


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_required_grammar_does_not_duplicate_the_serial_function_boundary(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    closer = "</function>"
    if tool_format is ToolCallFormat.LIQUID:
        closer = ")"
    elif tool_format is ToolCallFormat.MUSE_ATEM:
        closer = "</atem:invoke>"
    raw = call(tool_format).split(closer, 1)[0] + closer
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert current.accept_token(1) and not current.is_completed()
    assert not allowed(token_mask(current, 2), 0)
    codec.tokenizer.add_tokens([raw])
    decoder = codec.decode_stream(prompt="", tools=[tool("wanted")], parallel_tool_calls=False)
    decoder.step(codec.tokenizer.token_to_id(raw))
    assert decoder.tool_call_position == len(raw)


@pytest.mark.parametrize("prefix", ["<think>\n", "<think>\n\n</think>\n\n", ""])
@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.LIQUID])
def test_canonical_thinking_prefix_is_consumed_once_before_the_required_call(
    tool_format: ToolCallFormat, prefix: str
) -> None:
    codec = _chat_codec(tool_call_format=tool_format, output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)
    assert codec.decode_stream(prompt=prefix, tools=[tool("wanted")]).prefix == prefix
    continuation = call(tool_format)
    if prefix and "</think>" not in prefix:
        continuation = "reasoning</think>" + continuation
    current = matcher(codec, [tool("wanted")], ["<EOS>", continuation], prefix=prefix)
    assert not allowed(token_mask(current, 2), 0)
    assert current.accept_token(1) and current.accept_token(0)
    assert (
        codec.parse_response(prefix + continuation, tools=[tool("wanted")]).tool_calls[0]["function"]["name"]
        == "wanted"
    )


@pytest.mark.parametrize("recipient", ["self", "user", "wanted"])
def test_muse_prefilled_recipient_remains_the_canonical_grammar_prefix(recipient: str) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    prefix = f"to={recipient}<|message|>"
    assert codec.decode_stream(prompt="<|start|>assistant " + prefix, tools=[tool("wanted")]).prefix == " " + prefix
    continuation = call(ToolCallFormat.MUSE_ATEM)
    if recipient == "wanted":
        continuation = continuation.split("<|message|>", 1)[1]
    else:
        body = "plain" if recipient == "self" else ""
        continuation = body + "<|eom|><|start|>assistant " + continuation
    current = matcher(codec, [tool("wanted")], ["<EOS>", continuation], prefix=prefix)
    assert not allowed(token_mask(current, 2), 0)
    assert current.accept_token(1) and current.accept_token(0)
    assert (
        codec.parse_response(prefix + continuation, tools=[tool("wanted")]).tool_calls[0]["function"]["name"]
        == "wanted"
    )


@pytest.mark.parametrize("recipient", ["wanted", "other"])
def test_muse_recipient_and_invoke_must_match_even_when_both_tools_are_declared(recipient: str) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    name = "wanted"
    if recipient == "wanted":
        name = "other"
    raw = call(ToolCallFormat.MUSE_ATEM, name).replace(f"to={name}", f"to={recipient}", 1)
    current = matcher(codec, [tool("wanted"), tool("other")], ["<EOS>", raw])
    assert not current.accept_token(1)


@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.MUSE_ATEM])
def test_xml_choice_remains_non_strict_about_parameter_names_and_values(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    raw = call(tool_format, value="oops")
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert current.accept_token(1) and current.accept_token(0)
    with pytest.raises(ValueError):
        codec.parse_response(raw, tools=[tool("wanted")])
    raw = raw.replace("count", "extra")
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert current.accept_token(1) and current.accept_token(0)
    assert codec.parse_response(raw, tools=[tool("wanted")]).tool_calls[0]["function"]["arguments"] == {
        "extra": "oops"
    }


@pytest.mark.parametrize(
    "value", ["+2", "-2", "00", "2.", ".2", "02e1", "True", "None", "'é\\n'", '[1, {"city": "Tokyo"}, False, None,]']
)
def test_liquid_complete_literal_arguments_reach_the_existing_strict_parser(value: str) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    raw = call(ToolCallFormat.LIQUID, value=value)
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert current.accept_token(1) and current.accept_token(0)
    assert codec.parse_response(raw, tools=[tool("wanted")]).tool_calls[0]["function"]["arguments"] == {
        "count": ast.literal_eval(value)
    }


@pytest.mark.parametrize("value", ["--2", "- +2", "1 2", "[1 2]", "[,1]", "{,}", "02", "'\\xZ'"])
def test_liquid_malformed_literals_cannot_finish_the_required_call(value: str) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    raw = call(ToolCallFormat.LIQUID, value=value)
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw])
    assert not current.accept_token(1)


def test_liquid_argument_names_accept_keywords_and_preserve_declared_unicode_identity() -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    tools: list[ToolSchema] = [
        {
            "type": "function",
            "function": {
                "name": "wanted",
                "parameters": {
                    "properties": {name: {"type": "integer"} for name in ("café", "\u212a", "é", "\uff45")}
                },
            },
        }
    ]
    info = xgr.TokenizerInfo(["<EOS>"], stop_token_ids=[0])
    compiled = xgr.GrammarCompiler(info).compile_grammar(codec.tool_call_grammar(tools, prefix=""))
    for name in [
        *keyword.kwlist,
        "city",
        "class_",
        "classifier",
        "case",
        "café",
        "extra",
        "Falsehood",
        "some-text",
        "1prop",
        "\u212a",
        "é",
        "\uff45",
    ]:
        raw = f"<|tool_call_start|>[wanted({name}=1)]<|tool_call_end|>"
        current = xgr.GrammarMatcher(compiled)
        assert current.accept_string(raw) and current.is_completed()
        assert codec.parse_response(raw, tools=tools).tool_calls[0]["function"]["arguments"] == {name: 1}


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_optional_subset_allows_text_and_selected_calls_but_never_an_excluded_native_call(
    tool_format: ToolCallFormat,
) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    tools = [tool("wanted")]
    good = call(tool_format)
    wrong = call(tool_format, "other")
    text = "plain answer"
    separator = ""
    if tool_format is ToolCallFormat.MUSE_ATEM:
        text = "to=user<|message|>plain answer"
        separator = "<|eom|><|start|>assistant"
    vocabulary = ["<EOS>", good, wrong, wrong + separator + good, text]
    current = matcher(codec, tools, vocabulary, require_call=False)
    mask = token_mask(current, len(vocabulary))
    assert [allowed(mask, index) for index in range(len(vocabulary))] == [True, True, False, False, True]
    assert current.accept_token(4) and current.accept_token(0)
    assert codec.parse_response(text, tools=tools).response == "plain answer"
    assert codec.parse_response(text, tools=tools).tool_calls == ()

    opener = tool_format.opening_tag
    if tool_format is ToolCallFormat.MUSE_ATEM:
        opener = "to="
    current = matcher(codec, tools, ["<EOS>"], require_call=False)
    assert current.accept_string(opener)
    assert not allowed(token_mask(current, 1), 0)
    assert not current.accept_string(wrong.removeprefix(opener))


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_optional_subset_preserves_text_before_between_and_after_multiple_selected_calls(
    tool_format: ToolCallFormat,
) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    tools = [tool("wanted")]
    first, second = call(tool_format), call(tool_format, value="2")
    raw = "before" + first + "between" + second + "after"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        next_turn = "<|eom|><|start|>assistant"
        raw = (
            "to=user<|message|>before"
            + next_turn
            + first
            + next_turn
            + "to=user<|message|>between"
            + next_turn
            + second
            + next_turn
            + "to=user<|message|>after"
        )
    current = matcher(codec, tools, ["<EOS>", raw], require_call=False)
    assert current.accept_token(1) and current.accept_token(0)
    message = codec.parse_response(raw, tools=tools)
    assert message.response == "beforebetweenafter"
    assert [entry["function"] for entry in message.tool_calls] == [
        {"name": "wanted", "arguments": {"count": 1}},
        {"name": "wanted", "arguments": {"count": 2}},
    ]


@pytest.mark.parametrize("prefix", ["<think>\n", "<think>\n\n</think>\n\n", ""])
@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.LIQUID])
def test_optional_subset_accepts_text_after_the_canonical_thinking_prefix(
    tool_format: ToolCallFormat,
    prefix: str,
) -> None:
    codec = _chat_codec(tool_call_format=tool_format, output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)
    tools = [tool("wanted")]
    prefix = codec.decode_stream(prompt=prefix, tools=tools).prefix
    continuation = "plain answer"
    if prefix and "</think>" not in prefix:
        continuation = "reasoning</think>" + continuation
    current = matcher(codec, tools, ["<EOS>", continuation], prefix=prefix, require_call=False)
    assert allowed(token_mask(current, 2), 0) == (not prefix or "</think>" in prefix)
    assert current.accept_token(1) and current.accept_token(0)
    message = codec.parse_response(prefix + continuation, tools=tools)
    assert message.response == "plain answer" and message.tool_calls == ()


@pytest.mark.parametrize("recipient", ["self", "user", "wanted"])
def test_optional_muse_subset_respects_prefilled_recipient_and_native_turn_boundaries(recipient: str) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    tools = [tool("wanted")]
    prompt = f"<|start|>assistant to={recipient}<|message|>"
    prefix = codec.decode_stream(prompt=prompt, tools=tools).prefix
    continuation = "plain answer"
    if recipient == "wanted":
        continuation = call(ToolCallFormat.MUSE_ATEM).split("<|message|>", 1)[1]
        continuation += "<|eom|><|start|>assistant to=user<|message|>plain answer"
    current = matcher(codec, tools, ["<EOS>", continuation], prefix=prefix, require_call=False)
    assert allowed(token_mask(current, 2), 0) == (recipient != "wanted")
    assert current.accept_token(1) and current.accept_token(0)
    message = codec.parse_response(prefix + continuation, tools=tools)
    if recipient == "self":
        assert message.chain_of_thought == "plain answer" and message.response == ""
    else:
        assert message.response == "plain answer"
    assert len(message.tool_calls) == (recipient == "wanted")


@pytest.mark.parametrize("stream", [False, True])
def test_optional_liquid_empty_native_list_is_a_zero_call_stop(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    raw = "<|tool_call_start|>[]<|tool_call_end|>"
    current = matcher(codec, [tool("wanted")], ["<EOS>", raw], require_call=False)
    assert current.accept_token(1) and current.accept_token(0)
    assert codec.parse_response(raw, tools=[tool("wanted")]).tool_calls == ()
    content, calls, finish, _ = request_output(
        monkeypatch,
        ToolCallFormat.LIQUID,
        [raw],
        stream=stream,
        finish_reason=FinishReason.STOP,
    )
    assert (content, calls, finish) == ("", [], "stop")


@pytest.mark.parametrize("terminator", ["<|eom|>", "<|eot|>", "<|end_of_text|>"])
@pytest.mark.parametrize("with_call", [False, True])
def test_optional_muse_complete_turn_accepts_native_termination_and_only_eom_can_continue(
    terminator: str,
    with_call: bool,
) -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    tools = [tool("wanted")]
    raw = " to=user<|message|>plain answer"
    if with_call:
        raw = call(ToolCallFormat.MUSE_ATEM)
    raw += terminator
    following = raw + "<|start|>assistant to=user<|message|>later"
    current = matcher(codec, tools, ["<EOS>", raw, following], require_call=False)
    mask = token_mask(current, 3)
    assert allowed(mask, 1)
    assert allowed(mask, 2) == (terminator == "<|eom|>")
    assert current.accept_token(1) and current.accept_token(0)
    message = codec.parse_response(raw, tools=tools)
    expected_content = "plain answer"
    if with_call:
        expected_content = ""
    assert message.response == expected_content and len(message.tool_calls) == with_call


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("named", [False, True])
def test_structured_tool_branch_rejects_public_preamble_and_retains_private_reasoning(
    tool_format: ToolCallFormat, named: bool
) -> None:
    codec = _chat_codec(tool_call_format=tool_format, output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)
    tools = [tool("wanted"), tool("other")]
    wrong = "unlisted"
    if named:
        tools = tools[:1]
        wrong = "other"
    good = call(tool_format)
    public = "public text "
    private = "<think>private</think>\n"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        public = "to=user<|message|>public text<|eom|><|start|>assistant"
        private = "to=self<|message|>private<|eom|><|start|>assistant"
    vocabulary = ["<EOS>", public + good, private + good, good, call(tool_format, wrong)]
    current = matcher(codec, tools, vocabulary, response_schema={"type": "object"})
    mask = token_mask(current, len(vocabulary))
    assert [allowed(mask, i) for i in range(len(vocabulary))] == [False, False, True, True, False]
    assert current.accept_token(2) and current.is_completed()
    message = codec.parse_response(vocabulary[2], tools=tools)
    assert message.response == "" and message.chain_of_thought == "private"
    assert [entry["function"]["name"] for entry in message.tool_calls] == ["wanted"]


def test_structured_prefilled_muse_public_turn_must_close_empty_before_real_call() -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    good = call(ToolCallFormat.MUSE_ATEM)
    tail = "<|eom|><|start|>assistant" + good
    vocabulary = ["<EOS>", "public text" + tail, tail]
    current = matcher(codec, [tool("wanted")], vocabulary, prefix="to=user<|message|>", response_schema={})
    assert [allowed(token_mask(current, len(vocabulary)), i) for i in range(len(vocabulary))] == [False, False, True]
    assert current.accept_token(2) and current.is_completed()


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_structured_auto_union_requires_json_public_body_or_a_real_selected_call(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(tool_call_format=tool_format, output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)
    schema: dict[str, JSON] = {"type": "object", "additionalProperties": True}
    json_body = '{"answer":1}'
    plain = "plain text"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        json_body = "to=user<|message|>" + json_body
        plain = "to=user<|message|>" + plain
    vocabulary = ["<EOS>", plain, json_body, call(tool_format), call(tool_format, "unlisted")]
    grammar = xgr.Grammar.union(
        codec.json_response_grammar(schema, prefix=""),
        xgr.Grammar.from_ebnf(codec.tool_call_grammar([tool("wanted")], prefix="", response_schema=schema)),
    )
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(vocabulary, stop_token_ids=[0])).compile_grammar(grammar)
    current = xgr.GrammarMatcher(compiled)
    assert [allowed(token_mask(current, len(vocabulary)), i) for i in range(len(vocabulary))] == [
        False,
        False,
        True,
        True,
        False,
    ]
