import ast
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.models.chat_codec import ToolCallFormat
from lalamo.utils.json import JSON
from tests.unit.test_chat_codec import _chat_codec
from tests.unit.test_generated_tool_output import TOOL, native_call
from tests.unit.test_server import _echo_client

pytestmark = pytest.mark.fast


def multiple_calls(tool_format: ToolCallFormat, *, second_complete: bool) -> str:
    first = native_call(tool_format, "count", "1", complete=True)
    second = native_call(tool_format, "count", "oops", complete=second_complete)
    if tool_format is ToolCallFormat.LIQUID:
        return first.removesuffix("]<|tool_call_end|>") + "," + second.split("[", 1)[1]
    if tool_format is ToolCallFormat.QWEN_XML:
        return first.removesuffix("</tool_call>") + second.removeprefix("<tool_call>")
    return first.removesuffix("</atem:function_calls><|eot|>") + second.split("<atem:function_calls>", 1)[1]


def serial_response(
    monkeypatch: pytest.MonkeyPatch,
    tool_format: ToolCallFormat,
    pieces: list[str],
    *,
    stream: bool,
    parallel: bool | None = False,
    budget: int = 32,
    stop: str | None = None,
    n: int = 1,
    prompt: str = "prompt",
    output_parser_regex: str | None = None,
) -> tuple[list[dict], dict]:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens([prompt, *pieces])
    tokenizer.decoder = Fuse()
    with _echo_client(
        monkeypatch,
        tokenizer,
        tool_call_format=tool_format,
        prompt_template=prompt,
        output_parser_regex=output_parser_regex,
        output_token_ids=tuple(tokenizer.token_to_id(piece) for piece in pieces),
    ) as http:
        body: dict[str, JSON] = {
            "model": "test",
            "messages": [{"role": "user", "content": "call"}],
            "tools": [TOOL],
            "max_completion_tokens": budget,
            "stream": stream,
            "n": n,
            "logprobs": True,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if parallel is not None:
            body["parallel_tool_calls"] = parallel
        if stop is not None:
            body["stop"] = stop
        response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if not stream:
        result = response.json()
        return result["choices"], result["usage"]
    assert response.text.endswith("data: [DONE]\n\n")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all("error" not in event for event in events)
    choices = []
    for index in range(n):
        chunks = [choice for event in events for choice in event["choices"] if choice["index"] == index]
        choices.append(
            {
                "index": index,
                "message": {
                    "content": "".join(chunk["delta"].get("content", "") for chunk in chunks),
                    "tool_calls": [call for chunk in chunks for call in chunk["delta"].get("tool_calls", [])],
                },
                "finish_reason": chunks[-1]["finish_reason"],
                "logprobs": {
                    "content": [
                        entry for chunk in chunks if chunk["logprobs"] for entry in chunk["logprobs"]["content"]
                    ]
                },
            }
        )
    return choices, next(event["usage"] for event in events if event.get("usage") is not None)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("second_complete", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_serial_tool_calls_stop_at_the_first_completed_native_function(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    tool_format: ToolCallFormat,
    second_complete: bool,
    split: bool,
    n: int,
) -> None:
    raw = multiple_calls(tool_format, second_complete=second_complete)
    if tool_format is ToolCallFormat.MUSE_ATEM:
        raw = raw.replace("<|message|>", "<|message|>\n  ", 1)
    pieces = [raw]
    if split:
        pieces = [raw[position : position + 7] for position in range(0, len(raw), 7)]
    choices, usage = serial_response(monkeypatch, tool_format, pieces, stream=stream, n=n)
    for choice in choices:
        assert choice["message"]["content"] in (None, "") and choice["finish_reason"] == "tool_calls"
        assert [call["function"] for call in choice["message"]["tool_calls"]] == [
            {"name": "lookup", "arguments": '{"count":1}'}
        ]
    function_end = "</function>"
    if tool_format is ToolCallFormat.LIQUID:
        function_end = ")"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        function_end = "</atem:invoke>"
    source_end = raw.index(function_end) + len(function_end)
    expected_tokens = 1
    if split:
        expected_tokens = (source_end + 6) // 7
    assert usage["completion_tokens"] == n * expected_tokens


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("parallel", [None, True])
def test_default_parallel_behavior_keeps_both_generated_calls(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat, parallel: bool | None
) -> None:
    choices, _ = serial_response(
        monkeypatch, tool_format, [multiple_calls(tool_format, second_complete=True)], stream=stream, parallel=parallel
    )
    assert choices[0]["finish_reason"] == "length"
    assert [call["function"]["arguments"] for call in choices[0]["message"]["tool_calls"]] == [
        '{"count":1}',
        '{"count":oops}',
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("budget", [1, 32])
@pytest.mark.parametrize("before_call", [False, True])
def test_the_earlier_raw_stop_or_native_call_limit_determines_the_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    tool_format: ToolCallFormat,
    budget: int,
    before_call: bool,
) -> None:
    raw = multiple_calls(tool_format, second_complete=False)
    if before_call:
        raw = raw.replace("1", "1STOP", 1)
    else:
        raw += "STOP"
    choices, _ = serial_response(monkeypatch, tool_format, [raw], stream=stream, budget=budget, stop="STOP")
    choice = choices[0]
    expected = "tool_calls"
    if budget == 1:
        expected = "length"
    if before_call:
        expected = "stop"
    assert choice["finish_reason"] == expected
    arguments = '{"count":1}'
    if before_call:
        arguments = '{"count":1'
    assert [call["function"]["arguments"] for call in choice["message"]["tool_calls"]] == [arguments]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_serial_clipping_preserves_sampled_token_byte_and_logprob_ownership(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat
) -> None:
    raw = "é🙂before" + multiple_calls(tool_format, second_complete=True)
    if tool_format is ToolCallFormat.MUSE_ATEM:
        raw = "to=user<|message|>é🙂before<|eom|>" + multiple_calls(tool_format, second_complete=True)
    choices, _ = serial_response(monkeypatch, tool_format, [raw], stream=stream)
    choice = choices[0]
    assert choice["message"]["content"] == "é🙂before" and choice["finish_reason"] == "tool_calls"
    assert [entry["bytes"] for entry in choice["logprobs"]["content"]] == [list(raw.encode())]
    assert [entry["logprob"] for entry in choice["logprobs"]["content"]] == [-0.1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_call_limit_source_offsets_include_the_actual_prefilled_header_and_response_group(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat
) -> None:
    raw = "reasoning</think>ébefore" + multiple_calls(tool_format, second_complete=False)
    prompt = "<think>"
    regex = OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
    expected_content = "ébefore"
    function_end = "</function>"
    if tool_format is ToolCallFormat.LIQUID:
        function_end = ")"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        prompt = "<|start|>assistant to=lookup<|message|>"
        raw = "\n  " + multiple_calls(tool_format, second_complete=False).split("<|message|>", 1)[1]
        expected_content = ""
        function_end = "</atem:invoke>"
    pieces = [raw[position : position + 7] for position in range(0, len(raw), 7)]
    choices, usage = serial_response(
        monkeypatch, tool_format, pieces, stream=stream, prompt=prompt, output_parser_regex=regex
    )
    assert (choices[0]["message"]["content"] or "") == expected_content
    assert choices[0]["finish_reason"] == "tool_calls"
    assert [call["function"] for call in choices[0]["message"]["tool_calls"]] == [
        {"name": "lookup", "arguments": '{"count":1}'}
    ]
    source_end = raw.index(function_end) + len(function_end)
    assert usage["completion_tokens"] == (source_end + 6) // 7


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("complete", [False, True])
def test_serial_call_limit_keeps_malformed_or_budget_truncated_first_arguments(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat, complete: bool
) -> None:
    value = "1"
    budget = 1
    expected_reason = "length"
    arguments = '{"count":1'
    if complete:
        value = "oops"
        budget = 32
        expected_reason = "tool_calls"
        arguments = '{"count":oops}'
    raw = native_call(tool_format, "count", value, complete=complete)
    choices, usage = serial_response(monkeypatch, tool_format, [raw], stream=stream, budget=budget)
    assert choices[0]["finish_reason"] == expected_reason and usage["completion_tokens"] == 1
    assert [call["function"]["arguments"] for call in choices[0]["message"]["tool_calls"]] == [arguments]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("output_parser_regex", [None, OPTIONAL_THINKING_OUTPUT_PARSER_REGEX])
def test_first_completed_empty_native_function_stops_serial_generation(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    tool_format: ToolCallFormat,
    output_parser_regex: str | None,
) -> None:
    raw = {
        ToolCallFormat.LIQUID: "<|tool_call_start|>[lookup(),lookup(count=oops",
        ToolCallFormat.QWEN_XML: "<tool_call><function=lookup></function><function=lookup><parameter=count>oops",
        ToolCallFormat.MUSE_ATEM: (
            'to=lookup<|message|>\n  <atem:function_calls><atem:invoke name="lookup"></atem:invoke>'
            '<atem:invoke name="lookup"><atem:parameter name="count">oops'
        ),
    }[tool_format]
    raw = "\n  " + raw
    choices, usage = serial_response(
        monkeypatch, tool_format, [raw, "later output"], stream=stream, output_parser_regex=output_parser_regex
    )
    assert choices[0]["finish_reason"] == "tool_calls" and usage["completion_tokens"] == 1
    assert [call["function"] for call in choices[0]["message"]["tool_calls"]] == [
        {"name": "lookup", "arguments": "{}"}
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.LIQUID])
@pytest.mark.parametrize("parallel", [False, True])
def test_unmarked_private_tool_examples_stay_reasoning_until_the_channel_closes(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat, parallel: bool
) -> None:
    private = "private " + native_call(tool_format, "count", "1", complete=True)
    choices, usage = serial_response(
        monkeypatch,
        tool_format,
        [private, "</think>answer"],
        stream=stream,
        parallel=parallel,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
    )
    assert choices[0]["message"]["content"] == "answer"
    assert not choices[0]["message"].get("tool_calls")
    assert usage["completion_tokens"] == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_malformed_native_parameter_before_a_function_closer_does_not_trigger_the_cap(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_format: ToolCallFormat
) -> None:
    raw = {
        ToolCallFormat.LIQUID: "<|tool_call_start|>[lookup(city='oops)]<|tool_call_end|>",
        ToolCallFormat.QWEN_XML: "<tool_call><function=lookup><parameter=city>oops</function></tool_call>",
        ToolCallFormat.MUSE_ATEM: (
            'to=lookup<|message|><atem:function_calls><atem:invoke name="lookup">'
            '<atem:parameter name="city">oops</atem:invoke></atem:function_calls>'
        ),
    }[tool_format]
    choices, usage = serial_response(monkeypatch, tool_format, [raw], stream=stream)
    assert choices[0]["finish_reason"] == "length" and usage["completion_tokens"] == 1
    calls = choices[0]["message"]["tool_calls"]
    assert len(calls) == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("+2", True),
        ("-2", True),
        ("-(\n2)", True),
        ("+(2)", True),
        ("--2", False),
        ("---2", False),
        ("- +2", False),
        ("- -2", False),
        ("-(-2)", False),
        ("+(+2)", False),
        ("1+ (2)", False),
    ],
)
def test_liquid_numeric_signs_match_the_complete_python_argument_literal(value: str, valid: bool) -> None:
    expression = ast.parse(f"lookup(count={value})", mode="eval").body
    assert isinstance(expression, ast.Call)
    argument = expression.keywords[0].value
    raw = native_call(ToolCallFormat.LIQUID, "count", value, complete=True)
    if valid:
        expected = ast.literal_eval(argument)
        message = _chat_codec(tool_call_format=ToolCallFormat.LIQUID).parse_response(raw, tools=[TOOL])
        assert message.tool_calls[0]["function"]["arguments"] == {"count": expected}
    else:
        with pytest.raises(ValueError):
            ast.literal_eval(argument)
        with pytest.raises(ValueError):
            _chat_codec(tool_call_format=ToolCallFormat.LIQUID).parse_response(raw, tools=[TOOL])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", ["--2", "---2", "- +2", "- -2", "-(-2)", "+(+2)", "1+ (2)"])
def test_liquid_repeated_or_binary_signs_remain_malformed_in_generated_arguments(
    monkeypatch: pytest.MonkeyPatch, stream: bool, value: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "count", value, complete=True)
    choices, _ = serial_response(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    with pytest.raises(json.JSONDecodeError):
        json.loads(choices[0]["message"]["tool_calls"][0]["function"]["arguments"])


@pytest.mark.parametrize("stream", [False, True])
def test_liquid_keyword_function_headers_trigger_the_serial_cap(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    choices, usage = serial_response(
        monkeypatch, ToolCallFormat.LIQUID, ["<|tool_call_start|>[class(count=1)]<|tool_call_end|>"], stream=stream
    )
    assert choices[0]["finish_reason"] == "tool_calls" and usage["completion_tokens"] == 1
    assert choices[0]["message"]["tool_calls"][0]["function"] == {"name": "class", "arguments": '{"count":1}'}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("second_complete", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_an_incomplete_first_call_hides_later_envelopes_without_stopping_generation(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    tool_format: ToolCallFormat,
    second_complete: bool,
    split: bool,
) -> None:
    function_end = "</function>"
    first_arguments = '{"count":1'
    if tool_format is ToolCallFormat.LIQUID:
        function_end = ")"
        first_arguments += "]"
    elif tool_format is ToolCallFormat.MUSE_ATEM:
        function_end = "</atem:invoke>"
    first = native_call(tool_format, "count", "1", complete=True).replace(function_end, "", 1)
    raw = first + native_call(tool_format, "count", "2", complete=second_complete)
    pieces = [raw]
    if split:
        pieces = [raw[position : position + 7] for position in range(0, len(raw), 7)]
    choices, usage = serial_response(monkeypatch, tool_format, pieces, stream=stream, budget=len(pieces) + 1)
    assert choices[0]["finish_reason"] == "length" and usage["completion_tokens"] == len(pieces)
    assert [call["function"] for call in choices[0]["message"]["tool_calls"]] == [
        {"name": "lookup", "arguments": first_arguments}
    ]
