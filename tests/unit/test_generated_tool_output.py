import json
from collections.abc import Sequence

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from lalamo.inference.continuous_batching import FinishReason
from lalamo.models.chat_codec import ToolCallFormat, ToolSchema
from tests.unit.test_server import _echo_client

pytestmark = pytest.mark.fast

TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
                "details": {"type": "object"},
                "city": {"type": "string"},
            },
        },
    },
}


def native_call(tool_format: ToolCallFormat, parameter: str, value: str, *, complete: bool) -> str:
    if tool_format is ToolCallFormat.LIQUID:
        text = f"<|tool_call_start|>[lookup({parameter}={value}"
        if complete:
            text += ")]<|tool_call_end|>"
        return text
    if tool_format is ToolCallFormat.QWEN_XML:
        text = f"<tool_call><function=lookup><parameter={parameter}>{value}"
        if complete:
            text += "</parameter></function></tool_call>"
        return text
    text = (
        'to=lookup<|message|><atem:function_calls><atem:invoke name="lookup">'
        f'<atem:parameter name="{parameter}">{value}'
    )
    if complete:
        text += "</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
    return text


def wire_output(body: str, *, stream: bool) -> tuple[str, list[dict], str, list[dict]]:
    if not stream:
        choice = json.loads(body)["choices"][0]
        return (
            choice["message"]["content"] or "",
            choice["message"].get("tool_calls", []),
            choice["finish_reason"],
            (choice["logprobs"] or {}).get("content", []),
        )
    events = [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all("error" not in event for event in events)
    assert body.endswith("data: [DONE]\n\n")
    choices = [choice for event in events for choice in event["choices"]]
    return (
        "".join(choice["delta"].get("content", "") for choice in choices),
        [call for choice in choices for call in choice["delta"].get("tool_calls", [])],
        choices[-1]["finish_reason"],
        [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]],
    )


def request_output(
    monkeypatch: pytest.MonkeyPatch,
    tool_format: ToolCallFormat,
    pieces: Sequence[str],
    *,
    stream: bool,
    stop: str | None = None,
    finish_reason: FinishReason = FinishReason.LENGTH,
) -> tuple[str, list[dict], str, list[dict]]:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens(["prompt", *pieces])
    tokenizer.decoder = Fuse()
    output = tuple(tokenizer.token_to_id(piece) for piece in pieces)
    with _echo_client(
        monkeypatch,
        tokenizer,
        tool_call_format=tool_format,
        prompt_template="prompt",
        output_token_ids=output,
        finish_reason=finish_reason,
    ) as http:
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "call"}],
            "tools": [TOOL],
            "max_completion_tokens": len(pieces),
            "stream": stream,
            "logprobs": True,
        }
        if stop is not None:
            body["stop"] = stop
        response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    return wire_output(response.text, stream=stream)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("value", ["", "-", "-1e", "oops"])
@pytest.mark.parametrize("complete", [False, True])
def test_generated_scalar_prefix_is_returned_without_repair(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool, value: str, complete: bool
) -> None:
    raw = native_call(tool_format, "count", value, complete=complete)
    content, calls, finish, logprobs = request_output(monkeypatch, tool_format, [raw], stream=stream)
    assert content == "" and finish == "length" and logprobs == []
    expected = '{"count":' + value
    if complete:
        expected += "}"
    assert [call["function"] for call in calls] == [{"name": "lookup", "arguments": expected}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("parameter", "value", "arguments"),
    [
        ("active", "Tr", '{"active":tr'),
        ("active", "False", '{"active":false'),
        ("active", "(Tr", '{"active": tr'),
        ("city", "'quoted\\'", '{"city":"quoted\''),
        ("city", "'a\\u12", '{"city":"a\\u12'),
        ("details", "{'x': [True, None", '{"details":{"x": [true, null'),
        ("details", "{'x': [1, {'y': 'é🙂'}]", r'{"details":{"x": [1, {"y": "\u00e9\ud83d\ude42"}]'),
    ],
)
def test_liquid_nested_and_string_prefixes_preserve_observed_values(
    monkeypatch: pytest.MonkeyPatch, stream: bool, parameter: str, value: str, arguments: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, parameter, value, complete=False)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length"
    assert [call["function"] for call in calls] == [{"name": "lookup", "arguments": arguments}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
def test_supplied_raw_stop_precedes_native_parsing_and_preserves_sampled_logprobs(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool, split: bool, malformed: bool
) -> None:
    tail = native_call(tool_format, "count", "oops" if malformed else "2", complete=True)
    before = "beforeST"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        before = "to=user<|message|>beforeST"
        tail = "<|eom|>" + tail
    pieces = [before, "OP" + tail] if split else [before + "OP" + tail]
    content, calls, finish, logprobs = request_output(monkeypatch, tool_format, pieces, stream=stream, stop="STOP")
    assert (content, calls, finish) == ("before", [], "stop")
    assert [entry["bytes"] for entry in logprobs] == [list(pieces[0].encode())]
    assert [entry["logprob"] for entry in logprobs] == [-0.1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_stop_inside_arguments_preserves_partial_call(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool
) -> None:
    raw = native_call(tool_format, "count", "123", complete=True)
    content, calls, finish, _ = request_output(monkeypatch, tool_format, [raw], stream=stream, stop="23")
    assert content == "" and finish == "stop"
    assert [call["function"] for call in calls] == [{"name": "lookup", "arguments": '{"count":1'}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_completed_first_call_and_partial_second_call_are_both_returned(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool
) -> None:
    pieces = [
        native_call(tool_format, "count", "1", complete=True),
        native_call(tool_format, "count", "2", complete=False),
    ]
    content, calls, finish, _ = request_output(monkeypatch, tool_format, pieces, stream=stream)
    assert content == "" and finish == "length"
    assert [call["function"] for call in calls] == [
        {"name": "lookup", "arguments": '{"count":1}'},
        {"name": "lookup", "arguments": '{"count":2'},
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("value", "expected"),
    [(r"r'\n'", r"\n"), (r"'\ud800'", "\ud800"), ("'a' 'b'", "ab"), ("'''line\ntext'''", "line\ntext")],
)
def test_complete_liquid_strings_retain_python_literal_semantics(
    monkeypatch: pytest.MonkeyPatch, stream: bool, value: str, expected: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "city", value, complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length"
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"city": expected}


@pytest.mark.parametrize("stream", [False, True])
def test_complete_liquid_grouped_nested_literals_keep_their_values(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "details", "({'x': [(True), (None), (- ( 2 ))]})", complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length"
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"details": {"x": [True, None, -2]}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("value", "arguments"),
    [
        ("1 2", "1 2"),
        ("1 .2", "1 0.2"),
        ("[1 2]", "[1 2]"),
        ("(1 2)", " 1 2"),
        ("(1)2", " 1 2"),
        ("1(2)", "1 2"),
    ],
)
def test_malformed_liquid_value_tokens_are_never_joined_into_a_valid_value(
    monkeypatch: pytest.MonkeyPatch, stream: bool, value: str, arguments: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "count", value, complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length"
    assert [call["function"] for call in calls] == [{"name": "lookup", "arguments": '{"count":' + arguments + "}"}]
    with pytest.raises(json.JSONDecodeError):
        json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", ["[, ]", "{, }", "()2"])
def test_invalid_liquid_empty_containers_are_never_repaired(
    monkeypatch: pytest.MonkeyPatch, stream: bool, value: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "count", value, complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length" and len(calls) == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("t", ""),
        ("to", ""),
        (" t", ""),
        (" to", ""),
        ("to=", ""),
        ("to ", "to "),
        ("tomorrow", "tomorrow"),
        ("to the store", "to the store"),
    ],
)
def test_muse_budget_cutoff_hides_only_unfinished_recipient_headers(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str, stream: bool
) -> None:
    content, calls, finish, logprobs = request_output(monkeypatch, ToolCallFormat.MUSE_ATEM, [raw], stream=stream)
    assert content == expected and calls == [] and finish == "length"
    assert [entry["token"] for entry in logprobs] == ([raw] if expected else [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_native_tool_output_at_every_budget_cutoff(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool
) -> None:
    raw = native_call(tool_format, "count", "1234", complete=True)
    pieces = [raw[position : position + 8] for position in range(0, len(raw), 8)]
    for budget in range(1, len(pieces) + 1):
        content, calls, finish, _ = request_output(monkeypatch, tool_format, pieces[:budget], stream=stream)
        assert finish == "length"
        for call in calls:
            assert call["function"]["name"] == "lookup"
            assert call["function"]["arguments"].startswith("{")
        if budget == len(pieces):
            assert content == "" and len(calls) == 1
            assert json.loads(calls[0]["function"]["arguments"]) == {"count": 1234}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_natural_stop_with_a_tool_call_has_the_tool_calls_finish_reason(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool
) -> None:
    raw = native_call(tool_format, "count", "2", complete=True)
    content, calls, finish, _ = request_output(
        monkeypatch, tool_format, [raw], stream=stream, finish_reason=FinishReason.STOP
    )
    assert content == "" and finish == "tool_calls"
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"count": 2}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ("count", "[1, # comment\n 2]", {"count": [1, 2]}),
        ("count", "1 # comment\n", {"count": 1}),
        ("count", "1# comment\n", {"count": 1}),
        ("count", "\\\n1\\\n", {"count": 1}),
        ("city", "'a' # comment\n'b'", {"city": "ab"}),
        ("é", "1", {"é": 1}),
        ("\uff45", "1", {"\uff45": 1}),
    ],
)
def test_liquid_generated_layout_and_names_keep_observed_values(
    monkeypatch: pytest.MonkeyPatch, stream: bool, parameter: str, value: str, expected: dict
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, parameter, value, complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length" and len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", ["1# comment\n2", "1\\\n2"])
def test_liquid_layout_keeps_malformed_separate_tokens_malformed(
    monkeypatch: pytest.MonkeyPatch, stream: bool, value: str
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "count", value, complete=True)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length" and len(calls) == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize("stream", [False, True])
def test_partial_liquid_comment_does_not_supply_argument_closures(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    raw = native_call(ToolCallFormat.LIQUID, "count", "[1, # comment", complete=False)
    content, calls, finish, _ = request_output(monkeypatch, ToolCallFormat.LIQUID, [raw], stream=stream)
    assert content == "" and finish == "length"
    assert [call["function"] for call in calls] == [{"name": "lookup", "arguments": '{"count":[1, \n'}]
