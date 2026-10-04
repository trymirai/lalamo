import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import pytest
from cattrs.errors import ClassValidationError
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, ByteLevel, Decoder, Fuse
from tokenizers.models import BPE, WordLevel

from lalamo.data.huggingface_message import HFConversation, load_hf_parquet
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import (
    GEMMA4_OUTPUT_PARSER_REGEX,
    GRANITE_THINKING_OUTPUT_PARSER_REGEX,
    OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
)
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.model_registry import ModelRegistry
from lalamo.models.chat_codec import (
    AssistantMessage,
    ChatCodec,
    ChatCodecConfig,
    ReasoningConfig,
    ReasoningEffort,
    ToolCall,
    ToolCallFormat,
    ToolSchema,
    UserMessage,
    parse_hf_message,
)


def _chat_codec(
    *,
    prompt_template: str = "",
    output_parser_regex: str | None = None,
    reasoning_config: ReasoningConfig | None = None,
    tool_call_format: ToolCallFormat | None = None,
) -> ChatCodec:
    config = ChatCodecConfig(
        prompt_template=prompt_template,
        output_parser_regex=output_parser_regex,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        reasoning_config=reasoning_config,
        tool_call_format=tool_call_format,
    )
    return ChatCodecConfig.from_json(config.to_json()).init(
        Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    )


@pytest.mark.parametrize(
    "missing_field", [None, "default_reasoning_effort", "field_name", "reasoning_effort_to_field_value"]
)
def test_loading_rejects_incomplete_or_unmapped_reasoning_effort(missing_field: str | None) -> None:
    config = _chat_codec().config.to_json()
    assert isinstance(config, dict)
    reasoning: dict = {
        "default_reasoning_effort": "medium",
        "field_name": "reasoning_effort",
        "reasoning_effort_to_field_value": {"medium": "medium"},
    }
    if missing_field is None:
        reasoning["default_reasoning_effort"] = "high"
    else:
        reasoning.pop(missing_field)
    config["reasoning_config"] = reasoning
    with pytest.raises(ClassValidationError):
        ChatCodecConfig.from_json(config)


def test_reasoning_effort_is_rendered_through_the_configured_field() -> None:
    codec = _chat_codec(
        prompt_template="{{ reasoning_effort }}",
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.MEDIUM: "medium",
                }
            ),
        ),
    )

    assert codec.render_request([UserMessage("hello")]) == "medium"
    assert codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.LOW) == "low"

    with pytest.raises(ValueError, match="not supported"):
        codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.HIGH)


def test_model_without_reasoning_config_rejects_an_explicit_effort() -> None:
    codec = _chat_codec(prompt_template="{{ reasoning_effort | default('unset') }}")

    assert codec.render_request([UserMessage("hello")]) == "unset"
    with pytest.raises(ValueError, match="does not support configurable reasoning effort"):
        codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.MEDIUM)


def test_boolean_template_field_uses_medium_as_enabled() -> None:
    codec = _chat_codec(
        prompt_template="{% if enable_thinking %}on{% else %}off{% endif %}",
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )

    assert codec.render_request([UserMessage("hello")]) == "on"
    assert (
        codec.render_request(
            [UserMessage("hello")],
            reasoning_effort=ReasoningEffort.NO_REASONING,
        )
        == "off"
    )


def test_mixture_parquet_row_renders_as_huggingface_request(tmp_path: Path) -> None:
    calls = [{"type": "function", "id": "call_1", "function": {"name": "add", "arguments": {"a": 1}}}]
    tools = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "user", "content": "Add."},
        {"role": "assistant", "content": "", "reasoning_content": "Think.", "tool_calls": calls},
        {"role": "tool", "content": "1", "tool_call_id": "call_1"},
    ]
    stored_calls = [{**calls[0], "function": {"name": "add", "arguments": json.dumps({"a": 1})}}]
    stored_messages = [
        {"role": "user", "content": "Add.", "reasoning": None, "tool_calls": None, "tool_call_id": None},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Think.",
            "tool_calls": json.dumps(stored_calls),
            "tool_call_id": None,
        },
        {"role": "tool", "content": "1", "reasoning": None, "tool_calls": None, "tool_call_id": "call_1"},
    ]
    path = tmp_path / "mixture.parquet"
    row = {"messages": stored_messages, "tools": json.dumps(tools), "metadata": {"source_row": 0}}
    pl.DataFrame([row]).write_parquet(path)

    (loaded,) = load_hf_parquet(path).collect().to_dicts()
    conversation = HFConversation.from_dict(loaded)
    codec = _chat_codec(prompt_template='{{ {"messages": messages, "tools": tools} | tojson }}')
    assert json.loads(codec.render_request(conversation.messages, tools=conversation.tools)) == {
        "messages": messages,
        "tools": tools,
    }
    assert json.loads(codec.render_request(conversation.messages))["tools"] is None


@pytest.mark.parametrize("tool_calls", [None, []])
def test_rendered_request_preserves_empty_text_and_omits_absent_fields(tool_calls: list | None) -> None:
    messages: list[dict] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "", "reasoning_content": ""},
        {"role": "tool", "content": "", "name": "", "tool_call_id": ""},
        {"role": "tool", "content": ""},
    ]
    source = [*messages]
    source[2] = {**source[2], "tool_calls": tool_calls}
    codec = _chat_codec(prompt_template="{{ messages | tojson }}")
    assert json.loads(codec.render_request([parse_hf_message(message) for message in source])) == messages


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "user", "content": 123},
        {"role": "assistant", "content": {"text": "wrong shape"}},
        {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "example"}}]},
        {"role": "assistant", "name": "invalid"},
        {"role": "assistant", "reasoning": "one", "thinking": "two"},
        {"role": "tool", "reasoning": "invalid"},
        {"role": "user", "tool_calls": []},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "[]"}}]},
    ],
)
def test_message_rejects_invalid_wire_fields(payload: dict) -> None:
    with pytest.raises((TypeError, ClassValidationError)):
        parse_hf_message(payload)


@pytest.mark.parametrize(
    ("output_parser_regex", "full_output"),
    [
        (OPTIONAL_THINKING_OUTPUT_PARSER_REGEX, "<think>reasoning</think>answer"),
        (GRANITE_THINKING_OUTPUT_PARSER_REGEX, "<think>reasoning</think><response>answer</response>"),
        (GEMMA4_OUTPUT_PARSER_REGEX, "<|channel>thought\nreasoning<channel|>answer<turn|>"),
    ],
    ids=["optional-thinking", "granite", "gemma4"],
)
def test_generation_is_parsed_at_every_truncation_stage(output_parser_regex: str, full_output: str) -> None:
    codec = _chat_codec(output_parser_regex=output_parser_regex)
    mid_thinking = full_output[: full_output.index("reasoning") + len("reas")]
    mid_response = full_output[: full_output.index("answer") + len("answ")]

    assert codec.parse_response("answer") == AssistantMessage(chain_of_thought=None, response="answer")
    assert codec.parse_response(mid_thinking) == AssistantMessage(chain_of_thought="reas", response="")
    assert codec.parse_response(mid_response) == AssistantMessage(chain_of_thought="reasoning", response="answ")
    assert codec.parse_response(full_output) == AssistantMessage(chain_of_thought="reasoning", response="answer")


def test_optional_thinking_parses_response_without_an_opening_tag() -> None:
    codec = _chat_codec(output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)

    expected = AssistantMessage(chain_of_thought="reasoning", response="answer")
    assert codec.parse_response("reasoning</think>answer") == expected


def test_granite_parses_a_response_missing_its_wrapper() -> None:
    codec = _chat_codec(output_parser_regex=GRANITE_THINKING_OUTPUT_PARSER_REGEX)

    expected = AssistantMessage(chain_of_thought="reasoning", response="answer")
    assert codec.parse_response("<think>reasoning</think>answer") == expected


def _registered_codec(repo: str) -> ChatCodec:
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[repo]
    assert isinstance(spec, LanguageModelSpec)
    return _chat_codec(output_parser_regex=spec.output_parser_regex, tool_call_format=spec.tool_call_format)


def _stream_every_split(
    codec: ChatCodec, raw: str, expected: AssistantMessage, *, prompt: str = "", tools: Iterable[ToolSchema] = ()
) -> None:
    """Streams `raw` split into tokens in many ways; released text may only grow into the expected message."""
    partitions = [(raw,), tuple(raw)] + [(raw[:offset], raw[offset:]) for offset in range(1, len(raw))]
    for parts in partitions:
        vocabulary = {piece: index for index, piece in enumerate(dict.fromkeys(parts))}
        tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
        tokenizer.decoder = Fuse()
        stream = codec.config.init(tokenizer).decode_stream(prompt, tools=tools)
        reasoning = response = ""
        for piece in parts:
            reasoning_piece, response_piece = stream.step(vocabulary[piece])
            reasoning += reasoning_piece
            response += response_piece
            assert (expected.chain_of_thought or "").startswith(reasoning)
            assert expected.response.startswith(response)
        reasoning_piece, response_piece, message = stream.finish()
        assert (reasoning + reasoning_piece, response + response_piece) == (
            expected.chain_of_thought or "",
            expected.response,
        )
        assert message == expected


@pytest.mark.parametrize(
    ("repo", "full_output", "stop_suffix", "prompt", "reasoning"),
    [
        (
            "meta-models/Muse-Glimmer-30B",
            "to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            "<|eot|>",
            "",
            "reasoning",
        ),
        ("meta-models/Muse-Glimmer-30B", "to=user<|message|>answer<|eot|>", "<|eot|>", "", None),
        ("Qwen/Qwen3.8-27B", "reasoning\n</think>\n\nanswer", "", "<think>\n", "reasoning\n"),
        ("Qwen/Qwen3.8-27B", "reasoning</think>answer", "", "<think>\n", "reasoning"),
        ("Qwen/Qwen3.5-9B", "answer", "", "<think>\n\n</think>\n\n", None),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "<think>reasoning</think>answer", "", "", "reasoning"),
        (
            "ibm-granite/granite-3.3-2b-instruct",
            "<think>reasoning</think><response>answer</response>",
            "",
            "",
            "reasoning",
        ),
        ("google/gemma-4-E2B-it", "<|channel>thought\nreasoning<channel|>answer<turn|>", "<turn|>", "", "reasoning"),
        ("google/gemma-4-E2B-it", "answer<turn|>", "<turn|>", "", None),
        (
            "openai/gpt-oss-20b",
            "<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>"
            "answer<|return|>",
            "<|return|>",
            "",
            "reasoning",
        ),
        ("openai/gpt-oss-20b", "<|channel|>final<|message|>answer<|return|>", "<|return|>", "", None),
    ],
)
def test_registered_chat_formats_stream_only_parsed_text(
    repo: str, full_output: str, stop_suffix: str, prompt: str, reasoning: str | None
) -> None:
    codec = _registered_codec(repo)
    expected = AssistantMessage(chain_of_thought=reasoning, response="answer")
    assert codec.parse_response(full_output) == expected
    _stream_every_split(codec, full_output.removesuffix(stop_suffix), expected, prompt=prompt)


_TOOLS: tuple[ToolSchema, ...] = (
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "data": {"type": "object"},
                    "missing": {"type": "null"},
                },
            },
        },
    },
)
_CALL: ToolCall = {
    "type": "function",
    "function": {
        "name": "lookup",
        "arguments": {
            "text": "  null é🙂  ",
            "count": -2,
            "enabled": True,
            "data": {"flags": [True, None, False]},
            "missing": None,
        },
    },
}
_QWEN_CALL = (
    "<tool_call>\n<function=lookup>\n<parameter=text>\n  null é🙂  \n</parameter>\n"
    "<parameter=count>\n-2\n</parameter>\n<parameter=enabled>\nTrue\n</parameter>\n"
    '<parameter=data>\n{"flags": [true, null, false]}\n</parameter>\n'
    "<parameter=missing>\nNone\n</parameter>\n</function>\n</tool_call>"
)
_LIQUID_CALL = (
    '<|tool_call_start|>[lookup(text="  null é🙂  ", count=-2, enabled=True, '
    'data={"flags": [true, null, false]}, missing=None)]<|tool_call_end|>'
)
_MUSE_CALL = (
    'to=lookup<|message|><atem:function_calls>\n<atem:invoke name="lookup">\n'
    '<atem:parameter name="text">  null é🙂  </atem:parameter>\n'
    '<atem:parameter name="count">-2</atem:parameter>\n'
    '<atem:parameter name="enabled">true</atem:parameter>\n'
    '<atem:parameter name="data">{"flags": [true, null, false]}</atem:parameter>\n'
    '<atem:parameter name="missing">null</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


@pytest.mark.parametrize(
    "repo",
    [
        "Qwen/Qwen3.5-0.8B",
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LiquidAI/LFM2.5-1.2B-Thinking",
        "meta-models/Muse-Glimmer-30B",
    ],
)
def test_native_tool_calls_parse_and_stream_every_split(repo: str) -> None:
    codec = _registered_codec(repo)
    match codec.config.tool_call_format:
        case ToolCallFormat.QWEN_XML | ToolCallFormat.LIQUID as tool_format:
            call = _QWEN_CALL if tool_format is ToolCallFormat.QWEN_XML else _LIQUID_CALL
            raw = "before" + call + call + "after"
            reasoning = None
            if codec.config.output_parser_regex is not None:
                raw = "<think>reasoning</think>" + raw
                reasoning = "reasoning"
        case _:
            turn = "<|eom|><|start|>assistant "
            raw = (
                f"to=self<|message|>reasoning{turn}to=user<|message|>before{turn}{_MUSE_CALL}{turn}{_MUSE_CALL}"
                f"{turn}to=user<|message|>after"
            )
            reasoning = "reasoning"
    expected = AssistantMessage(chain_of_thought=reasoning, response="beforeafter", tool_calls=(_CALL, _CALL))
    assert codec.parse_response(raw, tools=_TOOLS) == expected
    assert codec.parse_response(raw).tool_calls == ()
    _stream_every_split(codec, raw, expected, tools=_TOOLS)


@pytest.mark.parametrize(
    ("tool_format", "raw"),
    [
        (ToolCallFormat.QWEN_XML, "<tool_call><function=lookup>"),
        (ToolCallFormat.QWEN_XML, "<tool_call>lookup</tool_call>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup("),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(text=evil())]<|tool_call_end|>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(**data)]<|tool_call_end|>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(text='x', text='y')]<|tool_call_end|>"),
    ],
)
def test_malformed_tool_calls_remain_text(tool_format: ToolCallFormat, raw: str) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    assert codec.parse_response(raw, tools=_TOOLS) == AssistantMessage(response=raw)


@pytest.mark.parametrize(
    ("tool_format", "call"), [(ToolCallFormat.QWEN_XML, _QWEN_CALL), (ToolCallFormat.LIQUID, _LIQUID_CALL)]
)
def test_tool_examples_inside_reasoning_are_never_called(tool_format: ToolCallFormat, call: str) -> None:
    codec = _chat_codec(output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX, tool_call_format=tool_format)
    assert codec.parse_response("<think>" + call + "</think>answer", tools=_TOOLS) == AssistantMessage(
        chain_of_thought=call, response="answer"
    )


def test_liquid_calls_accept_hyphenated_openai_names() -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    tools: tuple[ToolSchema, ...] = ({"type": "function", "function": {"name": "get-weather"}},)
    message = codec.parse_response("<|tool_call_start|>[get-weather(city='Paris')]<|tool_call_end|>", tools=tools)
    assert message.tool_calls == (
        {"type": "function", "function": {"name": "get-weather", "arguments": {"city": "Paris"}}},
    )


def test_stream_stops_before_a_stop_string() -> None:
    raw = "abcSTOPdef"
    vocabulary = {character: index for index, character in enumerate(dict.fromkeys(raw))}
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    tokenizer.decoder = Fuse()
    stream = _chat_codec().config.init(tokenizer).decode_stream("", stop_strings=("STOP",))
    released = "".join(stream.step(vocabulary[character])[1] for character in raw[: raw.index("P") + 1])
    assert stream.stopped
    assert released + stream.finish()[1] == "abc"


def test_liquid_hyphenated_names_are_only_replaced_in_call_positions() -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    tools: tuple[ToolSchema, ...] = ({"type": "function", "function": {"name": "get-weather"}},)
    raw = """<|tool_call_start|>[get-weather(text="x, get-weather(")]<|tool_call_end|>"""
    (call,) = codec.parse_response(raw, tools=tools).tool_calls
    assert call["function"] == {"name": "get-weather", "arguments": {"text": "x, get-weather("}}


@pytest.mark.parametrize(
    ("vocabulary", "decoder"),
    [({"<0xC3>": 0, "<0xA9>": 1}, ByteFallback()), ({"Ã": 0, "©": 1}, ByteLevel())],
)
def test_token_bytes_keep_partial_utf8_sequences(vocabulary: dict[str, int], decoder: Decoder) -> None:
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    tokenizer.decoder = decoder
    codec = _chat_codec().config.init(tokenizer)
    assert [codec.decode_token_bytes(token_id) for token_id in (0, 1)] == [b"\xc3", b"\xa9"]
