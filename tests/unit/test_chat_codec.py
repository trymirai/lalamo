import json
from pathlib import Path

import polars as pl
import pytest
from cattrs.errors import ClassValidationError
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, ByteLevel, Decoder
from tokenizers.models import BPE, WordLevel

from lalamo.data.huggingface_message import HFConversation, load_hf_parquet
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.gemma import Gemma4ResponseParser
from lalamo.model_import.model_specs.granite import GraniteResponseParser
from lalamo.model_import.model_specs.lfm2 import LiquidResponseParser, LiquidThinkingResponseParser
from lalamo.model_import.model_specs.output_parsers import ThinkingResponseParser
from lalamo.model_import.model_specs.qwen import QwenResponseParser
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.model_registry import ModelRegistry
from lalamo.models.chat_codec import (
    AssistantMessage,
    ChatCodec,
    ChatCodecConfig,
    ReasoningConfig,
    ReasoningEffort,
    ResponseParser,
    ToolCall,
    ToolSchema,
    UserMessage,
    parse_hf_message,
)


def _chat_codec(
    *,
    prompt_template: str = "",
    response_parser: type[ResponseParser] | None = None,
    reasoning_config: ReasoningConfig | None = None,
) -> ChatCodec:
    config = ChatCodecConfig(
        prompt_template=prompt_template,
        response_parser=response_parser,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        reasoning_config=reasoning_config,
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
    ("response_parser", "full_output"),
    [
        (ThinkingResponseParser, "<think>reasoning</think>answer"),
        (GraniteResponseParser, "<think>reasoning</think><response>answer</response>"),
        (Gemma4ResponseParser, "<|channel>thought\nreasoning<channel|>answer<turn|>"),
    ],
    ids=["optional-thinking", "granite", "gemma4"],
)
def test_generation_is_parsed_at_every_truncation_stage(
    response_parser: type[ResponseParser], full_output: str
) -> None:
    codec = _chat_codec(response_parser=response_parser)
    mid_thinking = full_output[: full_output.index("reasoning") + len("reas")]
    mid_response = full_output[: full_output.index("answer") + len("answ")]

    assert codec.parse_response("answer") == AssistantMessage(chain_of_thought=None, response="answer")
    assert codec.parse_response(mid_thinking) == AssistantMessage(chain_of_thought="reas", response="")
    assert codec.parse_response(mid_response) == AssistantMessage(chain_of_thought="reasoning", response="answ")
    assert codec.parse_response(full_output) == AssistantMessage(chain_of_thought="reasoning", response="answer")


def test_granite_parses_a_response_missing_its_wrapper() -> None:
    codec = _chat_codec(response_parser=GraniteResponseParser)

    expected = AssistantMessage(chain_of_thought="reasoning", response="answer")
    assert codec.parse_response("<think>reasoning</think>answer") == expected


def _registered_codec(repo: str) -> ChatCodec:
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[repo]
    assert isinstance(spec, LanguageModelSpec)
    return _chat_codec(response_parser=spec.response_parser)


@pytest.mark.parametrize(
    ("repo", "full_output", "reasoning"),
    [
        (
            "meta-models/Muse-Glimmer-30B",
            "to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            "reasoning",
        ),
        ("meta-models/Muse-Glimmer-30B", "to=user<|message|>answer<|eot|>", None),
        ("Qwen/Qwen3.8-27B", "reasoning\n</think>\n\nanswer", "reasoning\n"),
        ("Qwen/Qwen3.8-27B", "reasoning</think>answer", "reasoning"),
        ("Qwen/Qwen3.5-9B", "answer", None),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "<think>reasoning</think>answer", "reasoning"),
        (
            "ibm-granite/granite-3.3-2b-instruct",
            "<think>reasoning</think><response>answer</response>",
            "reasoning",
        ),
        ("google/gemma-4-E2B-it", "<|channel>thought\nreasoning<channel|>answer<turn|>", "reasoning"),
        ("google/gemma-4-E2B-it", "answer<turn|>", None),
        (
            "openai/gpt-oss-20b",
            "<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>"
            "answer<|return|>",
            "reasoning",
        ),
        ("openai/gpt-oss-20b", "<|channel|>final<|message|>answer<|return|>", None),
    ],
)
def test_registered_chat_formats_parse_text(repo: str, full_output: str, reasoning: str | None) -> None:
    codec = _registered_codec(repo)
    expected = AssistantMessage(chain_of_thought=reasoning, response="answer")
    assert codec.parse_response(full_output) == expected


@pytest.mark.parametrize(
    ("repo", "prompt", "generated", "reasoning"),
    [
        ("Qwen/Qwen3.8-27B", "<think>\n", "reasoning\n</think>\n\nanswer", "reasoning\n"),
        (
            "ibm-granite/granite-3.3-2b-instruct",
            "<think>",
            "reasoning</think><response>answer</response>",
            "reasoning",
        ),
        ("google/gemma-4-E2B-it", "<|channel>thought\n", "reasoning<channel|>answer<turn|>", "reasoning"),
        (
            "openai/gpt-oss-20b",
            "<|start|>assistant<|channel|>analysis<|message|>",
            "reasoning<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>",
            "reasoning",
        ),
    ],
)
def test_prompt_opened_reasoning_channel(repo: str, prompt: str, generated: str, reasoning: str) -> None:
    codec = _registered_codec(repo)
    assert codec.parse_response(generated, prompt=prompt) == AssistantMessage(
        chain_of_thought=reasoning, response="answer"
    )


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
    ("repo", "raw", "reasoning"),
    [
        ("Qwen/Qwen3.5-0.8B", "<think>reasoning</think>before" + _QWEN_CALL * 2 + "after", "reasoning"),
        ("LiquidAI/LFM2.5-1.2B-Instruct", "before" + _LIQUID_CALL * 2 + "after", None),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "<think>reasoning</think>before" + _LIQUID_CALL * 2 + "after", "reasoning"),
        (
            "meta-models/Muse-Glimmer-30B",
            "to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>before"
            "<|eom|><|start|>assistant "
            + _MUSE_CALL
            + "<|eom|><|start|>assistant "
            + _MUSE_CALL
            + "<|eom|><|start|>assistant to=user<|message|>after",
            "reasoning",
        ),
    ],
)
def test_native_tool_calls_parse(repo: str, raw: str, reasoning: str | None) -> None:
    codec = _registered_codec(repo)
    expected = AssistantMessage(chain_of_thought=reasoning, response="beforeafter", tool_calls=(_CALL, _CALL))
    assert codec.parse_response(raw, tools=_TOOLS) == expected
    assert codec.parse_response(raw).tool_calls == ()


@pytest.mark.parametrize(
    ("response_parser", "raw"),
    [
        (QwenResponseParser, "<tool_call><function=lookup>"),
        (QwenResponseParser, "<tool_call>lookup</tool_call>"),
        (QwenResponseParser, "<tool_call><function=lookup><parameter=count>2</parameter>junk</function></tool_call>"),
        (
            QwenResponseParser,
            "<tool_call><function=lookup><parameter=count>1</parameter>"
            "<parameter=count>2</parameter></function></tool_call>",
        ),
        (LiquidResponseParser, "<|tool_call_start|>[lookup("),
        (LiquidResponseParser, "<|tool_call_start|>[lookup(text=evil())]<|tool_call_end|>"),
        (LiquidResponseParser, "<|tool_call_start|>[lookup(**data)]<|tool_call_end|>"),
        (LiquidResponseParser, "<|tool_call_start|>[lookup(text='x', text='y')]<|tool_call_end|>"),
    ],
)
def test_malformed_tool_calls_remain_text(response_parser: type[ResponseParser], raw: str) -> None:
    codec = _chat_codec(response_parser=response_parser)
    assert codec.parse_response(raw, tools=_TOOLS) == AssistantMessage(response=raw)


@pytest.mark.parametrize(
    ("response_parser", "call"), [(QwenResponseParser, _QWEN_CALL), (LiquidThinkingResponseParser, _LIQUID_CALL)]
)
def test_tool_examples_inside_reasoning_are_never_called(response_parser: type[ResponseParser], call: str) -> None:
    codec = _chat_codec(response_parser=response_parser)
    assert codec.parse_response("<think>" + call + "</think>answer", tools=_TOOLS) == AssistantMessage(
        chain_of_thought=call, response="answer"
    )


def test_prompt_closed_reasoning_preserves_text_and_tool_calls() -> None:
    codec = _registered_codec("Qwen/Qwen3.5-0.8B")
    prompt = "<think>\n\n</think>\n\n"
    assert codec.parse_response("Use </think> here.", prompt=prompt) == AssistantMessage(response="Use </think> here.")
    assert codec.parse_response(_QWEN_CALL + "Use </think> here.", prompt=prompt, tools=_TOOLS) == AssistantMessage(
        response="Use </think> here.", tool_calls=(_CALL,)
    )


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        "[lookup.text(count=1)]",
        "[lookup(count={1, 2})]",
        "[lookup(text=b'x')]",
        "[lookup(count=1j)]",
        "[lookup(data=(1, 2))]",
        "[lookup(data={1: 'x'})]",
        "[lookup(data={[]: 'x'})]",
        "[lookup(count=1e999)]",
    ],
)
def test_liquid_non_json_calls_remain_text(body: str) -> None:
    codec = _registered_codec("LiquidAI/LFM2.5-1.2B-Instruct")
    raw = "<|tool_call_start|>" + body + "<|tool_call_end|>"
    assert codec.parse_response(raw, tools=_TOOLS) == AssistantMessage(response=raw)


def test_liquid_calls_accept_hyphenated_openai_names() -> None:
    codec = _chat_codec(response_parser=LiquidResponseParser)
    tools: tuple[ToolSchema, ...] = ({"type": "function", "function": {"name": "get-weather"}},)
    message = codec.parse_response("<|tool_call_start|>[get-weather(city='Paris')]<|tool_call_end|>", tools=tools)
    assert message.tool_calls == (
        {"type": "function", "function": {"name": "get-weather", "arguments": {"city": "Paris"}}},
    )


def test_liquid_hyphenated_names_are_only_replaced_in_call_positions() -> None:
    codec = _chat_codec(response_parser=LiquidResponseParser)
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
