import json
from pathlib import Path

import polars as pl
import pytest
from cattrs.errors import ClassValidationError
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.data.huggingface_message import HFConversation, load_hf_parquet
from lalamo.model_import.model_specs.output_parser_regexes import (
    GEMMA4_OUTPUT_PARSER_REGEX,
    GRANITE_THINKING_OUTPUT_PARSER_REGEX,
    OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
)
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.models.chat_codec import (
    AssistantMessage,
    ChatCodec,
    ChatCodecConfig,
    ReasoningConfig,
    ReasoningEffort,
    UserMessage,
    parse_hf_message,
)


def _chat_codec(
    *,
    prompt_template: str = "",
    output_parser_regex: str | None = None,
    reasoning_config: ReasoningConfig | None = None,
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
