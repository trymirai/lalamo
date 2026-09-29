import json
from pathlib import Path

import polars as pl
import pytest
from cattrs.errors import ClassValidationError
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer

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
    UserMessage,
    parse_hf_message,
)
from lalamo.utils.template_hacking import fix_chat_template


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


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    ("repo", "revision", "native_reasoning"),
    [
        ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", {"reasoning_content": "Think."}),
        ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", {"reasoning_content": "Think."}),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "f313478934a7612d22991f752959d7a1a8756fec", {"reasoning_content": "Think."}),
    ],
)
def test_mixture_parquet_preserves_huggingface_requests(
    tmp_path: Path, serialized: bool, repo: str, revision: str, native_reasoning: dict[str, str]
) -> None:
    function = {"name": "calculate", "arguments": {"z": [4, {"text": "π <&>"}], "a": True}}
    calls = [{"type": "function", "id": "call_1", "index": 0, "function": function}]
    messages: list[dict] = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Calculate."},
        {"role": "assistant", "content": "", "reasoning_content": "Think.", "tool_calls": calls},
        {"role": "tool", "content": "4", "tool_call_id": "call_1", "name": "calculate"},
        {"role": "assistant", "content": "4"},
    ]
    tools: list[dict] = [
        {
            "type": "function",
            "function": {"name": "calculate", "description": "Calculate.", "parameters": {"type": "object"}},
        }
    ]
    row = {"messages": json.loads(json.dumps(messages)), "tools": json.dumps(tools), "metadata": {"source_row_idx": 0}}
    for message in row["messages"]:
        if "reasoning_content" in message:
            message["reasoning"] = message.pop("reasoning_content")
        if "tool_calls" in message:
            message["tool_calls"][0]["function"]["arguments"] = json.dumps(function["arguments"])
            message["tool_calls"] = json.dumps(message["tool_calls"])
    path = tmp_path / "mixture.parquet"
    pl.DataFrame([row]).write_parquet(path)
    (saved,) = load_hf_parquet(path).collect().to_dicts()
    if not serialized:
        saved["tools"] = json.loads(saved["tools"])
        for message in saved["messages"]:
            if message["tool_calls"] is not None:
                message["tool_calls"] = json.loads(message["tool_calls"])
    conversation = HFConversation.from_dict(saved)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    assert tokenizer is not None
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[repo]
    assert isinstance(spec, LanguageModelSpec)
    config = ChatCodecConfig(
        prompt_template=fix_chat_template(tokenizer.get_chat_template()),
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=tokenizer.eos_token,
        bos_token=tokenizer.bos_token,
        reasoning_config=spec.reasoning_config,
    )
    codec = ChatCodecConfig.from_json(config.to_json()).init(tokenizer.backend_tokenizer)
    messages[2].pop("reasoning_content")
    messages[2].update(native_reasoning)
    assert [codec.message_to_dict(message) for message in conversation.messages] == messages
    assert conversation.tools == tuple(tools)
    assert codec.encode_request(conversation.messages[:-1], tools=conversation.tools) == tokenizer.apply_chat_template(
        messages[:-1], tools=[*tools], add_generation_prompt=True, return_dict=False
    )


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
