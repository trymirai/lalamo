import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from cattrs.errors import ClassValidationError
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import BPE, WordLevel

from lalamo.data.huggingface_message import HFConversation, load_hf_parquet
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.muse_glimmer import MUSE_GLIMMER_OUTPUT_PARSER_REGEX
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
    ToolMessage,
    ToolSchema,
    UserMessage,
    parse_hf_message,
)
from lalamo.utils.template_hacking import fix_chat_template


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


@pytest.mark.parametrize("missing_field", [None, "default_reasoning_effort", "reasoning_effort_to_template_fields"])
def test_loading_rejects_incomplete_or_unmapped_reasoning_effort(missing_field: str | None) -> None:
    config = _chat_codec().config.to_json()
    assert isinstance(config, dict)
    reasoning: dict = {
        "default_reasoning_effort": "medium",
        "reasoning_effort_to_template_fields": {"medium": {"reasoning_effort": "medium"}},
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
            reasoning_effort_to_template_fields=frozendict(
                {
                    ReasoningEffort.LOW: frozendict(reasoning_effort="low"),
                    ReasoningEffort.MEDIUM: frozendict(reasoning_effort="medium"),
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
    codec = _chat_codec(
        prompt_template='{{ {"messages": messages, "tools": tools} | tojson }}',
        tool_call_format=ToolCallFormat.LIQUID,
    )
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
        {"role": "assistant", "name": 123},
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


@pytest.mark.parametrize(
    ("repo", "full_output", "stop_suffix", "effort", "reasoning"),
    [
        (
            "meta-models/Muse-Glimmer-30B",
            "to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            "<|eot|>",
            None,
            "reasoning",
        ),
        ("meta-models/Muse-Glimmer-30B", "to=user<|message|>answer<|eot|>", "<|eot|>", None, None),
        ("Qwen/Qwen3.8-27B", "reasoning\n</think>\n\nanswer", "", None, "reasoning\n"),
        ("Qwen/Qwen3.8-27B", "reasoning</think>answer", "", None, "reasoning"),
        ("Qwen/Qwen3.5-9B", "answer", "", ReasoningEffort.NO_REASONING, None),
        ("LiquidAI/LFM2.5-1.2B-Thinking", "<think>reasoning</think>answer", "", None, "reasoning"),
        ("LiquidAI/LFM2.5-2.6B", "reasoning</think>answer", "", None, "reasoning"),
        (
            "ibm-granite/granite-3.3-2b-instruct",
            "<think>reasoning</think><response>answer</response>",
            "",
            ReasoningEffort.MEDIUM,
            "reasoning",
        ),
        (
            "google/gemma-4-E2B-it",
            "<|channel>thought\nreasoning<channel|>answer<turn|>",
            "<turn|>",
            ReasoningEffort.MEDIUM,
            "reasoning",
        ),
        ("google/gemma-4-E2B-it", "answer<turn|>", "<turn|>", None, None),
        (
            "openai/gpt-oss-20b",
            "<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>"
            "answer<|return|>",
            "<|return|>",
            None,
            "reasoning",
        ),
        ("openai/gpt-oss-20b", "<|channel|>final<|message|>answer<|return|>", "<|return|>", None, None),
    ],
)
def test_registered_chat_formats_stream_only_parsed_text(
    repo: str,
    full_output: str,
    stop_suffix: str,
    effort: ReasoningEffort | None,
    reasoning: str | None,
) -> None:
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[repo]
    assert isinstance(spec, LanguageModelSpec)
    config = replace(
        _chat_codec(output_parser_regex=spec.output_parser_regex, reasoning_config=spec.reasoning_config).config,
        end_of_thinking_tag=spec.end_of_thinking_tag,
    )
    expected = AssistantMessage(chain_of_thought=reasoning, response="answer")
    raw_output = full_output.removesuffix(stop_suffix)
    partitions = [(raw_output,), tuple(raw_output)] + [
        (raw_output[:offset], raw_output[offset:]) for offset in range(1, len(raw_output))
    ]
    for parts in partitions:
        vocabulary = {piece: index for index, piece in enumerate(dict.fromkeys(parts))}
        tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
        tokenizer.decoder = Fuse()
        codec = config.init(tokenizer)
        assert codec.parse_response(full_output) == expected
        prompt = ""
        if repo in ("Qwen/Qwen3.8-27B", "LiquidAI/LFM2.5-2.6B"):
            prompt = "<think>\n"
        elif repo == "Qwen/Qwen3.5-9B" and effort is ReasoningEffort.NO_REASONING:
            prompt = "<think>\n\n</think>\n\n"
        decoder = codec.decode_stream(effort, prompt=prompt)
        streamed_reasoning = ""
        streamed_response = ""
        for piece in parts:
            reasoning_piece, response_piece = decoder.step(vocabulary[piece])
            streamed_reasoning += reasoning_piece
            streamed_response += response_piece
            assert (expected.chain_of_thought or "").startswith(streamed_reasoning)
            assert expected.response.startswith(streamed_response)
        assert streamed_reasoning == (expected.chain_of_thought or "")
        assert streamed_response == expected.response
        assert decoder.finish() == expected


def test_streamed_unicode_matches_complete_byte_fallback_decoding() -> None:
    vocabulary = {f"<0x{byte:02X}>": byte for byte in range(256)}
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    tokenizer.decoder = Sequence([ByteFallback(), Fuse()])
    codec = _chat_codec().config.init(tokenizer)
    expected = "Aé🙂Z"
    token_ids = list(expected.encode())
    decoder = codec.decode_stream()
    assert "".join(decoder.step(token_id)[1] for token_id in token_ids) == expected
    assert decoder.finish().response == codec.decode_tokens(token_ids) == expected


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_participant_names_survive_message_rendering(role: str) -> None:
    source = {"role": role, "content": "hello", "name": "participant"}
    codec = _chat_codec(prompt_template="{{ messages | tojson }}")
    assert json.loads(codec.render_request([parse_hf_message(source)])) == [source]


_PARTICIPANT_TEMPLATES = {
    ToolCallFormat.QWEN_XML: """
{%- if messages[0].role == 'system' %}
{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role != 'system' %}
{{- '<|im_start|>' + message.role + '\\n<think>\\n</think>\\n' + message.content }}
{%- for tc in message.tool_calls | default([]) %}
{{- '<tool_call><function=' + tc.function.name + '></function></tool_call>' }}
{%- endfor %}{{- '<|im_end|>\\n' }}
{%- endif %}{%- endfor %}{{- '<|im_start|>assistant\\n<think>\\n' }}
""",
    ToolCallFormat.LIQUID: """
{%- set ns = namespace(system_prompt="") -%}
{%- if messages and messages[0]["role"] == "system" -%}
{%- set ns.system_prompt = messages[0]["content"] -%}
{%- set messages = messages[1:] -%}
{%- endif -%}
{%- if ns.system_prompt -%}
{{- "<|im_start|>system\\n" + ns.system_prompt + "<|im_end|>\\n" -}}
{%- endif -%}{%- for message in messages -%}
{{- "<|im_start|>" + message.role + "\\n" -}}{{- message.content -}}
{%- for tc in message.tool_calls | default([]) -%}
{{- "<|tool_call_start|>[" + tc.function.name + "()]<|tool_call_end|>" -}}
{%- endfor -%}{{- "<|im_end|>\\n" -}}
{%- endfor -%}{{- "<|im_start|>assistant\\n" -}}
""",
    ToolCallFormat.MUSE_ATEM: """
{%- macro render_content(content) -%}{{- content -}}{%- endmacro -%}
{%- for message in messages -%}
{%- if message.role == 'system' -%}{{- '<|start|>system<|message|>' -}}{{- message.content -}}
{%- elif message.role == 'user' -%}{{- '<|start|>user<|message|>' -}}{{- message.content -}}
{%- elif message.role == 'tool' -%}{{- '<|start|>tool ' + message.name + '<|message|>' -}}{{- message.content -}}
{%- elif message.role == 'assistant' -%}
{%- if message.get('reasoning_content') -%}
{{- '<|start|>assistant to=self<|message|>' + message['reasoning_content'] + '<|eom|>' -}}
{%- endif -%}
{%- if message.get('tool_calls') -%}
            {%- for tc in message['tool_calls'] -%}
{{- '<|start|>assistant to=' + tc.function.name + '<|message|>' -}}
{{- '<atem:function_calls><atem:invoke name="' + tc.function.name + '"></atem:invoke></atem:function_calls>' -}}
{%- endfor -%}
{%- else -%}{%- set recipient = 'user' -%}
{{- '<|start|>assistant' -}}
            {%- if recipient -%}{{- ' to=' + recipient -}}{%- endif -%}
{{- '<|message|>' -}}{{- message.content -}}
{%- endif -%}
{%- endif -%}{{- '<|eot|>' -}}
{%- endfor -%}{{- '<|start|>assistant' -}}
""",
}


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant", "assistant_tool_only"])
def test_native_participant_header_preserves_payload_tool_routing_and_unnamed_generation(
    tool_format: ToolCallFormat,
    role: str,
) -> None:
    codec = _chat_codec(prompt_template=_PARTICIPANT_TEMPLATES[tool_format], tool_call_format=tool_format)
    name = 'café Alice "quoted"\nline'
    source = {"role": role, "content": "hello", "name": name}
    if role == "assistant_tool_only":
        source = {
            "role": "assistant",
            "content": None,
            "name": name,
            "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "lookup", "arguments": {}}}],
        }
    named = parse_hf_message(source)
    unnamed = parse_hf_message({**source, "name": None})
    history = [named, UserMessage("Next")]
    if isinstance(named, AssistantMessage):
        history = [UserMessage("Hello"), named, UserMessage("Next")]
    ordinary = [unnamed if entry is named else entry for entry in history]
    before, after = codec.render_request(ordinary), codec.render_request(history)
    fragment = " name=" + json.dumps(name, ensure_ascii=False)
    assert fragment in after and after.replace(fragment, "") == before
    assert codec.decode_stream(prompt=after).prefix == codec.decode_stream(prompt=before).prefix
    if tool_format is ToolCallFormat.MUSE_ATEM:
        assert after.endswith("<|start|>assistant")
        if role == "assistant_tool_only":
            assert f"assistant{fragment} to=lookup<|message|><atem:function_calls>" in after
    else:
        assert after.endswith(("<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n"))
    assert codec.message_to_dict(named)["name"] == name
    if role == "assistant_tool_only":
        assert codec.message_to_dict(named)["tool_calls"][0]["function"]["name"] == "lookup"


def test_custom_json_template_keeps_named_tool_only_history_canonical() -> None:
    source = {
        "role": "assistant",
        "content": "",
        "name": 'Alice "quoted"',
        "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "lookup", "arguments": {}}}],
    }
    codec = _chat_codec(prompt_template="{{ messages | tojson }}", tool_call_format=ToolCallFormat.MUSE_ATEM)
    assert json.loads(codec.render_request([parse_hf_message(source)])) == [source]


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_native_tool_message_name_remains_a_function_route(tool_format: ToolCallFormat) -> None:
    codec = _chat_codec(prompt_template=_PARTICIPANT_TEMPLATES[tool_format], tool_call_format=tool_format)
    rendered = codec.render_request([ToolMessage("result", name="lookup"), UserMessage("Next")])
    if tool_format is ToolCallFormat.MUSE_ATEM:
        assert "<|start|>tool lookup<|message|>result<|eot|>" in rendered
    else:
        assert "<|im_start|>tool\n<think>\n</think>\nresult" in rendered or "<|im_start|>tool\nresult" in rendered


@pytest.mark.parametrize("name", [None, "", "Alice"])
def test_liquid_named_empty_system_emits_only_the_supplied_participant_header(name: str | None) -> None:
    codec = _chat_codec(
        prompt_template=_PARTICIPANT_TEMPLATES[ToolCallFormat.LIQUID],
        tool_call_format=ToolCallFormat.LIQUID,
    )
    message = parse_hf_message({"role": "system", "content": "", "name": name})
    rendered = codec.render_request([message, UserMessage("Next")])
    expected = "<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n"
    if name is not None:
        expected = "<|im_start|>system name=" + json.dumps(name) + "\n<|im_end|>\n" + expected
    assert rendered == expected


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_native_assistant_text_with_tool_calls_preserves_public_content_and_routing(
    tool_format: ToolCallFormat,
) -> None:
    codec = _chat_codec(prompt_template=_PARTICIPANT_TEMPLATES[tool_format], tool_call_format=tool_format)
    call: ToolCall = {"type": "function", "function": {"name": "lookup", "arguments": {}}, "id": "call_1"}
    message = AssistantMessage("private reasoning", "Distinctive public text", (call,), name="Alice")
    history = [UserMessage("Call lookup"), message, ToolMessage("result", "lookup", "call_1"), UserMessage("Next")]
    rendered = codec.render_request(history)
    call_start = {
        ToolCallFormat.QWEN_XML: "<tool_call>",
        ToolCallFormat.LIQUID: "<|tool_call_start|>",
        ToolCallFormat.MUSE_ATEM: "to=lookup<|message|><atem:function_calls>",
    }[tool_format]
    assert rendered.count(message.response) == 1
    assert rendered.index(message.response) < rendered.index(call_start)
    assert codec.message_to_dict(message)["content"] == message.response
    assert codec.message_to_dict(message)["tool_calls"][0]["function"]["name"] == "lookup"
    if tool_format is ToolCallFormat.MUSE_ATEM:
        public_span = '<|start|>assistant name="Alice" to=user<|message|>Distinctive public text<|eom|>'
        private_span = '<|start|>assistant name="Alice" to=self<|message|>private reasoning<|eom|>'
        assert private_span + public_span + '<|start|>assistant name="Alice" to=lookup' in rendered
        empty = AssistantMessage("private reasoning", "", (call,), name="Alice")
        without_text = codec.render_request([history[0], empty, *history[2:]])
        assert rendered.replace(public_span, "") == without_text
        assert codec.decode_stream(prompt=rendered).prefix == codec.decode_stream(prompt=without_text).prefix


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
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.8-27B",
        "LiquidAI/LFM2.5-230M",
        "LiquidAI/LFM2.5-350M",
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LiquidAI/LFM2.5-1.2B-Thinking",
        "LiquidAI/LFM2.5-2.6B",
        "meta-models/Muse-Glimmer-30B",
    ],
)
def test_catalog_tools_parse_and_stream_every_split(repo: str) -> None:
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model[repo]
    assert isinstance(spec, LanguageModelSpec)
    codec = _chat_codec(output_parser_regex=spec.output_parser_regex, tool_call_format=spec.tool_call_format)
    if spec.tool_call_format is ToolCallFormat.QWEN_XML:
        call = _QWEN_CALL
        raw = "<think>reasoning</think>before" + call + call + "after"
        reasoning = "reasoning"
    elif spec.tool_call_format is ToolCallFormat.LIQUID:
        call = _LIQUID_CALL
        raw = "before" + call + call + "after"
        reasoning = None
        if spec.output_parser_regex is not None:
            raw = "<think>reasoning</think>" + raw
            reasoning = "reasoning"
    else:
        call = _MUSE_CALL
        raw = (
            "to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>before"
            "<|eom|><|start|>assistant "
            + call
            + "<|eom|><|start|>assistant "
            + call
            + "<|eom|><|start|>assistant to=user<|message|>after"
        )
        reasoning = "reasoning"
    expected = AssistantMessage(chain_of_thought=reasoning, response="beforeafter", tool_calls=(_CALL, _CALL))
    assert codec.parse_response(raw, tools=_TOOLS) == expected
    assert codec.parse_response(raw).tool_calls == ()
    partitions = [tuple(raw), (raw,)] + [(raw[:position], raw[position:]) for position in range(1, len(raw))]
    for parts in partitions:
        vocabulary = {piece: index for index, piece in enumerate(dict.fromkeys(parts))}
        tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[]))
        tokenizer.decoder = Fuse()
        decoder = codec.config.init(tokenizer).decode_stream(prompt="", tools=_TOOLS)
        streamed_reasoning = ""
        streamed_response = ""
        for piece in parts:
            reasoning_piece, response_piece = decoder.step(vocabulary[piece])
            streamed_reasoning += reasoning_piece
            streamed_response += response_piece
            assert (reasoning or "").startswith(streamed_reasoning)
            assert expected.response.startswith(streamed_response)
            assert expected.tool_calls[: len(decoder.tool_calls)] == decoder.tool_calls
        assert streamed_reasoning == (reasoning or "")
        assert streamed_response == expected.response
        assert decoder.finish() == expected


@pytest.mark.parametrize(
    "tool_format,call",
    [
        (ToolCallFormat.QWEN_XML, _QWEN_CALL),
        (ToolCallFormat.LIQUID, _LIQUID_CALL),
    ],
)
def test_tool_examples_inside_reasoning_are_never_called(tool_format: ToolCallFormat, call: str) -> None:
    codec = _chat_codec(output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX, tool_call_format=tool_format)
    assert codec.parse_response("<think>" + call + "</think>answer", tools=_TOOLS) == AssistantMessage(
        chain_of_thought=call,
        response="answer",
    )


@pytest.mark.parametrize(
    "tool_format,raw",
    [
        (ToolCallFormat.QWEN_XML, "<tool_call><function=lookup>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup("),
        (ToolCallFormat.MUSE_ATEM, "to=lookup<|message|><atem:function_calls>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(text=evil())]<|tool_call_end|>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(**data)]<|tool_call_end|>"),
        (ToolCallFormat.LIQUID, "<|tool_call_start|>[lookup(text='x', text='y')]<|tool_call_end|>"),
        (
            ToolCallFormat.QWEN_XML,
            "<tool_call><function=lookup><parameter=count>bad</parameter></function></tool_call>",
        ),
    ],
)
def test_incomplete_or_invalid_tool_calls_fail_without_invented_calls(tool_format: ToolCallFormat, raw: str) -> None:
    codec = _chat_codec(tool_call_format=tool_format)
    with pytest.raises(ValueError):
        codec.parse_response(raw, tools=_TOOLS)


def test_absent_tool_capability_rejects_requested_tools() -> None:
    codec = _chat_codec(prompt_template="{{ messages | tojson }}")
    with pytest.raises(ValueError, match="does not support tool calling"):
        codec.render_request([UserMessage("hello")], tools=_TOOLS)
    assert json.loads(codec.render_request([UserMessage("hello")], tools=[])) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "function,expected",
    [
        (
            {"name": "lookup", "parameters": {"type": "object"}},
            {"name": "lookup", "description": "", "parameters": {"type": "object"}},
        ),
        (
            {"name": "lookup", "description": "Look up a value"},
            {"name": "lookup", "description": "Look up a value", "parameters": {}},
        ),
        ({"name": "lookup"}, {"name": "lookup", "description": "", "parameters": {}}),
        (
            {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object"}},
            {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object"}},
        ),
        (
            {"name": "lookup", "description": "", "parameters": {}},
            {"name": "lookup", "description": "", "parameters": {}},
        ),
        (
            {"name": "lookup", "description": None, "parameters": None},
            {"name": "lookup", "description": None, "parameters": None},
        ),
    ],
)
def test_muse_tool_definitions_render_optional_fields(function: dict, expected: dict) -> None:
    # This is the published Muse tool-definition expression, including its two optional fields.
    template = """{% for tool in tools %}{% set fn = tool.function if tool.function is defined else tool %}
{{- '\\n{"name": ' + (fn.name | tojson) + ', "description": ' + (fn.description | tojson)
    + ', "parameters": ' + (fn.parameters | tojson) + '}' -}}
{% endfor %}"""
    codec = _chat_codec(prompt_template=fix_chat_template(template), tool_call_format=ToolCallFormat.MUSE_ATEM)
    assert (
        json.loads(codec.render_request([UserMessage("hello")], tools=[{"type": "function", "function": function}]))
        == expected
    )


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
def test_native_tool_history_round_trip(tool_format: ToolCallFormat) -> None:
    # These renderers retain the shipped templates' exact tool argument expressions and protocol boundaries.
    if tool_format is ToolCallFormat.QWEN_XML:
        template = """{% for message in messages %}{% for tc in message.tool_calls | default([]) %}
<tool_call><function={{ tc.function.name }}>{% for args_name, args_value in tc.function.arguments.items() %}
<parameter={{ args_name }}>
{{ args_value | string if args_value is string else args_value | tojson | safe }}
</parameter>{% endfor %}</function></tool_call>{% endfor %}{% endfor %}"""
    elif tool_format is ToolCallFormat.LIQUID:
        template = """{% macro format_arg_value(arg_value) %}{% if arg_value is string %}
{{ "'" + arg_value + "'" }}{% elif arg_value is mapping or arg_value is iterable %}
{{ arg_value | tojson }}{% else %}{{ arg_value | string }}{% endif %}{% endmacro %}
{% for message in messages %}{% if message.tool_calls %}<|tool_call_start|>[{% for tc in message.tool_calls %}
{{ tc.function.name }}({% for key, value in tc.function.arguments.items() %}{{ key }}={{ format_arg_value(value) }}
{% if not loop.last %}, {% endif %}{% endfor %}){% if not loop.last %}, {% endif %}{% endfor %}]<|tool_call_end|>
{% endif %}{% endfor %}"""
        template = fix_chat_template(template)
    else:
        template = """{% for message in messages %}{% for tc in message.tool_calls | default([]) %}
<|start|>assistant to={{ tc.function.name }}<|message|><atem:function_calls>
<atem:invoke name="{{ tc.function.name }}">{% for key, value in tc.function.arguments.items() %}
<atem:parameter name="{{ key }}">{{ value if value is string else value | tojson }}</atem:parameter>
{% endfor %}</atem:invoke></atem:function_calls>{% if not loop.last %}<|eom|>{% endif %}{% endfor %}{% endfor %}"""
    arguments = {**_CALL["function"]["arguments"], "text": "  'quote' \\path\nnext\rline🙂  "}
    call: ToolCall = {"type": "function", "function": {"name": "lookup", "arguments": arguments}}
    codec = _chat_codec(prompt_template=template, tool_call_format=tool_format)
    raw = codec.render_request([AssistantMessage(tool_calls=(call, call))], tools=_TOOLS).strip()
    assert codec.parse_response(raw, tools=_TOOLS) == AssistantMessage(tool_calls=(call, call))


@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.MUSE_ATEM])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"]},
        {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        {"oneOf": [{"type": "string"}, {"type": "boolean"}]},
        {"$ref": "#/$defs/ambiguous"},
    ],
)
def test_unquoted_native_formats_reject_ambiguous_parameter_schemas(
    tool_format: ToolCallFormat,
    schema: dict,
) -> None:
    tools: tuple[ToolSchema, ...] = (
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "properties": {"text": schema},
                    "$defs": {"ambiguous": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                },
            },
        },
    )
    codec = _chat_codec(tool_call_format=tool_format)
    with pytest.raises(ValueError, match="unquoted string format cannot distinguish"):
        codec.render_request([UserMessage("hello")], tools=tools)
    liquid = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    liquid.render_request([UserMessage("hello")], tools=tools)


@pytest.mark.parametrize(
    "tool_format,raw",
    [
        (
            ToolCallFormat.QWEN_XML,
            "<tool_call><function=lookup><parameter=text>null</parameter>"
            '<parameter=data>{"answer":true}</parameter></function></tool_call>',
        ),
        (
            ToolCallFormat.MUSE_ATEM,
            'to=lookup<|message|><atem:function_calls><atem:invoke name="lookup">'
            '<atem:parameter name="text">null</atem:parameter>'
            '<atem:parameter name="data">{"answer":true}</atem:parameter></atem:invoke></atem:function_calls>',
        ),
    ],
)
def test_local_schema_references_preserve_strings_and_decode_objects(tool_format: ToolCallFormat, raw: str) -> None:
    tools: tuple[ToolSchema, ...] = (
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "properties": {"text": {"$ref": "#/$defs/string"}, "data": {"$ref": "#/$defs/object"}},
                    "$defs": {"string": {"type": "string"}, "object": {"type": "object"}},
                },
            },
        },
    )
    codec = _chat_codec(tool_call_format=tool_format)
    codec.render_request([UserMessage("hello")], tools=tools)
    assert codec.parse_response(raw, tools=tools).tool_calls[0]["function"]["arguments"] == {
        "text": "null",
        "data": {"answer": True},
    }


@pytest.mark.parametrize("recipient", ["self", "user", "lookup"])
def test_muse_stream_uses_the_actual_prefilled_recipient_header(recipient: str) -> None:
    body = "hello"
    expected = AssistantMessage(response=body)
    if recipient == "self":
        expected = AssistantMessage(chain_of_thought=body)
    elif recipient == "lookup":
        body = _MUSE_CALL.split("<|message|>", 1)[1]
        expected = AssistantMessage(tool_calls=(_CALL,))
    tokenizer = Tokenizer(BPE(vocab={character: i for i, character in enumerate(dict.fromkeys(body))}, merges=[]))
    tokenizer.decoder = Fuse()
    codec = _chat_codec(
        output_parser_regex=MUSE_GLIMMER_OUTPUT_PARSER_REGEX,
        tool_call_format=ToolCallFormat.MUSE_ATEM,
    ).config.init(tokenizer)
    decoder = codec.decode_stream(prompt=f"<|start|>assistant to={recipient}<|message|>\n", tools=_TOOLS)
    pieces = [decoder.step(tokenizer.token_to_id(character)) for character in body]
    assert "".join(piece[0] for piece in pieces) == (expected.chain_of_thought or "")
    assert "".join(piece[1] for piece in pieces) == expected.response
    assert decoder.finish() == expected


@pytest.mark.parametrize(
    "name,parameter",
    [
        ("search-web", "text"),
        ("1lookup", "text"),
        ("class", "text"),
        ("lookup", "some-text"),
        ("lookup", "class"),
        ("lookup", "\u212a"),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_liquid_native_names_preserve_tool_identity(name: str, parameter: str, strict: bool) -> None:
    tools: tuple[ToolSchema, ...] = (
        {
            "type": "function",
            "function": {
                "name": name,
                "strict": strict,
                "parameters": {
                    "type": "object",
                    "properties": {parameter: {"type": "integer"}},
                    "required": [parameter],
                    "additionalProperties": False,
                },
            },
        },
    )
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID)
    codec.render_request([UserMessage("hello")], tools=tools)
    raw = f"<|tool_call_start|>[{name}({parameter}=1)]<|tool_call_end|>"
    assert codec.parse_response(raw, tools=tools).tool_calls == (
        {"type": "function", "function": {"name": name, "arguments": {parameter: 1}}},
    )


@pytest.mark.parametrize("name", ["two words", "call()", "a=b", "a,b", "#comment", "<|tool_call_end|>"])
def test_liquid_rejects_names_containing_native_delimiters(name: str) -> None:
    tools: tuple[ToolSchema, ...] = (
        {"type": "function", "function": {"name": "lookup", "parameters": {"properties": {name: {"type": "string"}}}}},
    )
    with pytest.raises(ValueError, match="cannot be represented"):
        _chat_codec(tool_call_format=ToolCallFormat.LIQUID).render_request([UserMessage("hello")], tools=tools)


def test_token_response_decoding_uses_the_supplied_tool_schemas() -> None:
    tokenizer = Tokenizer(BPE(vocab={_LIQUID_CALL: 0}, merges=[]))
    tokenizer.decoder = Fuse()
    codec = _chat_codec(tool_call_format=ToolCallFormat.LIQUID).config.init(tokenizer)
    assert codec.decode_response([0], tools=_TOOLS) == AssistantMessage(tool_calls=(_CALL,))


@pytest.mark.parametrize("recipient", ["self", "user"])
@pytest.mark.parametrize("strict", [False, True])
def test_muse_matching_native_invocation_distinguishes_tools_from_channel_recipients(
    recipient: str, strict: bool
) -> None:
    tools: list[ToolSchema] = [
        {
            "type": "function",
            "function": {
                "name": recipient,
                "strict": strict,
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    raw = (
        f'to={recipient}<|message|><atem:function_calls><atem:invoke name="{recipient}">'
        '<atem:parameter name="count">1</atem:parameter></atem:invoke></atem:function_calls>'
    )
    tokenizer = Tokenizer(BPE(vocab={character: i for i, character in enumerate(dict.fromkeys(raw))}, merges=[]))
    tokenizer.decoder = Fuse()
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM).config.init(tokenizer)
    stream = codec.decode_stream(prompt="", tools=tools)
    assert all(stream.step(codec.tokenizer.token_to_id(piece)) == ("", "") for piece in raw)
    message = stream.finish()
    assert message.response == "" and message.chain_of_thought is None
    assert [entry["function"] for entry in message.tool_calls] == [{"name": recipient, "arguments": {"count": 1}}]
    assert codec.parse_response(raw, tools=tools) == message


@pytest.mark.parametrize("recipient", ["self", "user"])
@pytest.mark.parametrize(
    "body", ["plain text", "<atem:function_calls> example", '<atem:function_calls><atem:invoke name="other">example']
)
def test_muse_channel_text_preserves_markers_without_a_matching_invocation(recipient: str, body: str) -> None:
    raw = f"to={recipient}<|message|>" + body
    tools: list[ToolSchema] = [{"type": "function", "function": {"name": recipient}}]
    tokenizer = Tokenizer(BPE(vocab={character: i for i, character in enumerate(dict.fromkeys(raw))}, merges=[]))
    tokenizer.decoder = Fuse()
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM).config.init(tokenizer)
    stream = codec.decode_stream(prompt="", tools=tools)
    pieces = [stream.step(codec.tokenizer.token_to_id(piece)) for piece in raw]
    message = stream.finish()
    assert message.tool_calls == ()
    if recipient == "self":
        assert message.chain_of_thought == "".join(reasoning for reasoning, _ in pieces) == body
        assert message.response == ""
    else:
        assert message.response == "".join(content for _, content in pieces) == body
        assert message.chain_of_thought is None


def test_muse_tool_result_routes_only_to_a_preceding_call() -> None:
    template = """
{%- for message in messages -%}
    {%- if message.role == 'tool' -%}
        {%- set tcid = message.get('tool_call_id') -%}
            {%- set rns = namespace(name=tcid if tcid else '') -%}
            {%- for m in messages -%}
                {%- for tc in m.get('tool_calls', []) -%}
                    {%- if tc.id == tcid -%}
                        {%- set rns.name = tc.function.name -%}
                    {%- endif -%}
                {%- endfor -%}
            {%- endfor -%}
        {{- '<tool_output name="' + rns.name + '">' + message.content + '</tool_output>' -}}
    {%- endif -%}
{%- endfor -%}
"""
    codec = _chat_codec(prompt_template=template, tool_call_format=ToolCallFormat.MUSE_ATEM)
    first: ToolCall = {"type": "function", "id": "same", "function": {"name": "first", "arguments": {}}}
    second: ToolCall = {"type": "function", "id": "same", "function": {"name": "second", "arguments": {}}}
    rendered = codec.render_request(
        [
            UserMessage("Call first"),
            AssistantMessage(None, "", (first,)),
            ToolMessage("result A", tool_call_id="same"),
            UserMessage("Call second"),
            AssistantMessage(None, "", (second,)),
            ToolMessage("result B", tool_call_id="same"),
            UserMessage("Summarize"),
        ]
    )
    assert (
        rendered == '<tool_output name="first">result A</tool_output><tool_output name="second">result B</tool_output>'
    )


def test_muse_tool_body_whitespace_preserves_argument_string_spaces() -> None:
    codec = _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM)
    raw = _MUSE_CALL.replace("<|message|>", "<|message|>\n\n") + "\n\n"
    assert codec.parse_response(raw, tools=_TOOLS) == AssistantMessage(tool_calls=(_CALL,))


@pytest.mark.parametrize(
    "raw",
    [
        "<|tool_call_start|>[lookup(text='x')<|tool_call_end|>",
        "<|tool_call_start|>[lookup(text='x') lookup(text='y')]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=(1,))]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=1 2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=1 .2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=[1 2])]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=(1 2))]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=(1)2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=1(2))]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=()2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=[, ])]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count={, })]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=1# comment\n2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(count=1\\\n2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup(é=1,é=2)]<|tool_call_end|>",
        "<|tool_call_start|>[lookup\u00a0(count=1)]<|tool_call_end|>",
        "<tool_call></tool_call>",
        "<tool_call><function=lookup><parameter=count>1</function></tool_call>",
        "<tool_call><function=lookup></function>invalid</tool_call>",
        "<tool_call><function=lookup></function>",
    ],
)
def test_strict_domain_conversion_rejects_invalid_native_structure(raw: str) -> None:
    tool_format = ToolCallFormat.QWEN_XML
    if raw.startswith("<|tool_call_start|>"):
        tool_format = ToolCallFormat.LIQUID
    with pytest.raises(ValueError):
        _chat_codec(tool_call_format=tool_format).parse_response(raw, tools=_TOOLS)


@pytest.mark.parametrize("extra", ["unknown", "<atem:unknown>"])
@pytest.mark.parametrize("after", [False, True])
def test_strict_muse_conversion_rejects_unknown_bytes_around_calls(extra: str, after: bool) -> None:
    raw = _MUSE_CALL
    if after:
        raw += extra
    else:
        raw = raw.replace("<|message|>", "<|message|>" + extra)
    with pytest.raises(ValueError):
        _chat_codec(tool_call_format=ToolCallFormat.MUSE_ATEM).parse_response(raw, tools=_TOOLS)


@pytest.mark.parametrize(
    ("tool_format", "call"),
    [
        (ToolCallFormat.QWEN_XML, _QWEN_CALL),
        (ToolCallFormat.LIQUID, _LIQUID_CALL),
        (ToolCallFormat.MUSE_ATEM, _MUSE_CALL),
    ],
)
def test_stream_tool_calls_exposes_only_complete_native_calls(tool_format: ToolCallFormat, call: str) -> None:
    if tool_format is ToolCallFormat.MUSE_ATEM:
        call += "<|eom|>"
    incomplete = call[: len(call) // 2]
    tokenizer = Tokenizer(BPE(vocab={call: 0, incomplete: 1}, merges=[]))
    tokenizer.decoder = Fuse()
    decoder = _chat_codec(tool_call_format=tool_format).config.init(tokenizer).decode_stream(prompt="", tools=_TOOLS)
    decoder.step(1)
    assert decoder.tool_calls == ()
    decoder = _chat_codec(tool_call_format=tool_format).config.init(tokenizer).decode_stream(prompt="", tools=_TOOLS)
    decoder.step(0)
    assert decoder.tool_calls == (_CALL,)
    decoder.step(1)
    assert decoder.tool_calls == (_CALL,)


@pytest.mark.parametrize(
    ("body", "arguments"),
    [
        ("[lookup(count=[1, # comment\n 2])]", {"count": [1, 2]}),
        ("[lookup(count=1 # comment\r)]", {"count": 1}),
        ("[lookup(count=1# comment\n)]", {"count": 1}),
        ("# comment\n[ # comment\nlookup # comment\n(count # comment\n =1)]# comment", {"count": 1}),
        ("[lookup\\\n(count\\\n=\\\n1\\\n)]", {"count": 1}),
        ("[lookup(text='a' # comment\n'b')]", {"text": "ab"}),
        ("[lookup(é=1)]", {"é": 1}),
        ("[lookup(é=1,é=2)]", {"é": 1, "é": 2}),
        ("[lookup(\uff45=1)]", {"\uff45": 1}),
        ("[lookup(\uff43\uff4c\uff41\uff53\uff53=1)]", {"\uff43\uff4c\uff41\uff53\uff53": 1}),
    ],
)
def test_liquid_layout_and_values_keep_python_literal_semantics(body: str, arguments: dict) -> None:
    raw = "<|tool_call_start|>" + body + "<|tool_call_end|>"
    message = _chat_codec(tool_call_format=ToolCallFormat.LIQUID).parse_response(raw, tools=_TOOLS)
    assert message.tool_calls == ({"type": "function", "function": {"name": "lookup", "arguments": arguments}},)


@pytest.mark.parametrize("name", ["é", "\uff45"])
def test_liquid_function_names_preserve_literal_identity(name: str) -> None:
    raw = f"<|tool_call_start|>[{name}(count=1)]<|tool_call_end|>"
    message = _chat_codec(tool_call_format=ToolCallFormat.LIQUID).parse_response(raw, tools=_TOOLS)
    assert message.tool_calls == ({"type": "function", "function": {"name": name, "arguments": {"count": 1}}},)
