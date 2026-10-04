import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import contextmanager
from threading import Event
from time import monotonic
from typing import ClassVar

import httpx2
import pytest
import xgrammar
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from openai import APIError, AsyncOpenAI
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import Message as ASGIMessage
from starlette.types import Scope
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, ByteLevel, Fuse, Metaspace
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Whitespace

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
    TokenLogprobs,
)
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.model_specs.qwen import QWEN38_REASONING_CONFIG
from lalamo.model_import.model_specs.reasoning_configs import (
    BOOLEAN_REASONING_DEFAULT_OFF_CONFIG,
    BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
)
from lalamo.models import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig, ChatDecodeStream, ReasoningConfig, ToolCallFormat
from lalamo.server import create_app
from lalamo.utils.json import JSON
from tests.helpers import build_tiny_attention_decoder

pytestmark = pytest.mark.fast


@contextmanager
def _echo_client(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Tokenizer,
    *,
    output_parser_regex: str | None = None,
    reasoning_config: ReasoningConfig | None = None,
    tool_call_format: ToolCallFormat | None = None,
    prompt_template: str = '{{ messages[0].get("reasoning_content", messages[0].content) }}',
    cancellations: list[Event] | None = None,
    token_logprobs: TokenLogprobs | None = None,
    submission_error: Exception | None = None,
    output_token_ids: tuple[int, ...] | None = None,
    finish_reason: FinishReason = FinishReason.LENGTH,
) -> Generator[TestClient]:
    codec = ChatCodecConfig(
        prompt_template=prompt_template,
        output_parser_regex=output_parser_regex,
        reasoning_config=reasoning_config,
        tool_call_format=tool_call_format,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    ).init(tokenizer)
    decoder = build_tiny_attention_decoder((None,))
    model = LanguageModel(
        config=LanguageModelConfig(
            token_codec_config=codec.config,
            decoder_config=decoder.config,
            generation_config=GenerationConfig(),
        ),
        token_codec=codec,
        decoder=decoder,
        sharding_config=decoder.sharding_config,
    )
    if cancellations is None:
        cancellations = []

    class EchoEngine:
        context_limit: ClassVar[int] = 64

        def __init__(self, _model: LanguageModel, _config: ContinuousBatchingConfig) -> None:
            pass

        def step(self) -> bool:
            return False

        def submit(
            self,
            prompt_token_ids: tuple[int, ...],
            max_output_length: int,
            _generation_config: GenerationConfig,
            _seed: int,
            *,
            return_logprobs: bool,
            grammar_matcher: xgrammar.GrammarMatcher | None = None,
            on_events: Callable[[Sequence[TokenEvent]], object],
        ) -> Event:
            assert grammar_matcher is None, "Constrained generation must use the real inference engine."
            if submission_error is not None and any(not cancelled.is_set() for cancelled in cancellations):
                raise submission_error
            _generation_config.default_policy(model.decoder.vocab_size)
            cancelled = Event()
            cancellations.append(cancelled)
            output = (prompt_token_ids if output_token_ids is None else output_token_ids)[:max_output_length]
            events: list[TokenEvent] = [
                GeneratedToken(
                    token_id,
                    (token_logprobs or TokenLogprobs(-0.1, (token_id,), (-0.1,))) if return_logprobs else None,
                )
                for token_id in output
            ]
            on_events([*events, SequenceFinished(finish_reason, len(output))])
            return cancelled

    monkeypatch.setattr("lalamo.server.ContinuousBatchingEngine", EchoEngine)
    with TestClient(create_app(model, "test", ContinuousBatchingConfig())) as http:
        yield http
        assert all(cancelled.is_set() for cancelled in cancellations)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "ab": 1, "X": 2, "private": 3, "answer": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = Fuse()
    with _echo_client(monkeypatch, tokenizer) as http:
        yield http


@pytest.fixture
def function_tool() -> dict[str, JSON]:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a city.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "count": {"type": "integer"},
                    "active": {"type": "boolean"},
                    "details": {"type": "object"},
                    "items": {"type": "array"},
                },
            },
        },
    }


@pytest.mark.parametrize(
    "option,active_value",
    [
        ("verbosity", "low"),
        ("prediction", {"type": "content", "content": "ab"}),
        ("prompt_cache_key", "shared-prefix"),
        ("prompt_cache_retention", "in_memory"),
        ("moderation", {"model": "omni-moderation-latest"}),
        ("audio", {"voice": "alloy", "format": "wav"}),
    ],
)
def test_inactive_nullable_text_options_are_accepted_without_enabling_unsupported_behavior(
    client: TestClient, option: str, active_value: JSON
) -> None:
    body: dict[str, JSON] = {"model": "test", "messages": [{"role": "user", "content": "ab"}]}
    inactive = client.post("/v1/chat/completions", json={**body, option: None})
    assert inactive.status_code == 200
    assert inactive.json()["choices"][0]["message"]["content"] == "ab"
    active = client.post("/v1/chat/completions", json={**body, option: active_value})
    assert active.status_code == 400
    assert active.json()["error"]["type"] == "invalid_request_error"
    assert active.json()["error"]["param"] == option
    assert client.post("/v1/chat/completions", json=body).status_code == 200


@pytest.mark.parametrize(
    ("field", "value", "parameter"),
    [
        (
            "response_format",
            {"type": "json_schema", "json_schema": {"name": "Result", "schema": {}, "strict": True}},
            "response_format.json_schema",
        ),
        (
            "response_format",
            {"type": "json_schema", "json_schema": {"name": "Result", "schema": None}},
            "response_format.json_schema.schema",
        ),
        ("response_format", {"type": "json_object", "schema": {}}, "response_format.schema"),
        ("response_format", {"type": "wrong"}, "response_format"),
        ("response_format", {}, "response_format"),
        ("response_format", {"type": "text", "json_schema": {}}, "response_format.json_schema"),
        ("temperature", "wrong", "temperature"),
        ("tool_choice", {"type": "function", "function": {}}, "tool_choice"),
        (
            "tool_choice",
            {"type": "allowed_tools", "allowed_tools": {"mode": "wrong", "tools": []}},
            "tool_choice",
        ),
        (
            "tool_choice",
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "auto", "tools": [{"type": "custom", "function": {"name": "lookup"}}]},
            },
            "tool_choice",
        ),
        ("messages", [{"role": "user", "content": 123}], "messages.0.content"),
        ("messages", [{"role": "user", "content": [{"type": "text"}]}], "messages.0.content"),
        ("messages", [{"role": "assistant", "content": [{"type": "refusal", "refusal": 3}]}], "messages.0.content"),
        ("messages", [{"role": "assistant", "content": []}], "messages.0"),
        ("messages", [{"role": "assistant", "content": "ab", "audio": {"id": "audio_1"}}], "messages.0.audio"),
        ("messages", [{"role": "wrong", "content": "ab"}], "messages.0.role"),
        (
            "tools",
            [{"type": "function", "function": {"name": "lookup", "description": 3}}],
            "tools.0.function.description",
        ),
        ("str", "real extra field", "str"),
    ],
)
def test_request_errors_report_wire_paths_and_preserve_other_field_errors(
    client: TestClient,
    field: str,
    value: JSON,
    parameter: str,
) -> None:
    request = {"model": "test", "messages": [{"role": "user", "content": "ab"}]}
    response = client.post("/v1/chat/completions", json={**request, field: value})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == parameter
    if parameter == "tool_choice":
        assert response.json()["error"]["message"] == "Invalid tool_choice."
    elif parameter == "messages.0.content":
        assert response.json()["error"]["message"] == "Invalid message content."
    normal = client.post("/v1/chat/completions", json=request)
    assert normal.status_code == 200
    assert normal.json()["choices"][0]["message"]["content"] == "ab"


def test_health_reports_a_failed_inference_worker(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/health").status_code == 200

    def fail_step() -> bool:
        raise RuntimeError("Fatal inference failure.")

    monkeypatch.setattr("lalamo.server.ContinuousBatchingEngine.step", staticmethod(fail_step))
    deadline = monotonic() + 5
    while True:
        response = client.get("/health")
        if response.status_code != 200:
            break
        if monotonic() >= deadline:
            pytest.fail("Health did not reflect the failed inference worker.")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert client.get("/v1/health").status_code == 503


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("stop", "expected_content", "expected_tokens"), [("bc", "abX", ["ab", "X"]), ("b", "a", ["ab"])]
)
def test_stop_preserves_original_logprob_tokens(
    client: TestClient, stream: bool, stop: str, expected_content: str, expected_tokens: list[str]
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "ab X"}],
            "stop": stop,
            "logprobs": True,
            "stream": stream,
        },
    )
    assert response.status_code == 200
    if stream:
        choices = [
            json.loads(line.removeprefix("data: "))["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        logprobs = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
    else:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        logprobs = choice["logprobs"]["content"]
    assert content == expected_content
    assert [entry["token"] for entry in logprobs] == expected_tokens
    assert [entry["bytes"] for entry in logprobs] == [list(token.encode()) for token in expected_tokens]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outside_top_twenty", [False, True])
def test_selected_logprob_sentinel_preserves_sampled_bytes_and_alternatives(
    monkeypatch: pytest.MonkeyPatch, stream: bool, outside_top_twenty: bool
) -> None:
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "ab": 1} | {f"alt{index}": index + 2 for index in range(20)}, unk_token="[UNK]")
    )
    tokenizer.decoder = Fuse()
    top_ids = tuple(range(2, 22))
    top_logprobs = tuple(-10.0 - index for index in range(20))
    selected_logprob = -40.0
    if not outside_top_twenty:
        top_ids = (*top_ids[:-1], 1)
        selected_logprob = top_logprobs[-1]
    with _echo_client(
        monkeypatch, tokenizer, token_logprobs=TokenLogprobs(selected_logprob, top_ids, top_logprobs)
    ) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "ab"}],
                "logprobs": True,
                "top_logprobs": 1,
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        choices = [
            json.loads(line.removeprefix("data: "))["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        logprobs = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
    else:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        logprobs = choice["logprobs"]["content"]
    assert content == "ab"
    (entry,) = logprobs
    assert entry["token"] == "ab" and entry["bytes"] == list(b"ab")
    assert entry["logprob"] == (-9999.0 if outside_top_twenty else selected_logprob)
    assert entry["top_logprobs"] == [{"token": "alt0", "bytes": list(b"alt0"), "logprob": -10.0}]


def test_assistant_reasoning_remains_in_conversation_context(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "assistant", "content": "answer", "reasoning_content": "private"},
                {"role": "user", "content": "continue"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "private"


@pytest.mark.parametrize(
    "arguments",
    ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}', '{"x":{"y":NaN}}', "[]", "null", "{"],
)
def test_invalid_tool_history_arguments_do_not_poison_generation(client: TestClient, arguments: str) -> None:
    invalid = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_test", "type": "function", "function": {"name": "lookup", "arguments": arguments}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_test", "content": "result"},
            ],
        },
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    response = client.post(
        "/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "ab"}]}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ab"


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": True},
        {"type": "object", "properties": []},
        {"type": "object", "required": "x"},
        {"type": "object", "properties": {"x": {"type": "not_a_json_schema_type"}}},
    ],
)
def test_invalid_tool_parameter_schemas_return_request_error(client: TestClient, parameters: dict[str, JSON]) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "ab"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": parameters}}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "tools.0.function"


@pytest.mark.parametrize(("effort", "thinking"), [("medium", False), ("none", True)])
def test_conflicting_reasoning_settings_return_request_error(client: TestClient, effort: str, thinking: bool) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "ab"}],
            "reasoning_effort": effort,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    assert response.status_code == 400
    assert "conflicts" in response.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("settings", "thinking"),
    [
        ({}, True),
        ({"reasoning_effort": "low"}, True),
        ({"reasoning_effort": "medium"}, True),
        ({"reasoning_effort": "xhigh"}, True),
        ({"chat_template_kwargs": {"enable_thinking": True}}, True),
        ({"reasoning_effort": "none"}, False),
        ({"chat_template_kwargs": {"enable_thinking": False}}, False),
    ],
)
def test_qwen38_reasoning_modes_stream_and_parse_the_selected_channel(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, JSON], thinking: bool, stream: bool
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "answer": 1}, unk_token="[UNK]"))
    tokenizer.add_tokens(["<think>\n", "<think>\n\n</think>\n\n"])
    tokenizer.decoder = Fuse()
    with _echo_client(
        monkeypatch,
        tokenizer,
        reasoning_config=QWEN38_REASONING_CONFIG,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        prompt_template=(
            "<think>\n{% if enable_thinking is defined and enable_thinking is false %}\n</think>\n\n{% endif %}"
        ),
        output_token_ids=(1,),
    ) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hello"}], "stream": stream, **settings},
        )
    assert response.status_code == 200
    if stream:
        messages = [
            json.loads(line.removeprefix("data: "))["choices"][0]["delta"]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    else:
        messages = [response.json()["choices"][0]["message"]]
    assert "".join(message.get("content") or "" for message in messages) == ("" if thinking else "answer")
    assert "".join(message.get("reasoning_content") or "" for message in messages) == ("answer" if thinking else "")


@pytest.mark.parametrize(
    ("reasoning_config", "enabled_text"),
    [
        (QWEN38_REASONING_CONFIG, "xhigh"),
        (BOOLEAN_REASONING_DEFAULT_OFF_CONFIG, "enabled"),
        (BOOLEAN_REASONING_DEFAULT_ON_CONFIG, "enabled"),
    ],
)
def test_thinking_switch_preserves_the_model_enabled_default(
    monkeypatch: pytest.MonkeyPatch, reasoning_config: ReasoningConfig, enabled_text: str
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "xhigh": 1, "enabled": 2, "disabled": 3}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    with _echo_client(
        monkeypatch,
        tokenizer,
        reasoning_config=reasoning_config,
        prompt_template=(
            "{% if enable_thinking is defined and enable_thinking is false %}disabled"
            "{% else %}{{ reasoning_effort | default('enabled') }}{% endif %}"
        ),
    ) as http:
        for thinking, expected in ((True, enabled_text), (False, "disabled")):
            response = http.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "chat_template_kwargs": {"enable_thinking": thinking},
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == expected


@pytest.mark.parametrize("response_ids", [(), ("other",), ("call_test", "call_test")])
def test_tool_results_must_resolve_each_call_once(client: TestClient, response_ids: tuple[str, ...]) -> None:
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "call_test", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    }
    messages = [call, *({"role": "tool", "tool_call_id": call_id, "content": "ab"} for call_id in response_ids)]
    response = client.post("/v1/chat/completions", json={"model": "test", "messages": messages})
    assert response.status_code == 400
    valid = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [call, {"role": "tool", "tool_call_id": "call_test", "content": "ab"}]},
    )
    assert valid.status_code == 200


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_tool_schemas_return_request_error(client: TestClient, number: float) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=json.dumps(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "ab"}],
                "tool_choice": "none",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"default": number}}}],
            }
        ),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == "tools.0.function"


def test_context_limit_returns_standard_openai_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "ab " * 65}]},
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "This model's maximum context length is 64 tokens. Your messages resulted in 65 tokens.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }


@pytest.mark.parametrize("stream", [False, True])
def test_multiple_choices_have_independent_indexes_and_combined_usage(client: TestClient, stream: bool) -> None:
    body: dict[str, JSON] = {
        "model": "test",
        "messages": [{"role": "user", "content": "ab X"}],
        "n": 3,
        "stream": stream,
        "max_completion_tokens": 1,
        "logprobs": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        assert all(chunk["usage"] is None for chunk in chunks[:-1])
        assert chunks[-1]["choices"] == []
        usage = chunks[-1]["usage"]
        for index in range(3):
            row = [choice for choice in choices if choice["index"] == index]
            assert row[0]["delta"]["role"] == "assistant"
            assert "".join(choice["delta"].get("content", "") for choice in row) == "ab"
            assert row[-1]["finish_reason"] == "length"
            assert [
                token["token"] for choice in row if choice["logprobs"] for token in choice["logprobs"]["content"]
            ] == ["ab"]
    else:
        payload = response.json()
        choices = payload["choices"]
        usage = payload["usage"]
        assert len(choices) == 3
        assert all(choice["message"]["content"] == "ab" for choice in choices)
        assert all(choice["finish_reason"] == "length" for choice in choices)
        assert all(choice["logprobs"]["content"][0]["token"] == "ab" for choice in choices)
    assert {choice["index"] for choice in choices} == {0, 1, 2}
    assert usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


@pytest.mark.parametrize("include_obfuscation", [None, True, False])
def test_stream_obfuscation_preserves_content_and_pads_enabled_payloads(
    client: TestClient, include_obfuscation: bool | None
) -> None:
    body: dict[str, JSON] = {"model": "test", "messages": [{"role": "user", "content": "ab X"}], "stream": True}
    if include_obfuscation is not None:
        body["stream_options"] = {"include_obfuscation": include_obfuscation}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads.pop() == "[DONE]"
    chunks = [json.loads(payload) for payload in payloads]
    assert "".join(choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"]) == "abX"
    if include_obfuscation is False:
        assert all("obfuscation" not in chunk for chunk in chunks)
    else:
        assert all(isinstance(chunk["obfuscation"], str) for chunk in chunks)
        assert all(len(payload.encode()) % 256 == 0 for payload in payloads)


@pytest.mark.parametrize("seed", [-(2**63), -1, 2**32, 2**63 - 1])
def test_int64_seed_and_explicit_text_flags_are_accepted(client: TestClient, seed: int) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "ab X"}],
            "seed": seed,
            "top_p": 0,
            "response_format": {"type": "text"},
            "modalities": ["text"],
            "service_tier": "default",
            "store": False,
            "metadata": {"purpose": "contract"},
            "safety_identifier": "test-user",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "abX"
    assert payload["service_tier"] == "default"
    assert payload["metadata"] == {"purpose": "contract"}


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
@pytest.mark.parametrize("disconnect_at", ["role", "content"])
def test_disconnect_cancels_every_choice(
    monkeypatch: pytest.MonkeyPatch, spec_version: str, disconnect_at: str
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "ab": 1}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    cancellations: list[Event] = []

    async def disconnect(http: TestClient) -> None:
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
        chunk_seen = asyncio.Event()
        block_send = asyncio.Event()
        requested = False

        async def receive() -> ASGIMessage:
            nonlocal requested
            if not requested:
                requested = True
                return {
                    "type": "http.request",
                    "body": json.dumps(
                        {"model": "test", "messages": [{"role": "user", "content": "ab"}], "stream": True, "n": 3}
                    ).encode(),
                    "more_body": False,
                }
            await chunk_seen.wait()
            return {"type": "http.disconnect"}

        async def send(message: ASGIMessage) -> None:
            if message["type"] != "http.response.body":
                return
            body = message.get("body", b"")
            expected_chunk = b'"role":"assistant"' if disconnect_at == "role" else b'"content":"ab"'
            if expected_chunk in body:
                chunk_seen.set()
                if spec_version == "2.4":
                    raise OSError("Client closed the connection.")
                await block_send.wait()

        assert isinstance(http.app, FastAPI)
        endpoint = next(
            route.endpoint
            for route in http.app.routes
            if isinstance(route, APIRoute) and route.path == "/v1/chat/completions"
        )
        response = await endpoint(Request(scope, receive))
        assert isinstance(response, StreamingResponse)
        assert isinstance(response.body_iterator, AsyncGenerator)
        try:
            if spec_version == "2.4":
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(response(scope, receive, send), 5)
            else:
                await asyncio.wait_for(response(scope, receive, send), 5)
            assert chunk_seen.is_set()
            assert len(cancellations) == 3 and all(cancelled.is_set() for cancelled in cancellations)
        finally:
            await response.body_iterator.aclose()

    with _echo_client(monkeypatch, tokenizer, cancellations=cancellations) as http:
        asyncio.run(disconnect(http))


def test_unexpected_decoder_failure_returns_server_error_and_preserves_next_request(
    monkeypatch: pytest.MonkeyPatch, function_tool: dict[str, JSON]
) -> None:
    malformed = "<tool_call><function=lookup>"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, malformed: 1, "ab": 2}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()

    original_step = ChatDecodeStream.step

    def fail_bad_token(self: ChatDecodeStream, token_id: int) -> tuple[str, str]:
        if token_id == 1:
            raise RuntimeError("Unexpected internal decoder failure.")
        return original_step(self, token_id)

    monkeypatch.setattr(ChatDecodeStream, "step", fail_bad_token)

    async def requests(http: TestClient) -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=http.app, raise_app_exceptions=False),
            base_url="http://test",
            trust_env=False,
        ) as client:
            invalid = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": malformed}],
                        "tools": [function_tool],
                        "n": 3,
                    },
                ),
                5,
            )
            assert invalid.status_code == 500
            assert invalid.json()["error"]["type"] == "server_error"
            valid = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "ab"}]}
                ),
                5,
            )
            assert valid.status_code == 200
            assert valid.json()["choices"][0]["message"]["content"] == "ab"

    with _echo_client(monkeypatch, tokenizer, tool_call_format=ToolCallFormat.QWEN_XML) as http:
        asyncio.run(requests(http))


def test_unexpected_stream_decoder_failure_raises_standard_openai_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = "<tool_call><function=lookup>"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, malformed: 1, "ab": 2}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()

    original_step = ChatDecodeStream.step

    def fail_bad_token(self: ChatDecodeStream, token_id: int) -> tuple[str, str]:
        if token_id == 1:
            raise RuntimeError("Unexpected internal decoder failure.")
        return original_step(self, token_id)

    monkeypatch.setattr(ChatDecodeStream, "step", fail_bad_token)

    async def requests(http: TestClient) -> None:
        async with AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=http.app, raise_app_exceptions=False), trust_env=False
            ),
        ) as client:
            output = await asyncio.wait_for(
                client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": malformed}],
                    tools=[{"type": "function", "function": {"name": "lookup"}}],
                    n=2,
                    stream=True,
                ),
                5,
            )
            chunks = output.__aiter__()
            initial = await asyncio.wait_for(anext(chunks), 5)
            assert {choice.index for choice in initial.choices} == {0, 1}
            assert all(choice.delta.role == "assistant" for choice in initial.choices)
            with pytest.raises(APIError, match="Internal server error"):
                await asyncio.wait_for(anext(chunks), 5)
            valid = await asyncio.wait_for(
                client.chat.completions.create(model="test", messages=[{"role": "user", "content": "ab"}]), 5
            )
            assert valid.choices[0].message.content == "ab"

    with _echo_client(monkeypatch, tokenizer, tool_call_format=ToolCallFormat.QWEN_XML) as http:
        asyncio.run(requests(http))


def test_partial_submission_failure_cancels_choices_and_preserves_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "ab": 1}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    cancellations: list[Event] = []

    async def requests(http: TestClient) -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=http.app, raise_app_exceptions=False),
            base_url="http://test",
            trust_env=False,
        ) as client:
            invalid = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "ab"}], "n": 2},
                ),
                5,
            )
            assert invalid.status_code == 500
            assert invalid.json()["error"]["type"] == "server_error"
            assert len(cancellations) == 1 and cancellations[0].is_set()
            valid = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "ab"}]}
                ),
                5,
            )
            assert valid.status_code == 200
            assert valid.json()["choices"][0]["message"]["content"] == "ab"

    with _echo_client(
        monkeypatch, tokenizer, cancellations=cancellations, submission_error=RuntimeError("GPU submission failed.")
    ) as http:
        asyncio.run(requests(http))


@pytest.mark.parametrize("stream", [False, True])
def test_textual_stop_preserves_completed_tool_calls(
    monkeypatch: pytest.MonkeyPatch, function_tool: dict[str, JSON], stream: bool
) -> None:
    native_call = "<tool_call><function=lookup></function></tool_call>"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens([native_call, "afterZ"])
    tokenizer.decoder = Fuse()
    with _echo_client(monkeypatch, tokenizer, tool_call_format=ToolCallFormat.QWEN_XML) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": native_call + "afterZ"}],
                "tools": [function_tool],
                "n": 2,
                "stop": "Z",
                "logprobs": True,
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        call_ids = []
        for index in range(2):
            row = [choice for choice in choices if choice["index"] == index]
            assert "".join(choice["delta"].get("content", "") for choice in row) == "after"
            assert row[-1]["finish_reason"] == "stop"
            calls = [call for choice in row for call in choice["delta"].get("tool_calls", [])]
            assert len(calls) == 1 and calls[0]["index"] == 0
            assert calls[0]["function"] == {"name": "lookup", "arguments": "{}"}
            call_ids.append(calls[0]["id"])
            assert [
                entry["bytes"] for choice in row if choice["logprobs"] for entry in choice["logprobs"]["content"]
            ] == [list(b"afterZ")]
    else:
        choices = response.json()["choices"]
        assert len(choices) == 2
        assert all(choice["finish_reason"] == "stop" for choice in choices)
        assert all(choice["message"]["content"] == "after" for choice in choices)
        calls = [choice["message"]["tool_calls"][0] for choice in choices]
        assert all(call["function"] == {"name": "lookup", "arguments": "{}"} for call in calls)
        assert all(
            [entry["bytes"] for entry in choice["logprobs"]["content"]] == [list(b"afterZ")] for choice in choices
        )
        call_ids = [call["id"] for call in calls]
    assert len(set(call_ids)) == 2


@pytest.mark.parametrize("tool_format", list(ToolCallFormat))
@pytest.mark.parametrize("stream", [False, True])
def test_native_tool_calls_roundtrip_through_openai_wire(
    monkeypatch: pytest.MonkeyPatch, function_tool: dict[str, JSON], tool_format: ToolCallFormat, stream: bool
) -> None:
    match tool_format:
        case ToolCallFormat.QWEN_XML:
            native_call = (
                "<tool_call><function=lookup><parameter=city>Paris</parameter>"
                "<parameter=count>2</parameter><parameter=active>true</parameter>"
                '<parameter=details>{"x":"y"}</parameter><parameter=items>[1,2]</parameter>'
                "</function></tool_call>"
            )
            tokens = ["visible", native_call]
        case ToolCallFormat.LIQUID:
            native_call = (
                "<|tool_call_start|>[lookup(city='Paris', count=2, active=True, details={'x': 'y'}, items=[1, 2])]"
                "<|tool_call_end|>"
            )
            tokens = ["visible", native_call]
        case ToolCallFormat.MUSE_ATEM:
            native_call = (
                'to=lookup<|message|><atem:function_calls><atem:invoke name="lookup">'
                '<atem:parameter name="city">Paris</atem:parameter><atem:parameter name="count">2</atem:parameter>'
                '<atem:parameter name="active">true</atem:parameter>'
                '<atem:parameter name="details">{"x":"y"}</atem:parameter>'
                '<atem:parameter name="items">[1,2]</atem:parameter>'
                "</atem:invoke></atem:function_calls><|eot|>"
            )
            tokens = ["to=user<|message|>", "visible", "<|eom|>", native_call]
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens([*tokens, "Parisdone"])
    tokenizer.decoder = Fuse()
    history_template = (
        "{% if messages[0].tool_calls is defined %}"
        "{% if messages[1].tool_call_id != messages[0].tool_calls[0].id %}"
        '{{ raise_exception("Tool response has the wrong call ID.") }}{% endif %}'
        "{{ messages[0].tool_calls[0].function.arguments.city }}{{ messages[1].content }}"
        "{% else %}{{ messages[0].content }}{% endif %}"
    )
    with _echo_client(monkeypatch, tokenizer, tool_call_format=tool_format, prompt_template=history_template) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "".join(tokens)}],
                "tools": [function_tool],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "max_completion_tokens": len(tokens),
                "n": 2,
                "logprobs": True,
                "stream": stream,
            },
        )
        assert response.status_code == 200
        if stream:
            chunks = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [choice for chunk in chunks for choice in chunk["choices"]]
            messages = []
            for index in range(2):
                row = [choice for choice in choices if choice["index"] == index]
                assert row[-1]["finish_reason"] == "length"
                calls = [call for choice in row for call in choice["delta"].get("tool_calls", [])]
                assert len(calls) == 1 and calls[0]["index"] == 0
                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(choice["delta"].get("content", "") for choice in row),
                        "tool_calls": [
                            {key: value for key, value in call.items() if key != "index"} for call in calls
                        ],
                    }
                )
                logprobs = [entry for choice in row if choice["logprobs"] for entry in choice["logprobs"]["content"]]
                assert [entry["bytes"] for entry in logprobs] == [list(b"visible")]
        else:
            choices = response.json()["choices"]
            assert len(choices) == 2 and all(choice["finish_reason"] == "length" for choice in choices)
            messages = [choice["message"] for choice in choices]
            assert all(
                [entry["bytes"] for entry in choice["logprobs"]["content"]] == [list(b"visible")] for choice in choices
            )
        assert all(message["content"] == "visible" for message in messages)
        calls = [message["tool_calls"][0] for message in messages]
        assert len({call["id"] for call in calls}) == 2
        for call in calls:
            assert call["type"] == "function" and call["function"]["name"] == "lookup"
            assert json.loads(call["function"]["arguments"]) == {
                "city": "Paris",
                "count": 2,
                "active": True,
                "details": {"x": "y"},
                "items": [1, 2],
            }
        history = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [messages[0], {"role": "tool", "content": "done", "tool_call_id": calls[0]["id"]}],
                "tools": [function_tool],
            },
        )
        assert history.status_code == 200
        assert history.json()["choices"][0]["message"]["content"] == "Parisdone"
        disabled = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "".join(tokens)}],
                "tools": [function_tool],
                "tool_choice": "none",
            },
        )
        assert disabled.status_code == 200
        assert disabled.json()["choices"][0]["message"]["content"] == "".join(tokens)
        assert "tool_calls" not in disabled.json()["choices"][0]["message"]


@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.MUSE_ATEM, ToolCallFormat.LIQUID])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("schema", "value"),
    [
        (schema, 42)
        for schema in [
            {
                "$defs": {"count": {"allOf": [{"type": "number"}, {"type": "integer"}]}},
                "properties": {"count": scalar},
            }
            for scalar in [
                {"type": "integer"},
                {"allOf": [{"type": "number"}, {"type": "integer"}, {"minimum": 0}]},
                {"$ref": "#/$defs/count"},
                {"allOf": [{"type": "integer"}, {}, {"anyOf": [{"type": "string"}, {}]}]},
                {"allOf": [{"type": ["integer", "null"]}, {"minimum": 0}]},
            ]
        ]
        + [
            {"allOf": [{"properties": {"count": {"type": "integer"}}}]},
            {
                "$defs": {"arguments": {"properties": {"count": {"type": "integer"}}}},
                "$ref": "#/$defs/arguments",
            },
            {
                "anyOf": [
                    {"properties": {"count": {"type": "integer", "minimum": 0}}},
                    {"properties": {"count": {"type": "integer", "maximum": -1}}},
                ]
            },
            {
                "oneOf": [
                    {"properties": {"count": {"type": "integer", "minimum": 0}}},
                    {"properties": {"count": {"type": "integer", "maximum": -1}}},
                ]
            },
            {
                "properties": {"count": {"type": ["string", "number"]}},
                "allOf": [{"properties": {"count": {"type": "integer"}}}],
            },
            {"$ref": "#/anyOf/0", "anyOf": [{"properties": {"count": {"type": "integer"}}}, {}]},
            {
                "$defs": {"arguments": {"properties": {"count": {"type": "integer"}}}},
                "$ref": "#/%24defs/arguments",
            },
            *[
                {
                    combination: [
                        {"type": "string", "properties": {"count": {"type": "string"}}},
                        {"properties": {"count": {"type": "integer"}}},
                    ]
                }
                for combination in ("anyOf", "oneOf")
            ],
            {
                "allOf": [
                    {"type": "object"},
                    {
                        "anyOf": [
                            {"type": "string", "properties": {"count": {"type": "string"}}},
                            {"properties": {"count": {"type": "integer"}}},
                        ]
                    },
                ]
            },
            {
                "$defs": {"excluded": {"allOf": [{"type": "array"}, {"properties": {"count": {"type": "string"}}}]}},
                "anyOf": [{"$ref": "#/$defs/excluded"}, {"properties": {"count": {"type": "integer"}}}],
            },
            {
                "anyOf": [
                    {"type": "string", "properties": {"invalid-name": {"type": "string"}}},
                    {"properties": {"count": {"type": "integer"}}},
                ]
            },
        ]
    ]
    + [({"properties": {"count": {"$ref": "#"}}}, {})],
)
def test_composed_tool_parameter_schemas_preserve_argument_types(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, stream: bool, schema: dict[str, JSON], value: JSON
) -> None:
    argument = json.dumps(value)
    if tool_format is ToolCallFormat.QWEN_XML:
        native = f"<tool_call><function=lookup><parameter=count>{argument}</parameter></function></tool_call>"
    elif tool_format is ToolCallFormat.LIQUID:
        native = f"<|tool_call_start|>[lookup(count={argument})]<|tool_call_end|>"
    else:
        native = (
            'to=lookup<|message|><atem:function_calls><atem:invoke name="lookup">'
            f'<atem:parameter name="count">{argument}</atem:parameter></atem:invoke></atem:function_calls><|eot|>'
        )
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, native: 1}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    with _echo_client(monkeypatch, tokenizer, tool_call_format=tool_format) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": native}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", **schema},
                        },
                    }
                ],
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        calls = [
            call
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
            for choice in json.loads(line.removeprefix("data: "))["choices"]
            for call in choice["delta"].get("tool_calls", [])
        ]
    else:
        calls = response.json()["choices"][0]["message"]["tool_calls"]
    (call,) = calls
    assert json.loads(call["function"]["arguments"]) == {"count": value}


@pytest.mark.parametrize("tool_format", [ToolCallFormat.QWEN_XML, ToolCallFormat.MUSE_ATEM, ToolCallFormat.LIQUID])
@pytest.mark.parametrize("combination", ["$ref", "allOf", "anyOf", "oneOf"])
def test_composed_tool_roots_reject_unsupported_native_arguments(
    monkeypatch: pytest.MonkeyPatch, tool_format: ToolCallFormat, combination: str
) -> None:
    unsupported: dict[str, JSON]
    if tool_format is ToolCallFormat.LIQUID:
        unsupported = {"type": "object", "properties": {"invalid name": {"type": "integer"}}}
        message = "cannot be represented"
    else:
        unsupported = {"type": "object", "properties": {"count": {"type": ["string", "integer"]}}}
        message = "mixes string and nonstring"
    schema: dict[str, JSON]
    if combination == "$ref":
        schema = {"$defs": {"arguments": unsupported}, "$ref": "#/$defs/arguments"}
    else:
        schema = {combination: [unsupported]}
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "answer": 1}, unk_token="[UNK]"))
    tokenizer.decoder = Fuse()
    with _echo_client(monkeypatch, tokenizer, tool_call_format=tool_format) as http:
        request = {"model": "test", "messages": [{"role": "user", "content": "answer"}]}
        rejected = http.post(
            "/v1/chat/completions",
            json=request | {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]},
        )
        assert rejected.status_code == 400
        assert message in rejected.json()["error"]["message"]
        assert http.post("/v1/chat/completions", json=request).json()["choices"][0]["message"]["content"] == "answer"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("text", "tool_format"), [("<", None), ("<X", None), ("<tool_c", ToolCallFormat.QWEN_XML)])
def test_literal_marker_prefix_preserves_logprobs_when_released(
    monkeypatch: pytest.MonkeyPatch,
    function_tool: dict[str, JSON],
    stream: bool,
    text: str,
    tool_format: ToolCallFormat | None,
) -> None:
    characters = list(dict.fromkeys(text))
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens(characters)
    tokenizer.decoder = Fuse()
    with _echo_client(
        monkeypatch, tokenizer, output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX, tool_call_format=tool_format
    ) as http:
        body: dict[str, JSON] = {
            "model": "test",
            "messages": [{"role": "user", "content": text}],
            "logprobs": True,
            "stream": stream,
        }
        if tool_format is not None:
            body["tools"] = [function_tool]
        response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if stream:
        choices = [
            json.loads(line.removeprefix("data: "))["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        logprobs = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
    else:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        logprobs = choice["logprobs"]["content"]
    assert content == text
    assert [entry["token"] for entry in logprobs] == list(text)
    assert [entry["bytes"] for entry in logprobs] == [list(character.encode()) for character in text]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("prefix", "parts", "content", "reasoning", "expected_bytes"),
    [
        ("<think>", ("R", "</think>", "A", "</think>"), "A</think>", "R", (b"A", b"</think>")),
        ("<think></think>", ("A", "</think>"), "A</think>", "", (b"A", b"</think>")),
        ("prompt", ("reasoning", "</think>", "answer"), "answer", "reasoning", (b"answer",)),
        ("prompt", ("ð",), "�", "", (b"\xf0",)),
        ("prompt", ("ð", "Ł", "ĺ"), "�", "", (b"\xf0", b"\x9f", b"\x98")),
        (
            "prompt",
            ("<think>", "ð", "</think>A" + "ÿ" * 20),
            "A" + "�" * 20,
            "�",
            (b"</think>A" + b"\xff" * 20,),
        ),
        (
            "prompt",
            ("<think>", "ð", "Ł", "ĺ", "Ģ</think>A"),
            "A",
            "😀",
            (b"\x80</think>A",),
        ),
    ],
)
def test_content_logprobs_preserve_only_tokens_with_visible_bytes(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    prefix: str,
    parts: tuple[str, ...],
    content: str,
    reasoning: str,
    expected_bytes: tuple[bytes, ...],
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_tokens([*parts, prefix])
    tokenizer.decoder = ByteLevel()
    output = tuple(tokenizer.token_to_id(part) for part in parts)
    with _echo_client(
        monkeypatch,
        tokenizer,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        prompt_template=prefix,
        output_token_ids=output,
    ) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "logprobs": True,
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert all("error" not in event for event in events)
        choices = [choice for event in events for choice in event["choices"]]
        actual_content = "".join(choice["delta"].get("content", "") for choice in choices)
        actual_reasoning = "".join(choice["delta"].get("reasoning_content", "") for choice in choices)
        logprobs = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
    else:
        choice = response.json()["choices"][0]
        actual_content = choice["message"]["content"]
        actual_reasoning = choice["message"].get("reasoning_content", "")
        logprobs = choice["logprobs"]["content"]
    assert (actual_content, actual_reasoning) == (content, reasoning)
    assert [entry["bytes"] for entry in logprobs] == [list(raw) for raw in expected_bytes]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("decoder_family", ["byte_level", "byte_level_sequence", "byte_fallback"])
@pytest.mark.parametrize("output_parser_regex", [None, OPTIONAL_THINKING_OUTPUT_PARSER_REGEX])
def test_unicode_logprobs_preserve_each_tokens_raw_bytes(
    monkeypatch: pytest.MonkeyPatch, stream: bool, decoder_family: str, output_parser_regex: str | None
) -> None:
    if decoder_family == "byte_fallback":
        vocabulary = {f"<0x{byte:02X}>": index for index, byte in enumerate("😀".encode())}
        tokenizer = Tokenizer(BPE(vocabulary, [], byte_fallback=True))
        tokenizer.decoder = DecoderSequence([ByteFallback(), Fuse()])
    else:
        tokenizer = Tokenizer(BPE({character: index for index, character in enumerate("ðŁĺĢ")}, []))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        if decoder_family == "byte_level_sequence":
            tokenizer.decoder = DecoderSequence([ByteLevel(), Fuse()])
        else:
            tokenizer.decoder = ByteLevel()
    with _echo_client(monkeypatch, tokenizer, output_parser_regex=output_parser_regex) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "😀"}],
                "logprobs": True,
                "top_logprobs": 1,
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        choices = [
            json.loads(line.removeprefix("data: "))["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(choice["delta"].get("content", "") for choice in choices)
        logprobs = [entry for choice in choices if choice["logprobs"] for entry in choice["logprobs"]["content"]]
    else:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        logprobs = choice["logprobs"]["content"]
    assert content == "😀"
    assert [entry["bytes"] for entry in logprobs] == [[byte] for byte in content.encode()]
    assert [entry["token"] for entry in logprobs] == ["�"] * 4
    assert [entry["top_logprobs"][0]["bytes"] for entry in logprobs] == [[byte] for byte in content.encode()]
    assert [entry["top_logprobs"][0]["token"] for entry in logprobs] == ["�"] * 4


def test_metaspace_logprobs_preserve_visible_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = Tokenizer(WordLevel({"▁foo": 0, "▁bar": 1, "▁": 2, "prompt": 3}, unk_token="prompt"))
    tokenizer.decoder = Metaspace()
    with _echo_client(monkeypatch, tokenizer, prompt_template="prompt", output_token_ids=(0, 1, 2)) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "prompt"}], "logprobs": True},
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "foo bar "
    assert [entry["bytes"] for entry in choice["logprobs"]["content"]] == [list(b" foo"), list(b" bar"), list(b" ")]


@pytest.mark.parametrize("byte_level", [False, True])
def test_byte_fallback_spelling_is_literal_without_byte_fallback_decoder(
    monkeypatch: pytest.MonkeyPatch, byte_level: bool
) -> None:
    tokenizer = Tokenizer(BPE({"x": 0}, []))
    tokenizer.add_tokens(["<0xF0>"])
    if byte_level:
        tokenizer.decoder = ByteLevel()
    else:
        tokenizer.decoder = Fuse()
    with _echo_client(monkeypatch, tokenizer) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "<0xF0>"}],
                "logprobs": True,
                "top_logprobs": 1,
            },
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "<0xF0>"
    (token,) = choice["logprobs"]["content"]
    assert token["bytes"] == list(b"<0xF0>")
    assert token["top_logprobs"][0]["bytes"] == token["bytes"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("penalty", [1e300, 1e-300])
def test_unrepresentable_sampler_controls_fail_before_streaming(
    client: TestClient, stream: bool, penalty: float
) -> None:
    valid = {"model": "test", "messages": [{"role": "user", "content": "ab X"}]}
    response = client.post("/v1/chat/completions", json=valid | {"repetition_penalty": penalty, "stream": stream})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "repetition_penalty" in response.json()["error"]["message"]
    subsequent = client.post("/v1/chat/completions", json=valid | {"top_k": 2**100})
    assert subsequent.status_code == 200
    assert subsequent.json()["choices"][0]["message"]["content"] == "abX"


@pytest.mark.parametrize("stream", [False, True])
def test_textual_stop_ends_generation_before_queued_tokens(client: TestClient, stream: bool) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "ab X"}], "stop": "ab", "stream": stream},
    )
    assert response.status_code == 200
    if stream:
        choices = [
            json.loads(line.removeprefix("data: "))["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert "".join(choice["delta"].get("content", "") for choice in choices) == ""
        assert choices[-1]["finish_reason"] == "stop"
    else:
        body = response.json()
        assert body["choices"][0]["message"]["content"] == ""
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 1


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(mutation=("n", True))
@example(mutation=("n", 1.0))
@example(mutation=("messages", [{"role": "user", "content": None}]))
@example(mutation=("messages", [{"role": "tool", "tool_call_id": "orphan", "content": "ab"}]))
@example(mutation=("messages", [{"role": "user", "content": "ab", "reasoning_content": "ignored"}]))
@example(mutation=("temperature", 2.1))
@example(mutation=("top_p", 1.1))
@example(mutation=("presence_penalty", -2.1))
@given(
    mutation=st.one_of(
        st.tuples(
            st.sampled_from(["max_tokens", "top_k", "seed", "n", "temperature", "top_p", "repetition_penalty"]),
            st.one_of(st.booleans(), st.text(max_size=12), st.lists(st.integers(), max_size=2)),
        ),
        st.tuples(st.sampled_from(["max_tokens", "top_k", "seed", "n", "top_logprobs"]), st.floats()),
        st.tuples(
            st.sampled_from(["max_tokens", "max_completion_tokens", "repetition_penalty"]), st.integers(max_value=0)
        ),
        st.tuples(st.sampled_from(["top_k", "top_logprobs"]), st.integers(max_value=-1)),
        st.tuples(
            st.sampled_from(
                ["temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"]
            ),
            st.sampled_from([float("inf"), float("-inf"), float("nan")]),
        ),
        st.sampled_from(
            [
                ("n", 129),
                ("seed", 2**63),
                ("seed", -(2**63) - 1),
                ("top_logprobs", 21),
                ("logit_bias", {"wrong": 1}),
                ("logit_bias", {"-1": 1}),
                ("logit_bias", {"1": 101}),
                ("logit_bias", {"1": True}),
                ("logit_bias", {"1": 1, "01": 1}),
                ("logit_bias", {str(2**32): 1}),
                ("response_format", {"type": "wrong"}),
                ("modalities", ["wrong"]),
                ("service_tier", "wrong"),
                ("store", True),
                ("stream_options", {"include_obfuscation": "true"}),
                ("metadata", {"x" * 65: "v"}),
                ("metadata", {"key": "x" * 513}),
                ("metadata", {str(index): "v" for index in range(17)}),
                ("messages", []),
                ("messages", [{"role": None, "content": "ab"}]),
                ("messages", [{"role": "unknown", "content": "ab"}]),
                ("messages", [{"role": "user", "content": 7}]),
                ("messages", [{"role": "user", "content": [{"type": "text", "text": None}]}]),
            ]
        ),
    )
)
def test_invalid_requests_do_not_poison_generation(client: TestClient, mutation: tuple[str, JSON]) -> None:
    field, value = mutation
    valid = {"model": "test", "messages": [{"role": "user", "content": "ab X"}]}
    invalid = client.post("/v1/chat/completions", content=json.dumps(valid | {field: value}))
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    response = client.post("/v1/chat/completions", json=valid)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "abX"


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(body=st.binary(max_size=128))
def test_malformed_json_does_not_poison_generation(client: TestClient, body: bytes) -> None:
    invalid = client.post("/v1/chat/completions", content=b"{" + body + b"\x00")
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    response = client.post(
        "/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "ab"}]}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ab"
