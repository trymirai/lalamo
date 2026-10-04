import asyncio
import json
import logging
import secrets
import threading
import time
import traceback
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing, asynccontextmanager
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import uvicorn
import xgrammar
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from starlette.types import Receive, Scope, Send

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    GrammarConstraintError,
    SequenceFinished,
    TokenEvent,
)
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import (
    AssistantMessage,
    Message,
    ReasoningConfig,
    ReasoningEffort,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from lalamo.models.json_schema import check_json_schema_syntax, validate_json_schema
from lalamo.utils.json import JSON
from lalamo.utils.sharding import ShardingConfig

logger = logging.getLogger("lalamo.server")
_tool_json_adapter = TypeAdapter(dict[str, JSON], config=ConfigDict(strict=True, allow_inf_nan=False))

type OpenAIName = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]


class ChatRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCallParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: OpenAIName
    arguments: str


class ToolCallParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: Annotated[str, Field(min_length=1)]
    type: Literal["function"]
    function: FunctionCallParam

    def to_tool_call(self) -> ToolCall:
        # OpenAI arguments are encoded JSON; the chat template consumes a JSON object.
        arguments = _tool_json_adapter.validate_json(self.function.arguments)
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": arguments}}


class FunctionToolParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: OpenAIName
    description: str | None = None
    parameters: dict[str, JSON] | None = None
    strict: bool | None = None

    @model_validator(mode="after")
    def check_parameters(self) -> "FunctionToolParam":
        if self.parameters is not None:
            if self.strict:
                validate_json_schema(self.parameters, strict=True)
            else:
                check_json_schema_syntax(self.parameters)
        return self


class ToolParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["function"]
    function: FunctionToolParam


class FunctionNameParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: OpenAIName


class NamedToolChoiceParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["function"]
    function: FunctionNameParam


class AllowedToolsParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["auto", "required"]
    tools: list[NamedToolChoiceParam]


class AllowedToolChoiceParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["allowed_tools"]
    allowed_tools: AllowedToolsParam


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text", "json_object"]


class JsonSchemaParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: OpenAIName
    description: str = ""
    schema_: dict[str, JSON] = Field(default_factory=dict, alias="schema")
    strict: bool | None = None

    @model_validator(mode="after")
    def check_schema(self) -> "JsonSchemaParam":
        validate_json_schema(self.schema_, strict=bool(self.strict))
        return self


class JsonSchemaResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["json_schema"]
    json_schema: JsonSchemaParam


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str


class RefusalPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["refusal"]
    refusal: str


class ChatMessageParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: ChatRole
    content: str | list[TextPart | RefusalPart] | None = None
    refusal: str | None = None
    audio: None = None
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallParam] | None = None
    tool_call_id: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def check_role_fields(self) -> "ChatMessageParam":
        if self.role is ChatRole.ASSISTANT and self.content == []:
            raise ValueError("Assistant content arrays must contain at least one part.")
        if self.role is not ChatRole.ASSISTANT and self.content is None:
            raise ValueError("content is required for this message role.")
        if self.role is not ChatRole.ASSISTANT and self.refusal is not None:
            raise ValueError("refusal is supported only for assistant messages.")
        if isinstance(self.content, list) and any(isinstance(part, RefusalPart) for part in self.content):
            if self.role is not ChatRole.ASSISTANT:
                raise ValueError("Refusal content parts are supported only for assistant messages.")
            if len(self.content) != 1:
                raise ValueError("A refusal content part must be the only content part.")
        if self.role is not ChatRole.ASSISTANT and self.reasoning_content is not None:
            raise ValueError("reasoning_content is supported only for assistant messages.")
        if self.tool_calls is not None and self.role is not ChatRole.ASSISTANT:
            raise ValueError("tool_calls is supported only for assistant messages.")
        if (self.tool_call_id is not None) != (self.role is ChatRole.TOOL):
            raise ValueError("tool_call_id is required only for tool messages.")
        if self.role is ChatRole.ASSISTANT and self.content is None and self.refusal is None and not self.tool_calls:
            raise ValueError("Assistant messages require content, refusal, or tool_calls.")
        return self

    def to_message(self) -> Message:
        content = self.content or ""
        if isinstance(content, list):
            content = "".join(part.text if isinstance(part, TextPart) else part.refusal for part in content)
        if self.refusal is not None:
            content += self.refusal
        match self.role:
            case ChatRole.USER:
                return UserMessage(content, name=self.name)
            case ChatRole.ASSISTANT:
                return AssistantMessage(
                    self.reasoning_content,
                    content,
                    tuple(call.to_tool_call() for call in self.tool_calls or ()),
                    name=self.name,
                )
            case ChatRole.TOOL:
                return ToolMessage(content, self.name, self.tool_call_id)
            case ChatRole.SYSTEM | ChatRole.DEVELOPER:
                return SystemMessage(content, name=self.name)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool | None = False
    include_obfuscation: bool | None = True


class ChatTemplateKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enable_thinking: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    messages: Annotated[list[ChatMessageParam], Field(min_length=1)]
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    max_completion_tokens: Annotated[int, Field(gt=0)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    min_p: Annotated[float, Field(ge=0, le=1)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = Field(
        None, validation_alias=AliasChoices("repetition_penalty", "repeat_penalty")
    )
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    seed: Annotated[int, Field(ge=-(2**63), le=2**63 - 1)] | None = None
    logit_bias: dict[str, Annotated[float, Field(ge=-100, le=100)]] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None
    logprobs: bool | None = False
    top_logprobs: Annotated[int, Field(ge=0, le=20)] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    stop: (
        Annotated[str, Field(min_length=1)]
        | Annotated[list[Annotated[str, Field(min_length=1)]], Field(max_length=4)]
        | None
    ) = None
    n: Annotated[int, Field(ge=1, le=128)] | None = 1
    tools: Annotated[list[ToolParam], Field(max_length=128)] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoiceParam | AllowedToolChoiceParam | None = None
    parallel_tool_calls: bool | None = True
    response_format: Annotated[ResponseFormat | JsonSchemaResponseFormat, Field(discriminator="type")] | None = None
    modalities: Annotated[list[Literal["text"]], Field(min_length=1, max_length=1)] | None = None
    service_tier: Literal["auto", "default"] | None = None
    metadata: (
        Annotated[
            dict[Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=512)]], Field(max_length=16)
        ]
        | None
    ) = None
    user: str | None = None
    safety_identifier: Annotated[str, Field(max_length=64)] | None = None
    store: bool | None = None
    verbosity: None = None
    prediction: None = None
    prompt_cache_key: None = None
    prompt_cache_retention: None = None
    moderation: None = None
    audio: None = None

    @field_validator("logit_bias")
    @classmethod
    def check_logit_bias(cls, biases: dict[str, float] | None) -> dict[str, float] | None:
        if biases is not None and any(not token.isascii() or not token.isdecimal() for token in biases):
            raise ValueError("logit_bias keys must be non-negative token IDs.")
        if biases is not None and len({int(token) for token in biases}) != len(biases):
            raise ValueError("logit_bias keys must identify distinct tokens.")
        return biases

    @model_validator(mode="after")
    def check_consistency(self) -> "ChatCompletionRequest":
        if self.tool_choice == "required" or isinstance(self.tool_choice, NamedToolChoiceParam):
            if not self.tools:
                raise ValueError("A forced tool choice requires tools.")
            if (
                isinstance(self.tool_choice, NamedToolChoiceParam)
                and sum(tool.function.name == self.tool_choice.function.name for tool in self.tools) != 1
            ):
                raise ValueError("A named tool choice must identify exactly one supplied function.")
        if isinstance(self.tool_choice, AllowedToolChoiceParam):
            allowed_names = {tool.function.name for tool in self.tool_choice.allowed_tools.tools}
            if allowed_names - {tool.function.name for tool in self.tools or ()}:
                raise ValueError("Allowed tools must identify supplied functions.")
            if self.tool_choice.allowed_tools.mode == "required" and not allowed_names:
                raise ValueError("A required allowed-tools choice needs at least one function.")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true.")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true.")
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            raise ValueError("Specify only one of max_completion_tokens and max_tokens.")
        if self.store:
            raise ValueError("Stored completions are not supported.")
        if self.tools is not None and len({tool.function.name for tool in self.tools}) != len(self.tools):
            raise ValueError("Tool function names must be distinct.")
        if (
            self.reasoning_effort is not None
            and self.chat_template_kwargs is not None
            and self.chat_template_kwargs.enable_thinking is not None
            and self.chat_template_kwargs.enable_thinking != (self.reasoning_effort != "none")
        ):
            raise ValueError("reasoning_effort conflicts with chat_template_kwargs.enable_thinking.")
        pending_calls: set[str] = set()
        for message in self.messages:
            if message.role is ChatRole.TOOL:
                call_id = message.tool_call_id
                if call_id is None or call_id not in pending_calls:
                    raise ValueError("Tool messages must respond to an outstanding tool_call_id.")
                pending_calls.remove(call_id)
            else:
                if pending_calls:
                    raise ValueError("Every tool call must receive a tool response before the next message.")
                if message.tool_calls:
                    pending_calls = {call.id for call in message.tool_calls}
                    if len(pending_calls) != len(message.tool_calls):
                        raise ValueError("Tool-call IDs must be distinct within an assistant message.")
        if pending_calls:
            raise ValueError("Every tool call must receive a tool response before requesting a completion.")
        return self

    @property
    def stop_strings(self) -> list[str]:
        return [self.stop] if isinstance(self.stop, str) else self.stop or []

    def effective_reasoning_effort(self, reasoning_config: ReasoningConfig | None) -> ReasoningEffort | None:
        if self.reasoning_effort == "none":
            return ReasoningEffort.NO_REASONING
        if self.reasoning_effort is not None:
            return ReasoningEffort(self.reasoning_effort)
        if self.chat_template_kwargs is None or self.chat_template_kwargs.enable_thinking is None:
            return None
        if reasoning_config is None:
            raise ValueError("This model does not support configurable reasoning effort.")
        return reasoning_config.effort_for_thinking(enabled=self.chat_template_kwargs.enable_thinking)


class Chunk(NamedTuple):
    reasoning: str
    content: str
    logprobs: list[dict[str, Any]]
    finished: SequenceFinished | None
    tool_calls: tuple[dict[str, Any], ...] = ()


def _openai_error(message: str, status: int, param: str | None = None, code: str | None = None) -> JSONResponse:
    if status >= 500:
        error_type = "server_error"
    else:
        error_type = "invalid_request_error"
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": error_type, "param": param, "code": code}}
    )


def _sse(payload: dict[str, Any], *, obfuscate: bool = False) -> str:
    if obfuscate:
        padded = payload | {"obfuscation": ""}
        padding = -len(json.dumps(padded, separators=(",", ":"))) % 256
        payload = padded | {"obfuscation": secrets.token_hex(128)[:padding]}
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _choice(**fields: object) -> dict[str, object]:
    return {"index": 0, "finish_reason": None, "logprobs": None} | fields


def _logprob_entry(token_bytes: bytes, logprob: float) -> dict[str, Any]:
    return {"token": token_bytes.decode("utf-8", errors="replace"), "bytes": list(token_bytes), "logprob": logprob}


def create_app(model: LanguageModel, model_name: str, config: ContinuousBatchingConfig) -> FastAPI:
    engine = ContinuousBatchingEngine(model, config)

    @cache
    def grammar_compiler() -> xgrammar.GrammarCompiler:
        tokenizer_info = xgrammar.TokenizerInfo(
            [
                model.token_codec.decode_token_bytes(token)
                for token in range(model.token_codec.tokenizer.get_vocab_size())
            ],
            vocab_size=model.decoder.vocab_size,
            stop_token_ids=list(model.config.generation_config.stop_token_ids),
        )
        return xgrammar.GrammarCompiler(tokenizer_info, cache_limit_bytes=64 * 2**20)

    stop_event = threading.Event()
    engine_errors: list[BaseException] = []

    def run_engine() -> None:
        try:
            while not stop_event.is_set():
                if not engine.step():
                    stop_event.wait(0.001)
        except Exception as error:  # noqa: BLE001
            engine_errors.append(error)
            traceback.print_exception(error)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop_event.clear()
        worker = threading.Thread(target=run_engine, name="lalamo-continuous", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop_event.set()
            await asyncio.to_thread(worker.join)

    api = FastAPI(lifespan=lifespan)

    @api.exception_handler(Exception)
    async def unhandled_exception(_request: Request, error: Exception) -> JSONResponse:
        traceback.print_exception(error)
        return _openai_error("Internal server error.", 500)

    model_object = {"id": model_name, "object": "model", "created": 0, "owned_by": "lalamo"}

    @api.get("/health")
    @api.get("/v1/health")
    async def health() -> Response:
        if engine_errors:
            return _openai_error("Continuous inference engine failed.", 503)
        return JSONResponse({"status": "ok"})

    @api.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {"object": "list", "data": [model_object]}

    @api.get("/v1/models/{requested_model:path}")
    async def retrieve_model(requested_model: str) -> Response:
        if requested_model == model_name:
            return JSONResponse(content=model_object)
        return _openai_error(f"Model {requested_model!r} does not exist.", 404, "model", "model_not_found")

    @api.post("/v1/chat/completions")
    async def complete(request: Request) -> Response:
        if engine_errors:
            raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
        try:
            body = ChatCompletionRequest.model_validate_json(await request.body())
        except ValidationError as error:
            (first_error, *_) = error.errors()
            error_message = first_error["msg"].removeprefix("Value error, ")
            location = first_error["loc"]
            # Pydantic adds branch labels to union errors; only selected tagged branches have a precise nested path.
            if location[:1] == ("response_format",) and len(location) > 1:
                location = (location[0], *location[2:])
            elif location[:1] == ("tool_choice",):
                location = location[:1]
                error_message = "Invalid tool_choice."
            elif location[:1] == ("messages",) and location[2:3] == ("content",):
                location = location[:3]
                error_message = "Invalid message content."
            return _openai_error(error_message, 400, ".".join(map(str, location)) or None)
        if body.model != model_name:
            return _openai_error(f"Model {body.model!r} does not exist.", 404, "model", "model_not_found")

        tools = None
        if body.tools and body.tool_choice != "none":
            tools = [tool.model_dump(exclude_none=True) for tool in body.tools]
        try:
            reasoning_effort = body.effective_reasoning_effort(model.token_codec.config.reasoning_config)
            messages = [message.to_message() for message in body.messages]
            response_schema: dict[str, JSON] | None = None
            if body.response_format is not None and body.response_format.type == "json_object":
                for input_message in messages:
                    if isinstance(input_message, AssistantMessage):
                        content = input_message.response
                    else:
                        content = input_message.content
                    if "json" in content.lower():
                        break
                else:
                    return _openai_error(
                        "JSON mode requires a message instructing the model to produce JSON.", 400, "messages"
                    )
                response_schema = {"type": "object", "additionalProperties": True}
            elif isinstance(body.response_format, JsonSchemaResponseFormat):
                response_schema = body.response_format.json_schema.schema_
                instruction = "Respond with JSON matching this response format:\n" + json.dumps(
                    body.response_format.json_schema.model_dump(by_alias=True, exclude={"strict"}),
                    ensure_ascii=False,
                )
                if isinstance(messages[0], SystemMessage):
                    messages[0] = SystemMessage(messages[0].content + "\n\n" + instruction, name=messages[0].name)
                else:
                    messages.insert(0, SystemMessage(instruction))
            prompt_token_ids = model.token_codec.encode_request(
                messages, tools=tools, reasoning_effort=reasoning_effort
            )
        except (TypeError, ValueError) as error:
            return _openai_error(str(error), 400, "messages")
        # Like llama.cpp, the requested output budget is clamped to the room left after the prompt.
        remaining_context = engine.context_limit - len(prompt_token_ids)
        max_tokens = min(body.max_completion_tokens or body.max_tokens or remaining_context, remaining_context)
        if max_tokens < 1:
            return _openai_error(
                f"This model's maximum context length is {engine.context_limit} tokens. "
                f"Your messages resulted in {len(prompt_token_ids)} tokens.",
                400,
                "messages",
                "context_length_exceeded",
            )

        logit_bias = None
        if body.logit_bias is not None:
            logit_bias = tuple((int(token), bias) for token, bias in body.logit_bias.items())
        generation_config = model.config.generation_config.override_with(
            GenerationConfig(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                min_p=body.min_p,
                repetition_penalty=body.repetition_penalty,
                presence_penalty=body.presence_penalty,
                frequency_penalty=body.frequency_penalty,
                logit_bias=logit_bias,
            )
        )
        request_id = f"chatcmpl-{secrets.token_hex(16)}"
        loop = asyncio.get_running_loop()
        event_batches: list[asyncio.Queue[Sequence[TokenEvent]]] = [asyncio.Queue() for _ in range(body.n or 1)]
        call_tools = tools
        if isinstance(body.tool_choice, NamedToolChoiceParam):
            assert tools is not None
            call_tools = [tool for tool in tools if tool["function"]["name"] == body.tool_choice.function.name]
        elif isinstance(body.tool_choice, AllowedToolChoiceParam):
            allowed_names = {tool.function.name for tool in body.tool_choice.allowed_tools.tools}
            call_tools = [tool for tool in tools or () if tool["function"]["name"] in allowed_names] or None
        decoders = [
            model.token_codec.decode_stream(
                reasoning_effort,
                prompt=model.token_codec.decode_tokens(prompt_token_ids),
                tools=call_tools,
                stop_strings=tuple(body.stop_strings),
                parallel_tool_calls=False
                if isinstance(body.tool_choice, NamedToolChoiceParam)
                else body.parallel_tool_calls,
                response_schema=response_schema,
            )
            for _ in event_batches
        ]
        compiled_grammar = None
        if response_schema is not None or (
            call_tools
            and (
                body.tool_choice == "required"
                or isinstance(body.tool_choice, (NamedToolChoiceParam, AllowedToolChoiceParam))
                or any(tool["function"].get("strict") is True for tool in call_tools)
            )
        ):
            try:
                require_call = (
                    body.tool_choice == "required"
                    or isinstance(body.tool_choice, NamedToolChoiceParam)
                    or (
                        isinstance(body.tool_choice, AllowedToolChoiceParam)
                        and body.tool_choice.allowed_tools.mode == "required"
                    )
                )
                grammar: xgrammar.Grammar | None = None
                if response_schema is not None and not require_call:
                    grammar = await asyncio.to_thread(
                        model.token_codec.json_response_grammar, response_schema, prefix=decoders[0].prefix
                    )
                if call_tools:
                    tool_grammar = await asyncio.to_thread(
                        xgrammar.Grammar.from_ebnf,
                        model.token_codec.tool_call_grammar(
                            call_tools,
                            prefix=decoders[0].prefix,
                            require_call=require_call or response_schema is not None,
                            response_schema=response_schema,
                        ),
                    )
                    if grammar is not None and not require_call:
                        grammar = await asyncio.to_thread(xgrammar.Grammar.union, grammar, tool_grammar)
                    else:
                        grammar = tool_grammar
                assert grammar is not None
                compiler = await asyncio.to_thread(grammar_compiler)
                # to_thread loses XGrammar's grammar-object overload when inferring its arguments.
                compiled_grammar = await asyncio.to_thread(lambda: compiler.compile_grammar(grammar))
            except (TypeError, ValueError) as error:
                param = "tool_choice"
                if response_schema is not None:
                    param = "response_format"
                return _openai_error(str(error), 400, param)
        submitted_at = time.monotonic()
        cancellations: list[threading.Event] = []
        seed = body.seed
        if seed is None:
            seed = secrets.randbits(32)
        try:
            for index, queue in enumerate(event_batches):
                grammar_matcher = None
                if compiled_grammar is not None:
                    grammar_matcher = xgrammar.GrammarMatcher(compiled_grammar)
                    assert grammar_matcher.accept_string(decoders[index].prefix)
                cancellations.append(
                    engine.submit(
                        tuple(prompt_token_ids),
                        max_tokens,
                        generation_config,
                        (seed + index + 2**63) % 2**64 - 2**63,
                        return_logprobs=bool(body.logprobs),
                        grammar_matcher=grammar_matcher,
                        on_events=lambda events, queue=queue: loop.call_soon_threadsafe(queue.put_nowait, events),
                    )
                )
        except Exception as error:
            for cancelled in cancellations:
                cancelled.set()
            if isinstance(error, ValueError):
                return _openai_error(str(error), 400)
            raise
        top_logprobs_count = body.top_logprobs or 0

        async def incoming_events(index: int) -> AsyncIterator[TokenEvent | None]:
            """Yields None once per idle second so the stream can keep the connection alive while queued."""
            while not await request.is_disconnected():
                if engine_errors:
                    raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
                try:
                    for event in await asyncio.wait_for(event_batches[index].get(), 1.0):
                        yield event
                except TimeoutError:
                    yield None

        def logprob_entry(event: GeneratedToken) -> dict[str, Any]:
            assert event.logprobs is not None
            alternatives = [
                _logprob_entry(model.token_codec.decode_token_bytes(token_id), logprob)
                for token_id, logprob in zip(
                    event.logprobs.top_token_ids[:top_logprobs_count],
                    event.logprobs.top_logprobs[:top_logprobs_count],
                    strict=True,
                )
            ]
            # OpenAI uses a sentinel for sampled tokens outside the unfiltered top twenty.
            selected_logprob = -9999.0
            if event.token_id in event.logprobs.top_token_ids:
                selected_logprob = event.logprobs.logprob
            return _logprob_entry(model.token_codec.decode_token_bytes(event.token_id), selected_logprob) | {
                "top_logprobs": alternatives
            }

        def log_request(finished: SequenceFinished | None, first_token_at: float | None) -> None:
            now = time.monotonic()
            seconds_total = now - submitted_at
            logger.info(
                json.dumps(
                    {
                        "request": request_id,
                        "user": body.user,
                        "safety_identifier": body.safety_identifier,
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": None if finished is None else finished.completion_tokens,
                        "finish_reason": None if finished is None else finished.reason,
                        "seconds_to_first_token": None
                        if first_token_at is None
                        else round(first_token_at - submitted_at, 3),
                        "seconds_total": round(seconds_total, 3),
                        "completion_tokens_per_second": (
                            None
                            if finished is None or not seconds_total
                            else round(finished.completion_tokens / seconds_total, 2)
                        ),
                    }
                )
            )

        async def generate(index: int) -> AsyncGenerator[Chunk]:
            decoder = decoders[index]
            pending: list[tuple[GeneratedToken, int, int]] = []
            undecoded: list[GeneratedToken] = []
            sent = ""
            completion_tokens = 0
            first_token_at = None

            def resolve_token_positions(raw_length: int) -> None:
                if not undecoded or len(decoder.raw_response) <= raw_length:
                    return
                first_byte = len(decoder.raw_response[:raw_length].encode())
                spans = model.token_codec.decode_token_spans(token.token_id for token in undecoded)
                offset = len(decoder.raw_response.encode()) - max((end for _, end in spans), default=0)
                pending.extend(
                    (token, max(first_byte, offset + start), offset + end)
                    for token, (start, end) in zip(undecoded, spans, strict=True)
                )
                undecoded.clear()

            try:
                async for event in incoming_events(index):
                    if event is None:
                        yield Chunk("", "", [], None)
                        continue
                    if isinstance(event, GrammarConstraintError):
                        raise event
                    finished = None
                    reasoning = ""
                    visible = ""
                    message = None
                    tool_calls: tuple[dict[str, Any], ...] = ()
                    if isinstance(event, GeneratedToken):
                        completion_tokens += 1
                        first_token_at = first_token_at or time.monotonic()
                        raw_length = len(decoder.raw_response)
                        reasoning, visible = decoder.step(event.token_id)
                        if body.logprobs:
                            undecoded.append(event)
                            resolve_token_positions(raw_length)
                    else:
                        finished = event
                    if (
                        finished is not None
                        or decoder.stop_position is not None
                        or decoder.tool_call_position is not None
                    ):
                        prior_reasoning = decoder.reasoning
                        raw_length = len(decoder.raw_response)
                        message = decoder.finish_output()
                        resolve_token_positions(raw_length)
                        chain_of_thought = message.chain_of_thought or ""
                        if chain_of_thought.startswith(prior_reasoning):
                            reasoning += chain_of_thought[len(prior_reasoning) :]
                        if not message.response.startswith(sent):
                            raise RuntimeError("The parsed response changed after content was streamed.")
                        visible = message.response[len(sent) :]
                        stop_position = decoder.stop_position
                        tool_call_position = decoder.tool_call_position
                        if stop_position is not None and (
                            tool_call_position is None or stop_position < tool_call_position
                        ):
                            finished = SequenceFinished(FinishReason.STOP, completion_tokens)
                        elif tool_call_position is not None:
                            reason = FinishReason.TOOL_CALLS
                            if completion_tokens >= max_tokens:
                                reason = FinishReason.LENGTH
                            finished = SequenceFinished(reason, completion_tokens)
                        assert finished is not None
                        if message.tool_calls and finished.reason is FinishReason.STOP and stop_position is None:
                            finished = SequenceFinished(FinishReason.TOOL_CALLS, finished.completion_tokens)
                        tool_calls = tuple(
                            {
                                "id": f"call_{secrets.token_hex(12)}",
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                            for call in message.tool_calls
                        )
                    # Logprobs describe sampled tokens even when a stop hides part of their bytes.
                    ready = []
                    if body.logprobs and (visible or finished is not None):
                        visible_end = len((sent + visible).encode())
                        confirmed_end = max((end for _, end in decoder.parsed.response_spans), default=0)
                        held = []
                        for token, start, end in pending:
                            position = decoder.parsed.token_position(start, end)
                            if position is not None and position < visible_end:
                                ready.append(token)
                            elif position is not None or end > confirmed_end:
                                held.append((token, start, end))
                        pending = held
                    logprobs = [logprob_entry(event) for event in ready] if body.logprobs else []
                    sent += visible
                    if reasoning or visible or logprobs or finished is not None:
                        yield Chunk(reasoning, visible, logprobs, finished, tool_calls)
                    if finished is not None:
                        log_request(finished, first_token_at)
                        return
                log_request(None, first_token_at)
            finally:
                cancellations[index].set()

        async def choice_chunks() -> AsyncGenerator[tuple[int, Chunk]]:
            queue: asyncio.Queue[tuple[int, Chunk | Exception | None]] = asyncio.Queue()

            async def forward(index: int) -> None:
                try:
                    async with aclosing(generate(index)) as chunks:
                        async for chunk in chunks:
                            await queue.put((index, chunk))
                except Exception as error:  # noqa: BLE001
                    queue.put_nowait((index, error))
                finally:
                    queue.put_nowait((index, None))

            tasks = [asyncio.create_task(forward(index)) for index in range(len(cancellations))]
            try:
                remaining = len(cancellations)
                while remaining:
                    index, chunk = await queue.get()
                    if isinstance(chunk, Exception):
                        raise chunk
                    if chunk is None:
                        remaining -= 1
                    else:
                        yield index, chunk
            finally:
                for cancelled in cancellations:
                    cancelled.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        response_identity: dict[str, Any] = {"id": request_id, "created": int(time.time()), "model": model_name}
        if body.service_tier is not None:
            response_identity["service_tier"] = "default"
        chunk_identity = response_identity | {"object": "chat.completion.chunk"}
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        include_obfuscation = body.stream_options is None or body.stream_options.include_obfuscation is not False
        if include_usage:
            chunk_identity["usage"] = None

        def usage(completion_tokens: int) -> dict[str, int]:
            return {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_token_ids) + completion_tokens,
            }

        async def stream() -> AsyncGenerator[str]:
            completion_tokens = 0
            try:
                yield _sse(
                    chunk_identity
                    | {
                        "choices": [
                            _choice(index=index, delta={"role": "assistant", "content": ""})
                            for index in range(len(cancellations))
                        ]
                    },
                    obfuscate=include_obfuscation,
                )
                async with aclosing(choice_chunks()) as chunks:
                    async for index, (reasoning, content, logprobs, finished, tool_calls) in chunks:
                        if not (reasoning or content or logprobs or finished or tool_calls):
                            yield ": keep-alive\n\n"
                            continue
                        if reasoning or content or logprobs or tool_calls:
                            delta: dict[str, Any] = {"content": content} if content or not reasoning else {}
                            if reasoning:
                                delta["reasoning_content"] = reasoning
                            if tool_calls:
                                delta["tool_calls"] = [
                                    call | {"index": call_index} for call_index, call in enumerate(tool_calls)
                                ]
                            choice = _choice(
                                index=index, delta=delta, logprobs={"content": logprobs} if body.logprobs else None
                            )
                            yield _sse(chunk_identity | {"choices": [choice]}, obfuscate=include_obfuscation)
                        if finished is not None:
                            completion_tokens += finished.completion_tokens
                            finish_reason: str = finished.reason
                            yield _sse(
                                chunk_identity
                                | {
                                    "choices": [
                                        _choice(
                                            index=index,
                                            delta={},
                                            finish_reason=finish_reason,
                                        )
                                    ]
                                },
                                obfuscate=include_obfuscation,
                            )
                if include_usage:
                    yield _sse(
                        chunk_identity | {"choices": [], "usage": usage(completion_tokens)},
                        obfuscate=include_obfuscation,
                    )
                yield "data: [DONE]\n\n"
            except GrammarConstraintError as error:
                yield f"data: {bytes(_openai_error(str(error), 400).body).decode()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as error:  # noqa: BLE001
                traceback.print_exception(error)
                yield f"data: {bytes(_openai_error('Internal server error.', 500).body).decode()}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                for cancelled in cancellations:
                    cancelled.set()

        if body.stream:
            output = stream()

            class CompletionStream(StreamingResponse):
                async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Starlette does not close its iterator when the ASGI client disconnects.
                        for cancelled in cancellations:
                            cancelled.set()
                        await output.aclose()

            return CompletionStream(output, media_type="text/event-stream")
        outputs: list[list[Chunk]] = [[] for _ in cancellations]
        try:
            async for index, chunk in choice_chunks():
                outputs[index].append(chunk)
        except GrammarConstraintError as error:
            return _openai_error(str(error), 400)
        choices = []
        completion_tokens = 0
        for index, chunks in enumerate(outputs):
            finished = chunks[-1].finished if chunks else None
            if finished is None:
                return Response(status_code=499)
            completion_tokens += finished.completion_tokens
            message: dict[str, Any] = {"role": "assistant", "content": "".join(chunk.content for chunk in chunks)}
            if reasoning := "".join(chunk.reasoning for chunk in chunks):
                message["reasoning_content"] = reasoning
            tool_calls = [call for chunk in chunks for call in chunk.tool_calls]
            if tool_calls:
                message["tool_calls"] = tool_calls
                if not message["content"]:
                    message["content"] = None
            finish_reason = finished.reason
            choices.append(
                _choice(
                    index=index,
                    message=message,
                    finish_reason=finish_reason,
                    logprobs={"content": [entry for chunk in chunks for entry in chunk.logprobs]}
                    if body.logprobs
                    else None,
                )
            )
        if body.metadata is not None:
            response_identity["metadata"] = body.metadata
        return JSONResponse(
            content=response_identity
            | {"object": "chat.completion", "choices": choices, "usage": usage(completion_tokens)}
        )

    return api


def start_server(
    model_path: Path,
    model_name: str,
    host: str,
    port: int,
    batching_config: ContinuousBatchingConfig,
    sharding_config: ShardingConfig,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    imported = import_model(str(model_path), sharding_config=sharding_config).model
    if not isinstance(imported, LanguageModel):
        raise TypeError(f"Expected a language model, got {type(imported).__name__}.")
    uvicorn.run(create_app(imported, model_name, batching_config), host=host, port=port)
