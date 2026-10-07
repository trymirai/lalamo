import asyncio
import json
import secrets
import threading
import time
import traceback
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, Json, ValidationError
from tokenizers.decoders import DecodeStream

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    SequenceFinished,
    TokenEvent,
)
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import (
    AssistantMessage,
    FunctionCall,
    Message,
    ReasoningConfig,
    ReasoningEffort,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from lalamo.utils.json import JSON

__all__ = ["create_app"]

type OpenAIName = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
type Penalty = Annotated[float, Field(ge=-2, le=2)]


class ChatRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCallParam(BaseModel):
    name: OpenAIName
    arguments: Json[dict[str, JSON]]


class ToolCallParam(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCallParam


class FunctionToolParam(BaseModel):
    name: OpenAIName
    description: str | None = None
    parameters: dict[str, JSON] | None = None
    # Accepted for compatibility; generation is not constrained to the schema.
    strict: bool | None = None


class ToolParam(BaseModel):
    type: Literal["function"]
    function: FunctionToolParam


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessageParam(BaseModel):
    role: ChatRole
    content: str | list[TextPart] | None = None
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallParam] | None = None
    tool_call_id: str | None = None

    def to_message(self) -> Message:
        content = self.content if isinstance(self.content, str) else "".join(part.text for part in self.content or ())
        match self.role:
            case ChatRole.SYSTEM | ChatRole.DEVELOPER:
                return SystemMessage(content)
            case ChatRole.USER:
                return UserMessage(content)
            case ChatRole.ASSISTANT:
                tool_calls: tuple[ToolCall, ...] = tuple(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": FunctionCall(name=call.function.name, arguments=call.function.arguments),
                    }
                    for call in self.tool_calls or ()
                )
                return AssistantMessage(self.reasoning_content, content, tool_calls)
            case ChatRole.TOOL:
                return ToolMessage(content, self.name, self.tool_call_id)


class ChatTemplateKwargs(BaseModel):
    enable_thinking: bool | None = None


class ChatCompletionRequest(BaseModel):
    # Unsupported OpenAI features are rejected instead of silently ignored.
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: Annotated[list[ChatMessageParam], Field(min_length=1)]
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    max_completion_tokens: Annotated[int, Field(gt=0)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    min_p: Annotated[float, Field(ge=0, le=1)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0)] | None = None
    presence_penalty: Penalty | None = None
    frequency_penalty: Penalty | None = None
    seed: int | None = None
    stop: Annotated[str, Field(min_length=1)] | list[Annotated[str, Field(min_length=1)]] | None = None
    stream: Literal[False] | None = None
    tools: list[ToolParam] | None = None
    tool_choice: Literal["auto", "none"] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None
    logprobs: bool | None = None
    top_logprobs: Annotated[int, Field(ge=0, le=20)] | None = None
    n: Literal[1] | None = None
    user: str | None = None

    def effective_reasoning_effort(self, config: ReasoningConfig | None) -> ReasoningEffort | None:
        if self.reasoning_effort == "none":
            return ReasoningEffort.NO_REASONING
        if self.reasoning_effort is not None:
            return ReasoningEffort(self.reasoning_effort)
        enable_thinking = None if self.chat_template_kwargs is None else self.chat_template_kwargs.enable_thinking
        if enable_thinking is None or config is None:
            return None
        if not enable_thinking:
            return ReasoningEffort.NO_REASONING
        if config.default_reasoning_effort is ReasoningEffort.NO_REASONING:
            return ReasoningEffort.MEDIUM
        return None


def _error(message: str, status: int, param: str | None = None, code: str | None = None) -> JSONResponse:
    error_type = "server_error" if status >= 500 else "invalid_request_error"
    return JSONResponse({"error": {"message": message, "type": error_type, "param": param, "code": code}}, status)


def create_app(model: LanguageModel, model_name: str, config: ContinuousBatchingConfig) -> FastAPI:
    engine = ContinuousBatchingEngine(model, config)
    codec = model.token_codec
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
        return _error("Internal server error.", 500)

    model_object = {"id": model_name, "object": "model", "created": 0, "owned_by": "lalamo"}

    @api.get("/health")
    @api.get("/v1/health")
    async def health() -> Response:
        if engine_errors:
            return _error("Continuous inference engine failed.", 503)
        return JSONResponse({"status": "ok"})

    @api.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {"object": "list", "data": [model_object]}

    @api.get("/v1/models/{requested_model:path}")
    async def retrieve_model(requested_model: str) -> Response:
        if requested_model == model_name:
            return JSONResponse(model_object)
        return _error(f"Model {requested_model!r} does not exist.", 404, "model", "model_not_found")

    @api.post("/v1/chat/completions")
    async def complete(request: Request) -> Response:
        if engine_errors:
            raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
        try:
            body = ChatCompletionRequest.model_validate_json(await request.body())
        except ValidationError as error:
            first_error, *_ = error.errors()
            return _error(first_error["msg"], 400, ".".join(map(str, first_error["loc"])) or None)
        if body.model != model_name:
            return _error(f"Model {body.model!r} does not exist.", 404, "model", "model_not_found")

        tools = None
        if body.tools and body.tool_choice != "none":
            if codec.config.response_parser is None or codec.config.response_parser.tool_call_tags is None:
                return _error("This model does not support tool calling.", 400, "tools")
            tools = [tool.model_dump(exclude_none=True) for tool in body.tools]
        try:
            prompt = codec.render_request(
                [message.to_message() for message in body.messages],
                tools=tools,
                reasoning_effort=body.effective_reasoning_effort(codec.config.reasoning_config),
            )
        except (TypeError, ValueError) as error:
            return _error(str(error), 400, "messages")
        prompt_token_ids = codec.encode_text(prompt)
        # Like llama.cpp, the requested output budget is clamped to the room left after the prompt.
        remaining_context = engine.context_limit - len(prompt_token_ids)
        max_tokens = min(body.max_completion_tokens or body.max_tokens or remaining_context, remaining_context)
        if max_tokens < 1:
            return _error(
                f"This model's maximum context length is {engine.context_limit} tokens. "
                f"Your messages resulted in {len(prompt_token_ids)} tokens.",
                400,
                "messages",
                "context_length_exceeded",
            )

        generation_config = model.config.generation_config.override_with(
            GenerationConfig(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                min_p=body.min_p,
                repetition_penalty=body.repetition_penalty,
                presence_penalty=body.presence_penalty,
                frequency_penalty=body.frequency_penalty,
            )
        )
        stop_strings = [body.stop] if isinstance(body.stop, str) else body.stop or []
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Sequence[TokenEvent]] = asyncio.Queue()
        cancelled = engine.submit(
            tuple(prompt_token_ids),
            max_tokens,
            generation_config,
            secrets.randbits(32) if body.seed is None else body.seed,
            return_logprobs=bool(body.logprobs),
            on_events=lambda batch: loop.call_soon_threadsafe(events.put_nowait, batch),
        )

        def logprob(token_id: int, value: float) -> dict[str, Any]:
            token_bytes = codec.decode_token_bytes(token_id)
            return {"token": token_bytes.decode(errors="replace"), "bytes": list(token_bytes), "logprob": value}

        generated_token_ids: list[int] = []
        content_logprobs: list[dict[str, Any]] = []
        stop_decoder = DecodeStream() if stop_strings else None
        decoded_text = ""
        completion_tokens = 0
        finish_reason = None
        try:
            while not await request.is_disconnected():
                if engine_errors:
                    raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
                try:
                    batch = await asyncio.wait_for(events.get(), 1.0)
                except TimeoutError:
                    continue
                for event in batch:
                    if isinstance(event, SequenceFinished):
                        finish_reason, completion_tokens = event
                        break
                    generated_token_ids.append(event.token_id)
                    completion_tokens += 1
                    if event.logprobs is not None:
                        count = body.top_logprobs or 0
                        top_logprobs = [
                            logprob(token_id, value)
                            for token_id, value in zip(
                                event.logprobs.top_token_ids[:count], event.logprobs.top_logprobs[:count], strict=True
                            )
                        ]
                        content_logprobs.append(
                            logprob(event.token_id, event.logprobs.logprob) | {"top_logprobs": top_logprobs}
                        )
                    if stop_decoder is not None:
                        decoded_text += stop_decoder.step(codec.tokenizer, event.token_id) or ""
                        if any(stop in decoded_text for stop in stop_strings):
                            finish_reason = FinishReason.STOP
                            break
                if finish_reason is not None:
                    break
        finally:
            cancelled.set()

        if finish_reason is None:
            return Response(status_code=499)

        text = codec.decode_tokens(generated_token_ids)
        stop_positions = [position for stop in stop_strings if (position := text.find(stop)) >= 0]
        if stop_positions:
            text = text[: min(stop_positions)]
            finish_reason = FinishReason.STOP
        decoded_message = codec.parse_response(text, prompt=prompt, tools=tools or ())
        tool_calls = decoded_message.tool_calls
        if body.parallel_tool_calls is False:
            tool_calls = tool_calls[:1]
        if tool_calls and finish_reason is FinishReason.STOP and not stop_positions:
            finish_reason = FinishReason.TOOL_CALLS
        openai_tool_calls = [
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False),
                },
            }
            for call in tool_calls
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": decoded_message.response or (None if openai_tool_calls else ""),
        }
        if decoded_message.chain_of_thought:
            message["reasoning_content"] = decoded_message.chain_of_thought
        if openai_tool_calls:
            message["tool_calls"] = openai_tool_calls
        return JSONResponse(
            {
                "id": f"chatcmpl-{secrets.token_hex(16)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                        "logprobs": {"content": content_logprobs} if body.logprobs else None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_token_ids) + completion_tokens,
                },
            }
        )

    return api
