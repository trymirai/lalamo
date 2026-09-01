# fmt: off
import asyncio
import codecs
import json
import secrets
import threading
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lalamo.inference.continuous_batching import ContinuousBatchingConfig, ContinuousBatchingEngine, TokenEvent
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import AssistantMessage, ReasoningEffort, SystemMessage, UserMessage
from lalamo.utils.sharding import ShardingConfig


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str
class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart] | None
    name: str | None = None
class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool | None = False
    include_obfuscation: bool | None = False
class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    messages: Annotated[list[Message], Field(min_length=1)]
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    max_completion_tokens: Annotated[int, Field(gt=0)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    min_p: Annotated[float, Field(ge=0, le=1)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0)] | None = None
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    logprobs: bool | None = False
    top_logprobs: Annotated[int, Field(ge=0, le=20)] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: Literal[1] | None = 1
    metadata: dict[str, str] | None = None
    user: str | None = None
    store: bool | None = None
def _openai_error(message: str, status: int, param: str | None = None, code: str | None = None) -> JSONResponse:
    error_type = "server_error" if status == 500 else "invalid_request_error"
    error = {"message": message, "type": error_type, "param": param, "code": code}
    return JSONResponse(status_code=status, content={"error": error})
def create_app(model: LanguageModel, model_name: str, config: ContinuousBatchingConfig) -> FastAPI:
    engine = ContinuousBatchingEngine(model, config)
    context_limit = min((
        config.page_size * config.total_pages,
        *(rope.config.max_sequence_length for rope in model.decoder.transformer.ropes),
    ))
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
            body = ChatCompletionRequest.model_validate(await request.json())
            if body.model != model_name:
                return _openai_error(f"Model {body.model!r} does not exist.", 404, "model", "model_not_found")
            if body.stream_options is not None and not body.stream:
                raise ValueError("stream_options requires stream=true.")  # noqa: TRY301
            if body.top_logprobs is not None and not body.logprobs:
                raise ValueError("top_logprobs requires logprobs=true.")  # noqa: TRY301
            if body.max_completion_tokens is not None and body.max_tokens is not None:
                raise ValueError("Specify only one of max_completion_tokens and max_tokens.")  # noqa: TRY301
            top_logprobs = body.top_logprobs or 0 if body.logprobs else None
            requested_max_tokens = body.max_completion_tokens or body.max_tokens
            stop_strings = [body.stop] if isinstance(body.stop, str) else body.stop or []
            if len(stop_strings) > 4:
                raise ValueError("At most four stop strings are supported.")  # noqa: TRY301
            if any(not stop_string for stop_string in stop_strings):
                raise ValueError("Stop strings must not be empty.")  # noqa: TRY301
            stop_bytes = tuple(stop_string.encode() for stop_string in stop_strings)
            messages = []
            for message in body.messages:
                content = "" if message.content is None else message.content
                if isinstance(content, list):
                    content = "".join(part.text for part in content)
                if message.role in ("system", "developer"):
                    messages.append(SystemMessage(content))
                elif message.role == "user":
                    messages.append(UserMessage(content))
                else:
                    messages.append(AssistantMessage(None, content))
            if body.reasoning_effort == "none":
                effort = ReasoningEffort.NO_REASONING
            else:
                effort = None if body.reasoning_effort is None else ReasoningEffort(body.reasoning_effort)
            prompt_ids = model.token_codec.encode_request(messages, reasoning_effort=effort)
            max_tokens = context_limit - len(prompt_ids) if requested_max_tokens is None else requested_max_tokens
            if max_tokens < 1:
                raise ValueError("The prompt leaves no model context for a completion.")  # noqa: TRY301
            generation_config = model.config.generation_config.override_with(GenerationConfig(
                temperature=body.temperature, top_k=body.top_k, top_p=body.top_p, min_p=body.min_p,
                repetition_penalty=body.repetition_penalty,
                presence_penalty=body.presence_penalty, frequency_penalty=body.frequency_penalty,
            ))
            request_id = f"chatcmpl-{secrets.token_hex(16)}"
            ready = asyncio.Event()
            loop = asyncio.get_running_loop()
            events = engine.submit(
                tuple(prompt_ids), max_tokens, generation_config,
                body.seed if body.seed is not None else secrets.randbits(32),
                return_logprobs=body.logprobs is True,
                wake=lambda: loop.call_soon_threadsafe(ready.set),
            )
        except (ValidationError, ValueError) as error:
            return _openai_error(str(error), 400)
        response_identity: dict[str, Any] = {"id": request_id, "created": int(time.time()), "model": model_name}
        chunk_identity = response_identity | {"object": "chat.completion.chunk"}
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        if include_usage:
            chunk_identity["usage"] = None
        def chat_logprob(event: TokenEvent) -> dict[str, Any]:
            assert event.token_id is not None and event.logprob is not None
            token_bytes = model.token_codec.token_bytes(event.token_id)
            assert top_logprobs is not None
            top = zip(event.top_token_ids[:top_logprobs], event.top_logprobs[:top_logprobs], strict=True)
            alternatives = [
                {"token": value.decode(errors="replace"), "bytes": list(value), "logprob": logprob}
                for token_id, logprob in top for value in (model.token_codec.token_bytes(token_id),)
            ]
            return {"token": token_bytes.decode(errors="replace"), "bytes": list(token_bytes),
                    "logprob": event.logprob, "top_logprobs": alternatives}
        async def generate() -> AsyncIterator[tuple[str, list[dict[str, Any]], TokenEvent | None]]:
            pending_bytes = bytearray()
            pending_events: list[TokenEvent] = []
            pending_ends: list[int] = []
            completion_tokens = 0
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            try:
                while True:
                    while events.empty():
                        if engine_errors:
                            raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
                        if await request.is_disconnected():
                            return
                        with suppress(TimeoutError):
                            await asyncio.wait_for(ready.wait(), 1.0)
                        ready.clear()
                    event = events.get()
                    finish_event = event if event.token_id is None else None
                    if event.token_id is not None:
                        completion_tokens += 1
                        pending_events.append(event)
                        pending_bytes.extend(model.token_codec.token_bytes(event.token_id))
                        pending_ends.append(len(pending_bytes))
                        stop_at = min((position for stop in stop_bytes
                                       if (position := pending_bytes.find(stop)) >= 0), default=-1)
                        if stop_at >= 0:
                            finish_event = TokenEvent(finish_reason="stop", completion_tokens=completion_tokens)
                            visible_end = stop_at
                            engine.cancel(events)
                        else:
                            withheld = max((length for stop in stop_bytes for length in range(1, len(stop))
                                            if pending_bytes.endswith(stop[:length])), default=0)
                            visible_end = len(pending_bytes) - withheld
                    else:
                        visible_end = len(pending_bytes)
                    safe_event_count = 0
                    while safe_event_count < len(pending_ends) and pending_ends[safe_event_count] <= visible_end:
                        safe_event_count += 1
                    if finish_event is None:
                        visible_end = pending_ends[safe_event_count - 1] if safe_event_count else 0
                    piece = decoder.decode(bytes(pending_bytes[:visible_end]), final=finish_event is not None)
                    logprobs = [] if top_logprobs is None else [
                        chat_logprob(event) for event in pending_events[:safe_event_count]
                    ]
                    if finish_event is not None:
                        yield piece, logprobs, finish_event
                        return
                    del pending_bytes[:visible_end], pending_events[:safe_event_count]
                    pending_ends = [end - visible_end for end in pending_ends[safe_event_count:]]
                    if piece or logprobs:
                        yield piece, logprobs, None
            finally:
                engine.cancel(events)
        async def stream() -> AsyncIterator[str]:
            initial_choice = {"index": 0, "delta": {"role": "assistant", "content": ""},
                              "logprobs": None, "finish_reason": None}
            payload: dict[str, Any] = chunk_identity | {"choices": [initial_choice]}
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            async for piece, logprobs, finish_event in generate():
                if piece or logprobs:
                    choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None,
                              "logprobs": None if top_logprobs is None else {"content": logprobs}}
                    yield f"data: {json.dumps(chunk_identity | {'choices': [choice]}, separators=(',', ':'))}\n\n"
                if finish_event is None:
                    continue
                choice = {"index": 0, "delta": {}, "finish_reason": finish_event.finish_reason}
                yield f"data: {json.dumps(chunk_identity | {'choices': [choice]}, separators=(',', ':'))}\n\n"
                if include_usage:
                    usage = {"prompt_tokens": len(prompt_ids),
                             "completion_tokens": finish_event.completion_tokens,
                             "total_tokens": len(prompt_ids) + finish_event.completion_tokens}
                    payload = chunk_identity | {"choices": [], "usage": usage}
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
        if body.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        text_parts: list[str] = []
        token_entries: list[dict[str, Any]] = []
        finish = None
        async for piece, logprobs, finish_event in generate():
            text_parts.append(piece)
            token_entries.extend(logprobs)
            if finish_event is not None:
                finish = finish_event
        if finish is None:
            return Response(status_code=499)
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": finish.completion_tokens,
                 "total_tokens": len(prompt_ids) + finish.completion_tokens}
        choice = {"index": 0, "message": {"role": "assistant", "content": "".join(text_parts)},
                  "finish_reason": finish.finish_reason,
                  "logprobs": None if top_logprobs is None else {"content": token_entries}}
        return JSONResponse(content=response_identity | {
            "object": "chat.completion", "choices": [choice], "usage": usage})
    return api
def start_server(model_path: Path, model_name: str, host: str, port: int, batching_config: ContinuousBatchingConfig,
                 sharding_config: ShardingConfig) -> None:
    imported = import_model(str(model_path), sharding_config=sharding_config).model
    if not isinstance(imported, LanguageModel):
        raise TypeError(f"Expected a language model, got {type(imported).__name__}.")
    uvicorn.run(create_app(imported, model_name, batching_config), host=host, port=port)
