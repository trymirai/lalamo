import asyncio
import json
import logging
import secrets
import threading
import time
import traceback
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import AssistantMessage, Message, ReasoningEffort, SystemMessage, UserMessage
from lalamo.utils.sharding import ShardingConfig

logger = logging.getLogger("lalamo.server")


class ChatRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str


class ChatMessageParam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: ChatRole
    content: str | list[TextPart] | None
    name: str | None = None
    reasoning_content: str | None = None

    def to_message(self) -> Message:
        content = self.content or ""
        if isinstance(content, list):
            content = "".join(part.text for part in content)
        match self.role:
            case ChatRole.USER:
                return UserMessage(content)
            case ChatRole.ASSISTANT:
                return AssistantMessage(None, content)
            case ChatRole.SYSTEM | ChatRole.DEVELOPER:
                return SystemMessage(content)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool | None = False
    include_obfuscation: bool | None = False


class ChatTemplateKwargs(BaseModel):
    """llama.cpp's chat-template switches; `enable_thinking` maps to medium or no reasoning effort."""

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
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    min_p: Annotated[float, Field(ge=0, le=1)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0)] | None = Field(
        None, validation_alias=AliasChoices("repetition_penalty", "repeat_penalty")
    )
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None
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
    n: Literal[1] | None = 1
    metadata: dict[str, str] | None = None
    user: str | None = None
    store: bool | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> "ChatCompletionRequest":
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true.")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true.")
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            raise ValueError("Specify only one of max_completion_tokens and max_tokens.")
        return self

    @property
    def stop_strings(self) -> list[str]:
        return [self.stop] if isinstance(self.stop, str) else self.stop or []

    @property
    def effective_reasoning_effort(self) -> ReasoningEffort | None:
        if self.reasoning_effort == "none":
            return ReasoningEffort.NO_REASONING
        if self.reasoning_effort is not None:
            return ReasoningEffort(self.reasoning_effort)
        if self.chat_template_kwargs is None or self.chat_template_kwargs.enable_thinking is None:
            return None
        return ReasoningEffort.MEDIUM if self.chat_template_kwargs.enable_thinking else ReasoningEffort.NO_REASONING


class Chunk(NamedTuple):
    reasoning: str
    content: str
    logprobs: list[dict[str, Any]]
    finished: SequenceFinished | None


def _openai_error(message: str, status: int, param: str | None = None, code: str | None = None) -> JSONResponse:
    error_type = "server_error" if status == 500 else "invalid_request_error"
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": error_type, "param": param, "code": code}}
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _choice(**fields: object) -> dict[str, object]:
    return {"index": 0, "finish_reason": None, "logprobs": None} | fields


def _logprob_entry(text: str, logprob: float) -> dict[str, Any]:
    return {"token": text, "bytes": list(text.encode()), "logprob": logprob}


def _split_visible(
    pending: list[tuple[GeneratedToken, str]], visible_length: int
) -> tuple[list[tuple[GeneratedToken, str]], list[tuple[GeneratedToken, str]]]:
    """Splits pending tokens into those whose text lies entirely within the first `visible_length` characters."""
    ready = []
    rest = []
    offset = 0
    for event, text in pending:
        if offset + len(text) <= visible_length:
            ready.append((event, text))
        else:
            rest.append((event, text[max(0, visible_length - offset) :]))
        offset += len(text)
    return ready, rest


def create_app(model: LanguageModel, model_name: str, config: ContinuousBatchingConfig) -> FastAPI:
    engine = ContinuousBatchingEngine(model, config)
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

    @api.get("/health")
    @api.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
            return _openai_error(error_message, 400, ".".join(map(str, first_error["loc"])) or None)
        if body.model != model_name:
            return _openai_error(f"Model {body.model!r} does not exist.", 404, "model", "model_not_found")

        reasoning_effort = body.effective_reasoning_effort
        try:
            prompt_token_ids = model.token_codec.encode_request(
                [message.to_message() for message in body.messages], reasoning_effort=reasoning_effort
            )
        except ValueError:
            return _openai_error(
                "The requested reasoning_effort is not supported by this model.", 400, "reasoning_effort"
            )
        # Like llama.cpp, the requested output budget is clamped to the room left after the prompt.
        remaining_context = engine.context_limit - len(prompt_token_ids)
        max_tokens = min(body.max_completion_tokens or body.max_tokens or remaining_context, remaining_context)
        if max_tokens < 1:
            return _openai_error("The prompt exceeds the model context length.", 400, "messages")

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
        request_id = f"chatcmpl-{secrets.token_hex(16)}"
        loop = asyncio.get_running_loop()
        event_batches: asyncio.Queue[Sequence[TokenEvent]] = asyncio.Queue()
        submitted_at = time.monotonic()
        engine.submit(
            tuple(prompt_token_ids),
            max_tokens,
            generation_config,
            body.seed if body.seed is not None else secrets.randbits(32),
            return_logprobs=bool(body.logprobs),
            on_events=lambda events: loop.call_soon_threadsafe(event_batches.put_nowait, events),
        )
        top_logprobs_count = body.top_logprobs or 0

        async def incoming_events() -> AsyncIterator[TokenEvent | None]:
            """Yields None once per idle second so the stream can keep the connection alive while queued."""
            while not await request.is_disconnected():
                if engine_errors:
                    raise RuntimeError("Continuous inference engine failed.") from engine_errors[0]
                try:
                    for event in await asyncio.wait_for(event_batches.get(), 1.0):
                        yield event
                except TimeoutError:
                    yield None

        def logprob_entry(event: GeneratedToken, text: str) -> dict[str, Any]:
            assert event.logprobs is not None
            alternatives = [
                _logprob_entry(model.token_codec.decode_tokens([token_id]), logprob)
                for token_id, logprob in zip(
                    event.logprobs.top_token_ids[:top_logprobs_count],
                    event.logprobs.top_logprobs[:top_logprobs_count],
                    strict=True,
                )
            ]
            return _logprob_entry(text, event.logprobs.logprob) | {"top_logprobs": alternatives}

        def log_request(finished: SequenceFinished | None, first_token_at: float | None) -> None:
            now = time.monotonic()
            decode_seconds = None if first_token_at is None else now - first_token_at
            logger.info(
                json.dumps(
                    {
                        "request": request_id,
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": None if finished is None else finished.completion_tokens,
                        "finish_reason": None if finished is None else finished.reason,
                        "seconds_to_first_token": None
                        if first_token_at is None
                        else round(first_token_at - submitted_at, 3),
                        "seconds_total": round(now - submitted_at, 3),
                        "decode_tokens_per_second": (
                            None
                            if finished is None or not decode_seconds
                            else round(finished.completion_tokens / decode_seconds, 2)
                        ),
                    }
                )
            )

        async def generate() -> AsyncIterator[Chunk]:
            decoder = model.token_codec.decode_stream(reasoning_effort)
            pending: list[tuple[GeneratedToken, str]] = []
            text = ""
            sent = ""
            completion_tokens = 0
            first_token_at = None
            async for event in incoming_events():
                if event is None:
                    yield Chunk("", "", [], None)
                    continue
                finished = None
                reasoning = ""
                if isinstance(event, GeneratedToken):
                    completion_tokens += 1
                    first_token_at = first_token_at or time.monotonic()
                    reasoning, piece = decoder.step(event.token_id)
                    if decoder.response_started:
                        pending.append((event, piece))
                        text += piece
                else:
                    finished = event
                    message = decoder.finish()
                    chain_of_thought = message.chain_of_thought or ""
                    if chain_of_thought.startswith(decoder.reasoning):
                        reasoning = chain_of_thought[len(decoder.reasoning) :]
                    if not message.response.startswith(sent):
                        raise RuntimeError("The parsed response changed after content was streamed.")
                    text = message.response[len(sent) :]

                stop_position = min(
                    (position for stop in body.stop_strings if (position := text.find(stop)) >= 0), default=-1
                )
                if stop_position >= 0:
                    finished = SequenceFinished(FinishReason.STOP, completion_tokens)
                    visible = text[:stop_position]
                elif finished is None:
                    # Hold back a suffix that may still grow into a stop string.
                    held_back = max(
                        (
                            length
                            for stop in body.stop_strings
                            for length in range(1, len(stop))
                            if text.endswith(stop[:length])
                        ),
                        default=0,
                    )
                    visible = text[: len(text) - held_back]
                else:
                    visible = text

                ready, pending = _split_visible(pending, len(visible))
                logprobs = [logprob_entry(event, token_text) for event, token_text in ready] if body.logprobs else []
                text = text[len(visible) :]
                sent += visible
                if reasoning or visible or logprobs or finished is not None:
                    yield Chunk(reasoning, visible, logprobs, finished)
                if finished is not None:
                    log_request(finished, first_token_at)
                    return
            log_request(None, first_token_at)

        response_identity: dict[str, Any] = {"id": request_id, "created": int(time.time()), "model": model_name}
        chunk_identity = response_identity | {"object": "chat.completion.chunk"}
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        if include_usage:
            chunk_identity["usage"] = None

        def usage(finished: SequenceFinished) -> dict[str, int]:
            return {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": finished.completion_tokens,
                "total_tokens": len(prompt_token_ids) + finished.completion_tokens,
            }

        async def stream() -> AsyncIterator[str]:
            yield _sse(chunk_identity | {"choices": [_choice(delta={"role": "assistant", "content": ""})]})
            async for reasoning, content, logprobs, finished in generate():
                if not (reasoning or content or logprobs or finished):
                    yield ": keep-alive\n\n"
                    continue
                if reasoning or content or logprobs:
                    delta: dict[str, str] = {"content": content} if content or not reasoning else {}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    choice = _choice(delta=delta, logprobs={"content": logprobs} if body.logprobs else None)
                    yield _sse(chunk_identity | {"choices": [choice]})
                if finished is not None:
                    yield _sse(chunk_identity | {"choices": [_choice(delta={}, finish_reason=finished.reason)]})
                    if include_usage:
                        yield _sse(chunk_identity | {"choices": [], "usage": usage(finished)})
                    yield "data: [DONE]\n\n"

        if body.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        chunks = [chunk async for chunk in generate()]
        finished = chunks[-1].finished if chunks else None
        if finished is None:
            return Response(status_code=499)
        message: dict[str, Any] = {"role": "assistant", "content": "".join(chunk.content for chunk in chunks)}
        if reasoning := "".join(chunk.reasoning for chunk in chunks):
            message["reasoning_content"] = reasoning
        choice = _choice(
            message=message,
            finish_reason=finished.reason,
            logprobs={"content": [entry for chunk in chunks for entry in chunk.logprobs]} if body.logprobs else None,
        )
        return JSONResponse(
            content=response_identity | {"object": "chat.completion", "choices": [choice], "usage": usage(finished)}
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
