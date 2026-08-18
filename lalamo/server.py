import asyncio
import os
import secrets
import threading
import time
import traceback
import uuid
from _thread import LockType
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Self

import jax.numpy as jnp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from lalamo.data.huggingface_message import HFMessage
from lalamo.inference.continuous_batching import (
    CompletedRequest,
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    ContinuousDecodeCompletedEvent,
    ContinuousEngineEvent,
    ContinuousPrefillCompletedEvent,
    DecodeStepLogits,
    TokenizedRequest,
)
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.utils.sharding import ShardingConfig

BatchStatus = Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class LogitsResponseConfig:
    top_k: int = 256

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least one.")


@dataclass(frozen=True)
class RequestBody:
    sequence_id: str
    messages: list[HFMessage]
    model: str
    max_completion_tokens: int = 8192

    generation_config: GenerationConfig | None = None
    dtype: Literal["bfloat16", "float32"] | None = None
    seed: int | None = None
    enable_thinking: bool = True
    logits: LogitsResponseConfig | None = None


@dataclass(frozen=True)
class ResponseBody:
    sequence_id: str
    chain_of_thought: str | None
    response: str
    logits: tuple[DecodeStepLogits, ...] | None = None


@dataclass(frozen=True)
class Batch:
    id: str
    total: int
    completed: int = 0
    results: tuple[ResponseBody, ...] = ()
    status: BatchStatus = "in_progress"
    error: str | None = None

    @classmethod
    def init(cls, total: int) -> Self:
        return cls(id=f"batch_{uuid.uuid4().hex}", total=total)


@dataclass(frozen=True)
class _RequestContext:
    batch_id: str
    enable_thinking: bool


@dataclass(frozen=True)
class WeightedLatencySample:
    sample_count: int
    latency_seconds: float


@dataclass
class RequestBenchmarkMetrics:
    sequence_id: str
    admitted_at_seconds: float = 0.0
    prompt_token_count: int | None = None
    output_token_count: int = 0
    prefill_completed_at_seconds: float | None = None
    prefill_duration_seconds: float | None = None
    first_token_at_seconds: float | None = None
    last_token_at_seconds: float | None = None
    completed_at_seconds: float | None = None


@dataclass(frozen=True)
class BenchmarkMetricsSnapshot:
    batch_id: str
    elapsed_seconds: float
    prompt_token_count: int
    output_token_count: int
    completed_requests: int
    requests: tuple[RequestBenchmarkMetrics, ...]
    inter_token_latency_samples: tuple[WeightedLatencySample, ...] = ()


@dataclass
class BenchmarkMetricsCollector:
    batch_id: str
    started_at_seconds: float
    _requests: dict[str, RequestBenchmarkMetrics]
    _inter_token_latency_samples: list[WeightedLatencySample]
    _lock: LockType = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def init(
        cls,
        batch_id: str,
        requests: tuple[tuple[str, str], ...],
        started_at_seconds: float | None = None,
    ) -> Self:
        collector = cls(
            batch_id=batch_id,
            started_at_seconds=time.perf_counter() if started_at_seconds is None else started_at_seconds,
            _requests={},
            _inter_token_latency_samples=[],
        )
        collector.add_requests(requests, admitted_at_seconds=collector.started_at_seconds)
        return collector

    def add_requests(
        self,
        requests: tuple[tuple[str, str], ...],
        admitted_at_seconds: float | None = None,
    ) -> None:
        admitted_at_seconds = time.perf_counter() if admitted_at_seconds is None else admitted_at_seconds
        elapsed_seconds = admitted_at_seconds - self.started_at_seconds
        with self._lock:
            duplicate_ids = self._requests.keys() & dict(requests).keys()
            if duplicate_ids:
                raise ValueError(f"Benchmark request ids must be unique: {sorted(duplicate_ids)}")
            self._requests.update(
                {
                    request_id: RequestBenchmarkMetrics(
                        sequence_id=sequence_id,
                        admitted_at_seconds=elapsed_seconds,
                    )
                    for request_id, sequence_id in requests
                }
            )

    def record(self, event: ContinuousEngineEvent, completed_at_seconds: float | None = None) -> None:
        completed_at_seconds = time.perf_counter() if completed_at_seconds is None else completed_at_seconds
        elapsed_seconds = completed_at_seconds - self.started_at_seconds
        with self._lock:
            match event:
                case ContinuousPrefillCompletedEvent(request_ids, prompt_token_counts, duration_seconds):
                    for request_id, prompt_token_count in zip(request_ids, prompt_token_counts, strict=True):
                        request_metrics = self._requests.get(request_id)
                        if request_metrics is None or request_metrics.prompt_token_count is not None:
                            continue
                        request_metrics.prompt_token_count = prompt_token_count
                        request_metrics.prefill_completed_at_seconds = (
                            elapsed_seconds - request_metrics.admitted_at_seconds
                        )
                        request_metrics.prefill_duration_seconds = duration_seconds
                case ContinuousDecodeCompletedEvent(request_ids, completed, duration_seconds):
                    matched_request_count = 0
                    first_token_count = 0
                    for request_id, request_completed in zip(request_ids, completed, strict=True):
                        request_metrics = self._requests.get(request_id)
                        if request_metrics is None:
                            continue
                        matched_request_count += 1
                        first_token_count += request_metrics.first_token_at_seconds is None
                        request_metrics.output_token_count += 1
                        if request_metrics.first_token_at_seconds is None:
                            request_metrics.first_token_at_seconds = (
                                elapsed_seconds - request_metrics.admitted_at_seconds
                            )
                        request_metrics.last_token_at_seconds = elapsed_seconds - request_metrics.admitted_at_seconds
                        if request_completed:
                            request_metrics.completed_at_seconds = (
                                elapsed_seconds - request_metrics.admitted_at_seconds
                            )
                    interval_count = matched_request_count - first_token_count
                    if interval_count:
                        self._inter_token_latency_samples.append(
                            WeightedLatencySample(
                                sample_count=interval_count,
                                latency_seconds=duration_seconds,
                            )
                        )

    def snapshot(self, completed_at_seconds: float | None = None) -> BenchmarkMetricsSnapshot:
        completed_at_seconds = time.perf_counter() if completed_at_seconds is None else completed_at_seconds
        with self._lock:
            requests = tuple(replace(request) for request in self._requests.values())
            inter_token_latency_samples = tuple(self._inter_token_latency_samples)
        return BenchmarkMetricsSnapshot(
            batch_id=self.batch_id,
            elapsed_seconds=max(completed_at_seconds - self.started_at_seconds, 0.0),
            prompt_token_count=sum(request.prompt_token_count or 0 for request in requests),
            output_token_count=sum(request.output_token_count for request in requests),
            completed_requests=sum(request.completed_at_seconds is not None for request in requests),
            requests=requests,
            inter_token_latency_samples=inter_token_latency_samples,
        )


def _record_engine_event(event: ContinuousEngineEvent) -> None:
    collector = app.state.metrics_collector
    if collector is not None:
        collector.record(event)


def _run_engine(stop_event: threading.Event) -> None:
    try:
        app.state.engine.run(stop_event)
    except Exception as error:  # noqa: BLE001
        traceback.print_exception(error)
        app.state.engine_error = str(error)


async def _collect_completed_requests() -> None:
    while True:
        while completed := app.state.engine.pop_completed():
            _record_completion(completed)
        if app.state.engine_error is not None:
            for batch_id, batch in app.state.batches.items():
                if batch.status == "in_progress":
                    app.state.batches[batch_id] = replace(batch, status="failed", error=app.state.engine_error)
            return
        await asyncio.sleep(0.005)


def _record_completion(completed: CompletedRequest) -> None:
    context = app.state.request_contexts.pop(completed.request_id)
    batch = app.state.batches[context.batch_id]
    if completed.error is None:
        response = app.state.model.token_codec.decode_response(
            list(completed.output_token_ids),
            expect_thinking=context.enable_thinking,
        )
        result = ResponseBody(
            sequence_id=completed.sequence_id,
            chain_of_thought=response.chain_of_thought,
            response=response.response,
            logits=completed.logits,
        )
        results = (*batch.results, result)
        error = batch.error
    else:
        results = batch.results
        error = completed.error if batch.error is None else f"{batch.error}; {completed.error}"
    completed_count = batch.completed + 1
    if completed_count < batch.total:
        status: BatchStatus = "in_progress"
    elif error is None:
        status = "completed"
    else:
        status = "failed"
    app.state.batches[batch.id] = replace(
        batch,
        completed=completed_count,
        results=results,
        status=status,
        error=error,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if int(os.environ.get("WEB_CONCURRENCY", "1")) > 1:
        raise RuntimeError("This app must run with a single worker.")
    imported = import_model(
        str(app.state.model_path),
        sharding_config=app.state.sharding_config,
        dtype=app.state.dtype,
    ).model
    if not isinstance(imported, LanguageModel):
        raise TypeError(f"Expected a language model, got {type(imported).__name__}.")
    app.state.model = imported
    app.state.batches = {}
    app.state.request_contexts = {}
    app.state.metrics_collector = None
    app.state.engine_error = None
    app.state.engine = ContinuousBatchingEngine(
        imported,
        app.state.batching_config,
        event_callback=_record_engine_event if app.state.enable_benchmark_metrics else None,
    )
    stop_event = threading.Event()
    worker = threading.Thread(target=_run_engine, args=(stop_event,), name="lalamo-continuous", daemon=True)
    worker.start()
    collector = asyncio.create_task(_collect_completed_requests())
    try:
        yield
    finally:
        collector.cancel()
        stop_event.set()
        await asyncio.to_thread(worker.join)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
    traceback.print_exception(exc)
    return JSONResponse(status_code=500, content={"error": "internal server error"})


def _validate_requests(requests: list[RequestBody]) -> None:
    if not requests:
        raise HTTPException(400, "Empty request batch.")
    sequence_ids = [request.sequence_id for request in requests]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise HTTPException(400, "All requests in a batch must specify distinct ids, but found duplicates.")
    expected_model_path = app.state.model_path.resolve()
    for request in requests:
        if Path(request.model).resolve() != expected_model_path:
            raise HTTPException(400, f"This server is loaded with {expected_model_path}, not {request.model}.")
        if (
            request.dtype is not None
            and jnp.dtype(request.dtype) != app.state.model.decoder.embedding.embedding_matrix.dtype
        ):
            raise HTTPException(400, "Request dtype does not match the loaded model dtype.")
        if request.generation_config is not None:
            try:
                request.generation_config.default_policy()
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
        if request.max_completion_tokens < 1:
            raise HTTPException(400, "max_completion_tokens must be at least one.")
        if request.logits is not None and request.logits.top_k >= app.state.model.decoder.vocab_size:
            raise HTTPException(
                400, f"top_k must be smaller than vocabulary size {app.state.model.decoder.vocab_size}."
            )


@app.post("/batches", status_code=202)
async def create_batch(requests: list[RequestBody]) -> Batch:
    if app.state.engine_error is not None:
        raise HTTPException(503, f"Continuous inference engine failed: {app.state.engine_error}")
    _validate_requests(requests)
    started_at_seconds = time.perf_counter()
    batch = Batch.init(len(requests))
    app.state.batches[batch.id] = batch
    request_ids = tuple(f"{batch.id}:{request_index}" for request_index in range(len(requests)))
    if app.state.enable_benchmark_metrics:
        benchmark_requests = tuple(
            (request_id, request.sequence_id) for request_id, request in zip(request_ids, requests, strict=True)
        )
        collector = app.state.metrics_collector
        if collector is None:
            app.state.metrics_collector = BenchmarkMetricsCollector.init(
                batch.id,
                benchmark_requests,
                started_at_seconds=started_at_seconds,
            )
        else:
            collector.add_requests(benchmark_requests, admitted_at_seconds=started_at_seconds)
    for request_id, request in zip(request_ids, requests, strict=True):
        app.state.request_contexts[request_id] = _RequestContext(
            batch_id=batch.id,
            enable_thinking=request.enable_thinking,
        )
        app.state.engine.submit(
            TokenizedRequest(
                request_id=request_id,
                sequence_id=request.sequence_id,
                prompt_token_ids=tuple(
                    app.state.model.token_codec.encode_request(
                        [message.as_message() for message in request.messages],
                        enable_thinking=request.enable_thinking,
                    )
                ),
                max_output_length=request.max_completion_tokens,
                generation_config=request.generation_config or app.state.model.config.generation_config,
                seed=request.seed if request.seed is not None else secrets.randbits(32),
                num_top_logits=None if request.logits is None else request.logits.top_k,
            )
        )
    return batch


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> Batch:
    batch = app.state.batches.get(batch_id)
    if batch is None:
        raise HTTPException(404, "batch not found")
    if batch.status == "in_progress":
        return replace(batch, results=())
    return batch


@app.get("/benchmark-metrics")
async def get_benchmark_metrics() -> BenchmarkMetricsSnapshot:
    if not app.state.enable_benchmark_metrics:
        raise HTTPException(404, "benchmark metrics are disabled")
    collector = app.state.metrics_collector
    if collector is None:
        raise HTTPException(404, "no benchmark batch has started")
    return collector.snapshot()


def start_server(
    model_path: Path,
    host: str,
    port: int,
    batching_config: ContinuousBatchingConfig,
    sharding_config: ShardingConfig,
    *,
    dtype: str | None = None,
    enable_benchmark_metrics: bool = False,
) -> None:
    app.state.model_path = model_path
    app.state.batching_config = batching_config
    app.state.sharding_config = sharding_config
    app.state.dtype = None if dtype is None else jnp.dtype(dtype)
    app.state.enable_benchmark_metrics = enable_benchmark_metrics
    uvicorn.run(app, host=host, port=port)
