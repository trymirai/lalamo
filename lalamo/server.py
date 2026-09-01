import asyncio
import gc
import json
import logging
import os
import random
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

import cattrs
import jax
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from jax import numpy as jnp

from lalamo.data.huggingface_message import HFMessage
from lalamo.inference.batch_scheduler import _PROBE_CACHE, BatchSchedulerConfig, ContinuousBatchScheduler
from lalamo.model_import.common import import_model
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_registry import ModelRegistry
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ReasoningEffort
from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig, device_put_from_cpu

BatchStatus = Literal["in_progress", "completed", "failed"]

progress_logger = logging.getLogger("uvicorn.error.lalamo")
progress_logger.disabled = True


@dataclass(frozen=True)
class RequestBody:
    sequence_id: str
    messages: list[HFMessage]
    model: str
    max_completion_tokens: int = 8192

    generation_config: GenerationConfig | None = None
    dtype: Literal["bfloat16", "float32"] | None = None
    seed: int | None = None
    reasoning_effort: ReasoningEffort | None = None

    def shares_batch_params(self, other: Self, default_reasoning_effort: ReasoningEffort | None) -> bool:
        self_reasoning_effort = self.reasoning_effort or default_reasoning_effort
        other_reasoning_effort = other.reasoning_effort or default_reasoning_effort

        return (
            self.model == other.model
            and self.max_completion_tokens == other.max_completion_tokens
            and self.generation_config == other.generation_config
            and self.dtype == other.dtype
            and (self.seed is None) == (other.seed is None)
            and self_reasoning_effort is other_reasoning_effort
        )


@dataclass(frozen=True)
class ResponseBody:
    sequence_id: str
    chain_of_thought: str | None
    response: str


@dataclass(frozen=True)
class Batch:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    id: str
    total: int
    completed: int = 0
    results: tuple[ResponseBody, ...] = ()
    status: BatchStatus = "in_progress"
    error: str | None = None

    @classmethod
    def init(cls, total: int) -> Self:
        while True:
            batch_id = f"batch_{uuid.uuid4().hex[:6]}"
            if cls.from_id(batch_id) is None:
                return cls(id=batch_id, total=total)

    @classmethod
    def from_id(cls, batch_id: str) -> Self | None:
        path = app.state.cache_dir / f"{Path(batch_id).name}.json"
        if not path.exists():
            return None
        return cls._converter.structure(json.loads(path.read_text()), cls)

    def save(self) -> None:
        path = app.state.cache_dir / f"{self.id}.json"
        tmp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(self._converter.unstructure(self)))
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


gpu_lock = asyncio.Lock()
creation_lock = asyncio.Lock()


# Resident model across /batches requests: pay the safetensors reload + jit warmup once, not per request.
_resident_model: tuple[tuple[str, str | None], LanguageModel] | None = None


def _load_resident_model(model_path: str, dtype: str | None) -> LanguageModel:
    global _resident_model  # noqa: PLW0603

    cache_key = (model_path, dtype)
    if _resident_model is not None:
        cached_key, cached_model = _resident_model
        if cached_key == cache_key:
            progress_logger.info("model cache hit path=%s dtype=%s", model_path, dtype)
            return cached_model

        # Free the old model's device buffers before importing the new one (avoid two full models in VRAM).
        _resident_model = None
        del cached_model
        _PROBE_CACHE.clear()  # its entries are id(model)-keyed and must not outlive this model
        gc.collect()

    started_at = time.perf_counter()
    progress_logger.info("model load start path=%s dtype=%s", model_path, dtype)
    model = import_model(
        model_path,
        sharding_config=app.state.sharding_config,
        dtype=jnp.dtype(dtype) if dtype is not None else None,
    ).model
    if not isinstance(model, LanguageModel):
        raise TypeError(f"Expected a language model, got {type(model).__name__}")

    _resident_model = (cache_key, model)
    progress_logger.info(
        "model load complete path=%s dtype=%s elapsed_seconds=%.1f",
        model_path,
        dtype,
        time.perf_counter() - started_at,
    )
    return model


def active_batch_ids() -> set[str]:
    if not hasattr(app.state, "active_batch_ids"):
        app.state.active_batch_ids = set()
    return app.state.active_batch_ids


async def sweep_cache() -> None:
    while True:
        cutoff = time.time() - 96 * 3600
        for path in app.state.cache_dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if workers > 1:
        raise RuntimeError("This app must run with a single worker.")
    app.state.cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.tasks = set()
    app.state.active_batch_ids = set()
    sweeper = asyncio.create_task(sweep_cache())
    yield
    sweeper.cancel()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
    traceback.print_exception(exc)
    return JSONResponse(status_code=500, content={"error": "internal server error"})


def validate_requests(
    requests: list[RequestBody],
) -> list[RequestBody]:
    if not requests:
        raise HTTPException(400, "Empty request batch.")

    reference, *rest = requests
    model_spec = ModelRegistry.build().repo_to_model.get(reference.model)
    default_reasoning_effort = None
    if isinstance(model_spec, LanguageModelSpec) and model_spec.reasoning_config is not None:
        default_reasoning_effort = model_spec.reasoning_config.default_reasoning_effort
    for request in rest:
        if not reference.shares_batch_params(request, default_reasoning_effort):
            raise HTTPException(
                400,
                "All requests in a batch must specify identical model, sampling params and "
                f"token limits, got incompatible {reference} and {request}.",
            )

    sequence_ids = [request.sequence_id for request in requests]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise HTTPException(400, "All requests in a batch must specify distinct ids, but found duplicates.")

    return requests


def generate_replies(requests: list[RequestBody]) -> Iterator[ResponseBody]:
    reference, *_ = requests

    model = _load_resident_model(reference.model, reference.dtype)

    dataset = [[hf_message.as_message() for hf_message in request.messages] for request in requests]

    if reference.seed is not None:
        batch_key = jax.random.key(0)
        keys = jnp.stack([jax.random.fold_in(batch_key, jnp.uint32(request.seed)) for request in requests])
    else:
        batch_key, split_key = jax.random.split(jax.random.key(random.getrandbits(32)))
        keys = jax.random.split(split_key, len(requests))
    keys = device_put_from_cpu(keys, model.sharding_config.make_sharding((None,)))
    batch_key = device_put_from_cpu(batch_key, model.sharding_config.make_sharding(()))
    keychain = Keychain(vmapped_keys=keys, batch_key=batch_key, sharding_config=model.sharding_config)

    sequence_ids = [request.sequence_id for request in requests]
    batch_scheduler = ContinuousBatchScheduler(model=model)

    for reply_idx, reply in batch_scheduler.reply_many(
        dataset,
        generation_config=reference.generation_config,
        batch_scheduler_config=BatchSchedulerConfig(
            max_output_length=reference.max_completion_tokens,
            batch_size=app.state.batch_size,
        ),
        reasoning_effort=reference.reasoning_effort,
        keychain=keychain,
        vram_bytes=app.state.vram_bytes,
    ):
        yield ResponseBody(
            sequence_id=sequence_ids[reply_idx],
            chain_of_thought=reply.chain_of_thought,
            response=reply.response,
        )


async def execute_batch(batch: Batch, requests: list[RequestBody]) -> None:
    collected: list[ResponseBody] = []

    def run_generate_replies_with_stats() -> None:
        for response in generate_replies(requests):
            collected.append(response)
            replace(batch, completed=len(collected), results=tuple(collected)).save()
            progress_logger.info(
                "batch progress id=%s completed=%d/%d sequence_id=%s",
                batch.id,
                len(collected),
                batch.total,
                response.sequence_id,
            )

    try:
        async with gpu_lock:
            await asyncio.to_thread(run_generate_replies_with_stats)
        batch = replace(batch, results=tuple(collected), completed=len(collected), status="completed")
    except Exception as exc:  # noqa: BLE001
        batch = replace(batch, results=tuple(collected), completed=len(collected), status="failed", error=str(exc))
        traceback.print_exception(exc)
    finally:
        if batch.status == "in_progress":
            batch = replace(
                batch, results=tuple(collected), completed=len(collected), status="failed", error="interrupted"
            )
        batch.save()
        progress_logger.info(
            "batch finished id=%s status=%s completed=%d/%d error=%s",
            batch.id,
            batch.status,
            batch.completed,
            batch.total,
            batch.error,
        )


def finish_batch_task(task: asyncio.Task, batch_id: str) -> None:
    active_batch_ids().discard(batch_id)
    app.state.tasks.discard(task)


@app.post("/batches", status_code=202)
async def create_batch(
    requests: Annotated[list[RequestBody], Depends(validate_requests)],
) -> Batch:
    async with creation_lock:
        active_batches = active_batch_ids()
        if active_batches:
            batch_id = sorted(active_batches)[0]
            raise HTTPException(409, f"{batch_id} is in progress; starting new batches is not allowed.")

        batch = Batch.init(total=len(requests))
        batch.save()
        progress_logger.info("batch accepted id=%s total=%d", batch.id, batch.total)
        active_batches.add(batch.id)
        task = asyncio.create_task(execute_batch(batch, requests))
        app.state.tasks.add(task)
        task.add_done_callback(lambda completed_task: finish_batch_task(completed_task, batch.id))
    return batch


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> Batch:
    if (batch := Batch.from_id(batch_id)) is not None:
        return batch
    raise HTTPException(404, "batch not found")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def start_server(
    host: str,
    port: int,
    vram_bytes: int | None,
    batch_size: int | None,
    cache_dir: Path,
    sharding_config: ShardingConfig,
    log_progress: bool = False,
) -> None:
    app.state.vram_bytes = vram_bytes
    app.state.batch_size = batch_size
    app.state.cache_dir = cache_dir
    app.state.sharding_config = sharding_config
    progress_logger.disabled = not log_progress
    uvicorn.run(app, host=host, port=port, access_log=not log_progress)
