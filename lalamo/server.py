import asyncio
import json
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
from lalamo.inference.batch_scheduler import BatchSchedulerConfig, ContinuousBatchScheduler
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig

BatchStatus = Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class RequestBody:
    sequence_id: str
    messages: list[HFMessage]
    model: str
    max_completion_tokens: int = 8192

    generation_config: GenerationConfig | None = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    seed: int | None = None
    enable_thinking: bool = True

    def shares_batch_params(self, other: Self) -> bool:
        return (
            self.model == other.model
            and self.max_completion_tokens == other.max_completion_tokens
            and self.generation_config == other.generation_config
            and self.dtype == other.dtype
            and (self.seed is None) == (other.seed is None)
            and self.enable_thinking == other.enable_thinking
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
    for request in rest:
        if not reference.shares_batch_params(request):
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

    model = import_model(
        reference.model,
        sharding_config=ShardingConfig.replicated(),
        dtype=jnp.dtype(reference.dtype),
    ).model
    if not isinstance(model, LanguageModel):
        raise RuntimeError(f"Expected a language model, got {type(model).__name__}")  # noqa: TRY004

    dataset = [[hf_message.as_message() for hf_message in request.messages] for request in requests]

    if reference.seed is not None:
        batch_key = jax.random.key(0)
        keys = jnp.stack([jax.random.fold_in(batch_key, jnp.uint32(request.seed)) for request in requests])
    else:
        batch_key, split_key = jax.random.split(jax.random.key(random.getrandbits(32)))
        keys = jax.random.split(split_key, len(requests))
    keychain = Keychain(vmapped_keys=keys, batch_key=batch_key, sharding_config=model.sharding_config)

    sequence_ids = [request.sequence_id for request in requests]
    batch_scheduler = ContinuousBatchScheduler(model=model)

    for reply_idx, reply in batch_scheduler.reply_many(
        dataset,
        generation_config=reference.generation_config,
        batch_scheduler_config=BatchSchedulerConfig(
            max_output_length=reference.max_completion_tokens,
            batch_size=None,
        ),
        enable_thinking=reference.enable_thinking,
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
            replace(batch, completed=len(collected)).save()

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
        active_batches.add(batch.id)
        task = asyncio.create_task(execute_batch(batch, requests))
        app.state.tasks.add(task)
        task.add_done_callback(lambda completed_task: finish_batch_task(completed_task, batch.id))
    return batch


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> Batch:
    if (batch := Batch.from_id(batch_id)) is not None:
        if batch.status == "in_progress":
            return replace(batch, results=())
        return batch
    raise HTTPException(404, "batch not found")


def start_server(host: str, port: int, vram_bytes: int, cache_dir: Path) -> None:
    app.state.vram_bytes = vram_bytes
    app.state.cache_dir = cache_dir
    uvicorn.run(app, host=host, port=port)
