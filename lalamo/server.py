import asyncio
import json
import os
import random
import shutil
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

from lalamo import import_model
from lalamo.data.huggingface_message import HFMessage
from lalamo.models import GenerationConfig
from lalamo.models.common import InferenceConfig

BatchStatus = Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class RequestBody:
    sequence_id: str
    messages: list[HFMessage]
    model: str
    max_completion_tokens: int = 8192

    generation_config: GenerationConfig | None = None
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    seed: int | None = None

    def has_distinct_id(self, other: Self) -> bool:
        return self.sequence_id != other.sequence_id

    def shares_batch_params(self, other: Self) -> bool:
        return (
            self.model == other.model
            and self.max_completion_tokens == other.max_completion_tokens
            and self.generation_config == other.generation_config
            and self.precision == other.precision
            and (self.seed is None) == (other.seed is None)
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
        while cls.from_id(batch_id := f"batch_{uuid.uuid4().hex[:6]}") is not None:
            pass
        return cls(id=batch_id, total=total)

    @classmethod
    def from_id(cls, batch_id: str) -> Self | None:
        path = app.state.cache_dir / f"{batch_id}.json"
        if not path.exists():
            return None
        return cls._converter.structure(json.loads(path.read_text()), cls)

    def save(self) -> None:
        (app.state.cache_dir / f"{self.id}.json").write_text(json.dumps(self._converter.unstructure(self)))


gpu_lock = asyncio.Lock()
creation_lock = asyncio.Lock()


def cleanup_cache(cache_dir: Path) -> None:
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if workers > 1:
        raise RuntimeError("This app must run with a single worker.")
    cleanup_cache(app.state.cache_dir)
    app.state.tasks = set()
    yield


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
        if not (reference.has_distinct_id(request) and reference.shares_batch_params(request)):
            raise HTTPException(
                400,
                "All requests in a batch must specify distinct ids, but identical model, sampling params and "
                f"token limits, got incompatible {reference} and {request}.",
            )
    return requests


def generate_replies(requests: list[RequestBody]) -> Iterator[ResponseBody]:
    reference, *_ = requests

    model, _metadata = import_model(reference.model, precision=reference.precision)

    dataset = [[hf_message.as_message() for hf_message in request.messages] for request in requests]

    if reference.seed is not None:
        base_key = jax.random.key(0)
        keys = jnp.stack([jax.random.fold_in(base_key, jnp.uint32(request.seed)) for request in requests])
    else:
        base_key = jax.random.key(random.getrandbits(32))
        keys = jax.random.split(base_key, len(requests))

    sequence_ids = [request.sequence_id for request in requests]

    for reply_idx, reply in model.reply_many(  # type: ignore[possibly-missing-attribute]
        dataset,
        generation_config=reference.generation_config,
        inference_config=InferenceConfig(reference.max_completion_tokens),
        keys=keys,
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


@app.post("/batches", status_code=202)
async def create_batch(
    requests: Annotated[list[RequestBody], Depends(validate_requests)],
) -> Batch:
    async with creation_lock:
        batch = Batch.init(total=len(requests))
        batch.save()
    task = asyncio.create_task(execute_batch(batch, requests))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return batch


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> Batch:
    if (batch := Batch.from_id(batch_id)) is not None:
        return batch
    raise HTTPException(404, "batch not found")


def start_server(host: str, port: int, vram_bytes: int, cache_dir: Path) -> None:
    app.state.vram_bytes = vram_bytes
    app.state.cache_dir = cache_dir
    uvicorn.run(app, host=host, port=port)
