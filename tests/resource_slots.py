from __future__ import annotations

from collections.abc import Generator

import pytest
from filelock import FileLock
from xdist import get_xdist_worker_id

from lalamo.model_import.model_specs.common import ModelSpec

LARGE_MODELS: frozenset[str] = frozenset({
    "google/gemma-3-27b-it",
    "mlx-community/gemma-3-27b-it-4bit",
    "mlx-community/gemma-3-27b-it-8bit",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-AWQ",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3.5-27B",
    "mlx-community/Qwen3.5-27B-MLX-4bit",
    "mlx-community/Qwen3.5-27B-MLX-8bit",
    "mistral-community/Codestral-22B-v0.1",
    "mistralai/Devstral-Small-2505",
    "RekaAI/reka-flash-3.1",
})

NUM_SLOTS = 4


def _get_repo(request: pytest.FixtureRequest) -> str | None:
    if not hasattr(request.node, "callspec"):
        return None
    for param in request.node.callspec.params.values():
        if isinstance(param, ModelSpec):
            return param.repo
    return None


def resource_slots(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None]:
    worker_id = get_xdist_worker_id(request)
    if worker_id == "master":
        yield
        return

    lock_dir = tmp_path_factory.getbasetemp().parent / "resource_slots"
    lock_dir.mkdir(exist_ok=True)

    locks = [FileLock(lock_dir / f"slot_{i}.lock") for i in range(NUM_SLOTS)]
    worker_index = int(worker_id[2:])

    repo = _get_repo(request)
    if repo is not None and repo in LARGE_MODELS:
        for lock in locks:
            lock.acquire()
    else:
        locks[worker_index % NUM_SLOTS].acquire()

    yield

    for lock in locks:
        if lock.is_locked:
            lock.release()
