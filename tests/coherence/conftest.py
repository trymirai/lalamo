from __future__ import annotations

import pytest
from filelock import FileLock
from xdist import get_xdist_worker_id

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


def _is_large(request: pytest.FixtureRequest) -> bool:
    repo: str = request.node.callspec.params["converted_model_path"]
    return repo in LARGE_MODELS


@pytest.fixture(autouse=True)
def _gpu_slots(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> None:
    worker_id = get_xdist_worker_id(request)
    if worker_id == "master":
        yield
        return

    lock_dir = tmp_path_factory.getbasetemp().parent / "gpu_slots"
    lock_dir.mkdir(exist_ok=True)

    locks = [FileLock(lock_dir / f"slot_{i}.lock") for i in range(NUM_SLOTS)]
    worker_index = int(worker_id[2:])

    if _is_large(request):
        for lock in locks:
            lock.acquire()
    else:
        locks[worker_index].acquire()

    yield

    for lock in locks:
        if lock.is_locked:
            lock.release()
