import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

StartServer = Callable[[], str]

MODELS = ["google/gemma-3-1b-it"]

CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."
MATH_PROMPT = "What's 2 + 2? No thinking, answer right away."

MAX_OUTPUT_LENGTH = 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(base_url: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"lalamo server exited early with code {proc.returncode}")
        try:
            response = httpx.get(f"{base_url}/", timeout=2.0)
        except httpx.HTTPError:
            time.sleep(0.5)
            continue
        if response.status_code in {200, 404}:
            return
        time.sleep(0.5)
    raise TimeoutError("lalamo server did not become ready within 10s")


@pytest.fixture
def serve_process(tmp_path: Path) -> Iterator[StartServer]:
    procs: list[subprocess.Popen[bytes]] = []

    def _start() -> str:
        port = _free_port()
        proc = subprocess.Popen(
            [
                "lalamo",
                "server",
                "--port",
                str(port),
                "--vram-gb",
                "8",
                "--cache-dir",
                str(tmp_path / "batches"),
            ],
        )
        procs.append(proc)
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url, proc)
        return base_url

    yield _start

    for proc in procs:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _make_request(sequence_id: str, model_repo: str, prompt: str, *, seed: int | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "sequence_id": sequence_id,
        "model": model_repo,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_OUTPUT_LENGTH,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _wait_for_batch(base_url: str, batch_id: str, timeout: float = 600.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = httpx.get(f"{base_url}/batches/{batch_id}", timeout=10.0)
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] != "in_progress":
            return snapshot
        time.sleep(0.5)
    raise TimeoutError(f"Batch {batch_id} did not reach a terminal status within {timeout}s")


@pytest.mark.parametrize("model_repo", MODELS)
def test_server_batches(serve_process: StartServer, model_repo: str) -> None:
    base_url = serve_process()

    payload = [
        _make_request("q1", model_repo, CAPITAL_PROMPT),
        _make_request("q2", model_repo, APPLES_PROMPT),
        _make_request("q3", model_repo, MATH_PROMPT),
    ]
    create = httpx.post(f"{base_url}/batches", json=payload, timeout=10.0)
    assert create.status_code == 202
    created = create.json()
    assert created["status"] == "in_progress"
    assert created["total"] == 3
    assert created["completed"] == 0
    assert created["results"] == []
    assert created["error"] is None

    final = _wait_for_batch(base_url, str(created["id"]))
    assert final["status"] == "completed"
    assert final["total"] == 3
    assert final["completed"] == 3
    assert len(final["results"]) == 3

    replies_by_id = {reply["sequence_id"]: reply["response"] for reply in final["results"]}
    assert "london" in replies_by_id["q1"].lower(), f"Expected 'london' in {replies_by_id['q1']!r}"
    assert "yes" in replies_by_id["q2"].lower(), f"Expected 'yes' in {replies_by_id['q2']!r}"
    assert "4" in replies_by_id["q3"], f"Expected '4' in {replies_by_id['q3']!r}"

    refetch = httpx.get(f"{base_url}/batches/{created['id']}", timeout=10.0).json()
    assert refetch == final


@pytest.mark.parametrize("model_repo", MODELS)
def test_server_batches_unknown_id(serve_process: StartServer, model_repo: str) -> None:  # noqa: ARG001
    base_url = serve_process()
    response = httpx.get(f"{base_url}/batches/batch_does_not_exist", timeout=10.0)
    assert response.status_code == 404
