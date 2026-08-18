import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Self

import cattrs
import requests
from cattrs.gen import make_dict_structure_fn, override

OPENHERMES_DATASET = "teknium/OpenHermes-2.5"
OPENHERMES_REVISION = "b82037821055c377bed0d495e72e46de3bc72e84"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_PAGE_SIZE = 100
CORPUS_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 8
DEFAULT_SEED = 1337
DEFAULT_REQUEST_COUNT = 1024
DEFAULT_MAX_COMPLETION_TOKENS = 80_000
DEFAULT_DURATION_SECONDS = 60.0
DEFAULT_PORT = 8293
PERCENTILES = (50.0, 90.0, 95.0, 99.0)


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OpenHermesSpeaker(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    GPT = "gpt"


@dataclass(frozen=True)
class OpenHermesTurn:
    speaker: OpenHermesSpeaker
    value: str


@dataclass(frozen=True)
class OpenHermesRow:
    conversations: tuple[OpenHermesTurn, ...]


@dataclass(frozen=True)
class DatasetViewerRow:
    row_idx: int
    row: OpenHermesRow
    truncated_cells: tuple[str, ...]


@dataclass(frozen=True)
class DatasetRowsResponse:
    rows: tuple[DatasetViewerRow, ...]
    num_rows_total: int
    partial: bool


@dataclass(frozen=True)
class BenchmarkMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class CorpusRequest:
    sequence_id: str
    dataset_row_index: int
    messages: tuple[BenchmarkMessage, ...]


@dataclass(frozen=True)
class BenchmarkCorpus:
    schema_version: int
    dataset: str
    revision: str
    config: str
    split: str
    seed: int
    total_dataset_rows: int
    requests: tuple[CorpusRequest, ...]


@dataclass(frozen=True)
class GreedyGenerationConfig:
    temperature: float = 0.0


@dataclass(frozen=True)
class ServerRequest:
    sequence_id: str
    messages: tuple[BenchmarkMessage, ...]
    model: str
    max_completion_tokens: int
    generation_config: GreedyGenerationConfig
    enable_thinking: bool


@dataclass(frozen=True)
class BatchStatus:
    id: str
    total: int
    completed: int
    status: str
    error: str | None = None


@dataclass(frozen=True)
class WeightedLatencySample:
    sample_count: int
    latency_seconds: float


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PercentileValue:
    percentile: float
    value: float


@dataclass(frozen=True)
class MetricDistribution:
    count: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    percentiles: tuple[PercentileValue, ...]


@dataclass(frozen=True)
class BenchmarkSummary:
    server_lifetime_seconds: float
    prompt_tokens_per_second: float
    output_tokens_per_second: float
    total_tokens_per_second: float
    prompt_token_count: int
    output_token_count: int
    total_token_count: int
    completed_requests: int
    saturated_at_deadline: bool
    time_to_first_token_seconds: MetricDistribution
    inter_token_latency_seconds: MetricDistribution
    completed_request_latency_seconds: MetricDistribution
    prefill_latency_seconds: MetricDistribution


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PackageVersion:
    package: str
    version: str


@dataclass(frozen=True)
class SourceControlProvenance:
    commit: str
    status: tuple[str, ...]
    patch_path: str
    patch_sha256: str
    untracked_files: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class EnvironmentVariable:
    name: str
    value: str


@dataclass(frozen=True)
class EnvironmentProvenance:
    python_version: str
    packages: tuple[PackageVersion, ...]
    gpu_information: tuple[str, ...]
    runtime_environment: tuple[EnvironmentVariable, ...]
    source_control: SourceControlProvenance
    model_files: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class BenchmarkConfiguration:
    model_path: str
    corpus_path: str
    corpus_sha256: str
    request_count: int
    seed: int
    max_completion_tokens: int
    duration_seconds: float
    arrival_batch_size: int
    arrival_interval_seconds: float
    enable_thinking: bool
    host: str
    port: int
    server: "ServerConfiguration"


@dataclass(frozen=True)
class ServerConfiguration:
    page_size: int
    total_pages: int
    slots: int
    max_decode_batch_size: int
    prefill_batch_size: int
    prefill_chunk_size: int
    decode_steps_per_prefill: int
    decode_block_size: int


@dataclass(frozen=True)
class BenchmarkTiming:
    server_ready_at_seconds: float
    batch_accepted_at_seconds: float
    last_batch_accepted_at_seconds: float
    metrics_requested_at_seconds: float
    metrics_received_at_seconds: float


@dataclass(frozen=True)
class MeasurementSemantics:
    throughput: str
    time_to_first_token: str
    inter_token_latency: str


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: int
    configuration: BenchmarkConfiguration
    environment: EnvironmentProvenance
    timing: BenchmarkTiming
    measurement_semantics: MeasurementSemantics
    batch: BatchStatus
    metrics: BenchmarkMetricsSnapshot
    summary: BenchmarkSummary


@dataclass(frozen=True)
class BenchmarkOptions:
    model_path: Path
    output_dir: Path
    corpus_path: Path | None
    request_count: int
    seed: int
    max_completion_tokens: int
    duration_seconds: float
    enable_thinking: bool
    host: str
    port: int
    dataset_revision: str
    server_start_timeout_seconds: float
    require_saturated: bool
    arrival_batch_size: int | None = None
    arrival_interval_seconds: float = 0.0
    page_size: int = 32
    total_pages: int = 16_384
    slots: int = 64
    max_decode_batch_size: int = 64
    prefill_batch_size: int = 32
    prefill_chunk_size: int = 128
    decode_steps_per_prefill: int = 8
    decode_block_size: int = 512

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> Self:
        return cls(**vars(namespace))


def _build_converter() -> cattrs.Converter:
    converter = cattrs.Converter()
    converter.register_structure_hook(
        OpenHermesTurn,
        make_dict_structure_fn(
            OpenHermesTurn,
            converter,
            speaker=override(rename="from"),
        ),
    )
    return converter


CONVERTER = _build_converter()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _fingerprint_file(path: Path, relative_to: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return FileFingerprint(
        path=str(path.relative_to(relative_to)),
        size_bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(CONVERTER.unstructure(value), indent=2, sort_keys=True) + "\n").encode()


def save_json(path: Path, value: object) -> str:
    contents = _json_bytes(value)
    path.write_bytes(contents)
    return _sha256_bytes(contents)


def load_corpus(path: Path) -> BenchmarkCorpus:
    corpus = CONVERTER.structure(json.loads(path.read_text()), BenchmarkCorpus)
    if corpus.schema_version != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported corpus schema version: {corpus.schema_version}")
    return corpus


def _normalize_openhermes_row(row: DatasetViewerRow) -> CorpusRequest | None:
    if row.truncated_cells:
        raise ValueError(f"Dataset viewer truncated row {row.row_idx}: {row.truncated_cells}")

    conversations = row.row.conversations
    final_user_index = next(
        (
            index
            for index in range(len(conversations) - 1, -1, -1)
            if conversations[index].speaker is OpenHermesSpeaker.HUMAN
        ),
        None,
    )
    if final_user_index is None:
        return None

    def message_role(speaker: OpenHermesSpeaker) -> MessageRole:
        match speaker:
            case OpenHermesSpeaker.SYSTEM:
                return MessageRole.SYSTEM
            case OpenHermesSpeaker.HUMAN:
                return MessageRole.USER
            case OpenHermesSpeaker.GPT:
                return MessageRole.ASSISTANT

    messages = tuple(
        BenchmarkMessage(role=message_role(turn.speaker), content=turn.value)
        for turn in conversations[: final_user_index + 1]
        if turn.value
    )
    if not messages or messages[-1].role is not MessageRole.USER:
        return None
    return CorpusRequest(
        sequence_id=f"openhermes_{row.row_idx:07d}",
        dataset_row_index=row.row_idx,
        messages=messages,
    )


def _sample_page_indices(total_rows: int, request_count: int, seed: int) -> tuple[int, ...]:
    if total_rows < 1:
        raise ValueError("The dataset must contain at least one row.")
    if request_count < 1:
        raise ValueError("request_count must be at least one.")
    page_count = math.ceil(total_rows / DATASET_PAGE_SIZE)
    return tuple(random.Random(seed).sample(range(page_count), page_count))


def _fetch_rows(
    session: requests.Session,
    *,
    offset: int,
    length: int,
    expected_revision: str,
) -> DatasetRowsResponse:
    response: requests.Response | None = None
    for attempt in range(5):
        try:
            response = session.get(
                DATASET_ROWS_URL,
                params={
                    "dataset": OPENHERMES_DATASET,
                    "config": DATASET_CONFIG,
                    "split": DATASET_SPLIT,
                    "offset": offset,
                    "length": length,
                },
                timeout=60,
            )
            if response.status_code < 500:
                break
        except requests.RequestException:
            if attempt == 4:
                raise
        if attempt == 4:
            assert response is not None
            response.raise_for_status()
        time.sleep(2**attempt)
    assert response is not None
    response.raise_for_status()
    actual_revision = response.headers.get("x-revision")
    if actual_revision != expected_revision:
        raise ValueError(f"Expected OpenHermes revision {expected_revision}, got {actual_revision}")
    rows_response = CONVERTER.structure(response.json(), DatasetRowsResponse)
    if rows_response.partial:
        raise ValueError("Dataset viewer returned a partial result.")
    return rows_response


def materialize_corpus(
    *,
    request_count: int,
    seed: int,
    revision: str,
) -> BenchmarkCorpus:
    with requests.Session() as session:
        first_page = _fetch_rows(session, offset=0, length=1, expected_revision=revision)
        page_indices = _sample_page_indices(first_page.num_rows_total, request_count, seed)
        requests_by_row: dict[int, CorpusRequest] = {}
        for page_index in page_indices:
            page_offset = page_index * DATASET_PAGE_SIZE
            page_length = min(DATASET_PAGE_SIZE, first_page.num_rows_total - page_offset)
            page = _fetch_rows(
                session,
                offset=page_offset,
                length=page_length,
                expected_revision=revision,
            )
            for dataset_row in page.rows:
                request = _normalize_openhermes_row(dataset_row)
                if request is not None:
                    requests_by_row[request.dataset_row_index] = request
            if len(requests_by_row) >= request_count:
                break

    sampled_requests = tuple(requests_by_row.values())[:request_count]
    if len(sampled_requests) != request_count:
        raise ValueError(
            f"Only {len(sampled_requests)} usable requests were found in the deterministic page sample; "
            f"requested {request_count}."
        )
    return BenchmarkCorpus(
        schema_version=CORPUS_SCHEMA_VERSION,
        dataset=OPENHERMES_DATASET,
        revision=revision,
        config=DATASET_CONFIG,
        split=DATASET_SPLIT,
        seed=seed,
        total_dataset_rows=first_page.num_rows_total,
        requests=sampled_requests,
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _distribution(values: Iterable[float]) -> MetricDistribution:
    sorted_values = tuple(sorted(values))
    if not sorted_values:
        return MetricDistribution(count=0, mean=None, minimum=None, maximum=None, percentiles=())
    return MetricDistribution(
        count=len(sorted_values),
        mean=fmean(sorted_values),
        minimum=sorted_values[0],
        maximum=sorted_values[-1],
        percentiles=tuple(
            PercentileValue(percentile=percentile, value=_percentile(sorted_values, percentile))
            for percentile in PERCENTILES
        ),
    )


def _inter_token_latencies(metrics: BenchmarkMetricsSnapshot) -> Iterable[float]:
    for sample in metrics.inter_token_latency_samples:
        yield from (sample.latency_seconds for _ in range(sample.sample_count))


def summarize_metrics(
    metrics: BenchmarkMetricsSnapshot,
    *,
    server_lifetime_seconds: float,
    saturated_at_deadline: bool,
) -> BenchmarkSummary:
    if server_lifetime_seconds <= 0.0:
        raise ValueError("server_lifetime_seconds must be positive.")
    total_token_count = metrics.prompt_token_count + metrics.output_token_count
    return BenchmarkSummary(
        server_lifetime_seconds=server_lifetime_seconds,
        prompt_tokens_per_second=metrics.prompt_token_count / server_lifetime_seconds,
        output_tokens_per_second=metrics.output_token_count / server_lifetime_seconds,
        total_tokens_per_second=total_token_count / server_lifetime_seconds,
        prompt_token_count=metrics.prompt_token_count,
        output_token_count=metrics.output_token_count,
        total_token_count=total_token_count,
        completed_requests=metrics.completed_requests,
        saturated_at_deadline=saturated_at_deadline,
        time_to_first_token_seconds=_distribution(
            request.first_token_at_seconds
            for request in metrics.requests
            if request.first_token_at_seconds is not None
        ),
        inter_token_latency_seconds=_distribution(_inter_token_latencies(metrics)),
        completed_request_latency_seconds=_distribution(
            request.completed_at_seconds for request in metrics.requests if request.completed_at_seconds is not None
        ),
        prefill_latency_seconds=_distribution(
            request.prefill_duration_seconds
            for request in metrics.requests
            if request.prefill_duration_seconds is not None
        ),
    )


def _run_command(arguments: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(arguments, cwd=cwd, check=True, capture_output=True)


def _source_control_provenance(repository: Path, patch_path: Path) -> SourceControlProvenance:
    commit = _run_command(("git", "rev-parse", "HEAD"), cwd=repository).stdout.decode().strip()
    status_lines = tuple(
        line for line in _run_command(("git", "status", "--porcelain=v1"), cwd=repository).stdout.decode().splitlines()
    )
    tracked_diff = _run_command(("git", "diff", "--binary", "HEAD"), cwd=repository).stdout
    untracked_output = _run_command(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repository,
    ).stdout
    untracked_paths = tuple(
        repository / relative_path.decode() for relative_path in untracked_output.split(b"\0") if relative_path
    )
    untracked_patches = []
    for path in untracked_paths:
        relative_path = path.relative_to(repository)
        untracked_diff = subprocess.run(
            ("git", "diff", "--no-index", "--binary", "--", "/dev/null", str(relative_path)),
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if untracked_diff.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                untracked_diff.returncode,
                untracked_diff.args,
                untracked_diff.stdout,
                untracked_diff.stderr,
            )
        untracked_patches.append(untracked_diff.stdout)
    source_patch = b"".join((tracked_diff, *untracked_patches))
    patch_path.write_bytes(source_patch)
    return SourceControlProvenance(
        commit=commit,
        status=status_lines,
        patch_path=str(patch_path.resolve()),
        patch_sha256=_sha256_bytes(source_patch),
        untracked_files=tuple(
            _fingerprint_file(path, repository) for path in sorted(untracked_paths) if path.is_file()
        ),
    )


def _environment_provenance(
    repository: Path,
    model_path: Path,
    source_patch_path: Path,
    runtime_environment_values: Mapping[str, str],
) -> EnvironmentProvenance:
    packages = []
    for package in ("cattrs", "equinox", "jax", "jaxlib", "lalamo", "requests"):
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "not-installed"
        packages.append(PackageVersion(package=package, version=package_version))

    gpu_process = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_information = tuple(line for line in gpu_process.stdout.splitlines() if line)
    runtime_environment = tuple(
        EnvironmentVariable(name=name, value=value)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "JAX_DEFAULT_MATMUL_PRECISION",
            "JAX_EXEC_TIME_OPTIMIZATION_EFFORT",
            "JAX_MEMORY_FITTING_EFFORT",
            "JAX_PLATFORMS",
            "JAX_SCAN3",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
        if (value := runtime_environment_values.get(name)) is not None
    )
    model_files = tuple(
        _fingerprint_file(path, model_path) for path in sorted(model_path.rglob("*")) if path.is_file()
    )
    return EnvironmentProvenance(
        python_version=sys.version,
        packages=tuple(packages),
        gpu_information=gpu_information,
        runtime_environment=runtime_environment,
        source_control=_source_control_provenance(repository, source_patch_path),
        model_files=model_files,
    )


def _server_request_body(corpus: BenchmarkCorpus, options: BenchmarkOptions) -> tuple[ServerRequest, ...]:
    return tuple(
        ServerRequest(
            sequence_id=request.sequence_id,
            messages=request.messages,
            model=str(options.model_path.resolve()),
            max_completion_tokens=options.max_completion_tokens,
            generation_config=GreedyGenerationConfig(),
            enable_thinking=options.enable_thinking,
        )
        for request in corpus.requests
    )


def _wait_for_server(
    session: requests.Session,
    server_url: str,
    process: subprocess.Popen[bytes],
    deadline: float,
) -> None:
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode} before becoming ready.")
        try:
            response = session.get(f"{server_url}/openapi.json", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise TimeoutError("Server did not become ready before the startup timeout.")


def _wait_until_deadline(process: subprocess.Popen[bytes], deadline: float) -> None:
    while True:
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0.0:
            return
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}.")
        time.sleep(min(remaining_seconds, 0.5))


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _server_arguments(options: BenchmarkOptions) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "lalamo.main",
        "server",
        str(options.model_path.resolve()),
        "--host",
        options.host,
        "--port",
        str(options.port),
        "--page-size",
        str(options.page_size),
        "--total-pages",
        str(options.total_pages),
        "--slots",
        str(options.slots),
        "--max-decode-batch-size",
        str(options.max_decode_batch_size),
        "--prefill-batch-size",
        str(options.prefill_batch_size),
        "--prefill-chunk-size",
        str(options.prefill_chunk_size),
        "--decode-steps-per-prefill",
        str(options.decode_steps_per_prefill),
        "--decode-block-size",
        str(options.decode_block_size),
        "--benchmark-metrics",
    )


def _measurement_semantics() -> MeasurementSemantics:
    return MeasurementSemantics(
        throughput=(
            "Token counts at the metrics snapshot divided by wall time from server process spawn through "
            "receipt of that snapshot; this includes model load, compilation, and HTTP startup overhead."
        ),
        time_to_first_token="Elapsed time from each request's admission through its first completed decode step.",
        inter_token_latency="Wall time per individual continuous decode scheduling step, token-weighted.",
    )


def run_benchmark(options: BenchmarkOptions) -> BenchmarkResult:
    if options.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {options.output_dir}")
    if not options.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {options.model_path}")
    if options.duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if options.request_count < 1:
        raise ValueError("request_count must be at least one.")
    if options.arrival_batch_size is not None and options.arrival_batch_size < 1:
        raise ValueError("arrival_batch_size must be at least one.")
    if options.arrival_interval_seconds < 0.0:
        raise ValueError("arrival_interval_seconds cannot be negative.")

    options.output_dir.mkdir(parents=True)
    corpus_output_path = options.output_dir / "corpus.json"
    if options.corpus_path is None:
        corpus = materialize_corpus(
            request_count=options.request_count,
            seed=options.seed,
            revision=options.dataset_revision,
        )
    else:
        corpus = load_corpus(options.corpus_path)
        if corpus.revision != options.dataset_revision:
            raise ValueError(f"Corpus revision is {corpus.revision}, expected {options.dataset_revision}")
        if corpus.seed != options.seed:
            raise ValueError(f"Corpus seed is {corpus.seed}, expected {options.seed}")
        if len(corpus.requests) < options.request_count:
            raise ValueError(f"Corpus contains {len(corpus.requests)} requests; need {options.request_count}")
        corpus = BenchmarkCorpus(
            schema_version=corpus.schema_version,
            dataset=corpus.dataset,
            revision=corpus.revision,
            config=corpus.config,
            split=corpus.split,
            seed=corpus.seed,
            total_dataset_rows=corpus.total_dataset_rows,
            requests=corpus.requests[: options.request_count],
        )
    corpus_sha256 = save_json(corpus_output_path, corpus)

    repository = Path(__file__).resolve().parents[1]
    server_url = f"http://{options.host}:{options.port}"
    server_log_path = options.output_dir / "server.log"
    server_arguments = _server_arguments(options)
    server_environment = os.environ.copy()
    server_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    environment = _environment_provenance(
        repository,
        options.model_path.resolve(),
        options.output_dir / "source.patch",
        server_environment,
    )

    with server_log_path.open("wb") as server_log, requests.Session() as session:
        benchmark_started_at = time.perf_counter()
        process = subprocess.Popen(
            server_arguments,
            cwd=repository,
            env=server_environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(
                session,
                server_url,
                process,
                benchmark_started_at + options.server_start_timeout_seconds,
            )
            server_ready_at_seconds = time.perf_counter() - benchmark_started_at
            server_requests = _server_request_body(corpus, options)
            arrival_batch_size = options.arrival_batch_size or len(server_requests)
            request_batches = tuple(
                server_requests[start : start + arrival_batch_size]
                for start in range(0, len(server_requests), arrival_batch_size)
            )
            first_request_batch, *later_request_batches = request_batches
            last_arrival_seconds = (len(request_batches) - 1) * options.arrival_interval_seconds
            if last_arrival_seconds >= options.duration_seconds - server_ready_at_seconds:
                raise ValueError("The final request batch must arrive before the benchmark deadline.")

            first_admission_started_at = time.perf_counter()
            batch_response = session.post(
                f"{server_url}/batches",
                json=CONVERTER.unstructure(first_request_batch),
                timeout=60,
            )
            batch_response.raise_for_status()
            batches = [CONVERTER.structure(batch_response.json(), BatchStatus)]
            batch_accepted_at_seconds = time.perf_counter() - benchmark_started_at
            for batch_index, request_batch in enumerate(later_request_batches, start=1):
                _wait_until_deadline(
                    process,
                    first_admission_started_at + batch_index * options.arrival_interval_seconds,
                )
                batch_response = session.post(
                    f"{server_url}/batches",
                    json=CONVERTER.unstructure(request_batch),
                    timeout=60,
                )
                batch_response.raise_for_status()
                batches.append(CONVERTER.structure(batch_response.json(), BatchStatus))
            last_batch_accepted_at_seconds = time.perf_counter() - benchmark_started_at

            benchmark_deadline = benchmark_started_at + options.duration_seconds
            _wait_until_deadline(process, benchmark_deadline)
            metrics_requested_at_seconds = time.perf_counter() - benchmark_started_at
            metrics_response = session.get(f"{server_url}/benchmark-metrics", timeout=60)
            metrics_response.raise_for_status()
            metrics = CONVERTER.structure(metrics_response.json(), BenchmarkMetricsSnapshot)
            metrics_received_at_seconds = time.perf_counter() - benchmark_started_at

            final_batches = []
            for batch in batches:
                batch_response = session.get(f"{server_url}/batches/{batch.id}", timeout=60)
                batch_response.raise_for_status()
                final_batches.append(CONVERTER.structure(batch_response.json(), BatchStatus))
        finally:
            _stop_process(process)

    failed_batches = tuple(batch for batch in final_batches if batch.status == "failed")
    if failed_batches:
        raise RuntimeError(f"Server batches failed: {tuple(batch.error for batch in failed_batches)}")
    completed_count = sum(batch.completed for batch in final_batches)
    batch = BatchStatus(
        id=f"{len(final_batches)}_batches",
        total=options.request_count,
        completed=completed_count,
        status="completed" if completed_count == options.request_count else "in_progress",
    )
    saturated_at_deadline = metrics.completed_requests + options.max_decode_batch_size <= options.request_count
    summary = summarize_metrics(
        metrics,
        server_lifetime_seconds=metrics_received_at_seconds,
        saturated_at_deadline=saturated_at_deadline,
    )
    result = BenchmarkResult(
        schema_version=RESULT_SCHEMA_VERSION,
        configuration=BenchmarkConfiguration(
            model_path=str(options.model_path.resolve()),
            corpus_path=str(corpus_output_path.resolve()),
            corpus_sha256=corpus_sha256,
            request_count=options.request_count,
            seed=options.seed,
            max_completion_tokens=options.max_completion_tokens,
            duration_seconds=options.duration_seconds,
            arrival_batch_size=options.arrival_batch_size or options.request_count,
            arrival_interval_seconds=options.arrival_interval_seconds,
            enable_thinking=options.enable_thinking,
            host=options.host,
            port=options.port,
            server=ServerConfiguration(
                page_size=options.page_size,
                total_pages=options.total_pages,
                slots=options.slots,
                max_decode_batch_size=options.max_decode_batch_size,
                prefill_batch_size=options.prefill_batch_size,
                prefill_chunk_size=options.prefill_chunk_size,
                decode_steps_per_prefill=options.decode_steps_per_prefill,
                decode_block_size=options.decode_block_size,
            ),
        ),
        environment=environment,
        timing=BenchmarkTiming(
            server_ready_at_seconds=server_ready_at_seconds,
            batch_accepted_at_seconds=batch_accepted_at_seconds,
            last_batch_accepted_at_seconds=last_batch_accepted_at_seconds,
            metrics_requested_at_seconds=metrics_requested_at_seconds,
            metrics_received_at_seconds=metrics_received_at_seconds,
        ),
        measurement_semantics=_measurement_semantics(),
        batch=batch,
        metrics=metrics,
        summary=summary,
    )
    save_json(options.output_dir / "result.json", result)
    if options.require_saturated and not saturated_at_deadline:
        raise RuntimeError("The request queue drained before the benchmark deadline; increase --request-count.")
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a Lalamo server on OpenHermes.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--request-count", type=int, default=DEFAULT_REQUEST_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dataset-revision", default=OPENHERMES_REVISION)
    parser.add_argument("--server-start-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--require-saturated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arrival-batch-size", type=int)
    parser.add_argument("--arrival-interval-seconds", type=float, default=0.0)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--total-pages", type=int, default=16_384)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--max-decode-batch-size", type=int, default=64)
    parser.add_argument("--prefill-batch-size", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--decode-steps-per-prefill", type=int, default=8)
    parser.add_argument("--decode-block-size", type=int, default=512)
    return parser


def main() -> None:
    options = BenchmarkOptions.from_namespace(_build_argument_parser().parse_args())
    result = run_benchmark(options)
    print(json.dumps(CONVERTER.unstructure(result.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
