from pathlib import Path

from benchmarks.server import (
    BenchmarkCorpus,
    BenchmarkMessage,
    BenchmarkMetricsSnapshot,
    BenchmarkOptions,
    CorpusRequest,
    DatasetViewerRow,
    MessageRole,
    OpenHermesRow,
    OpenHermesSpeaker,
    OpenHermesTurn,
    RequestBenchmarkMetrics,
    WeightedLatencySample,
    _json_bytes,
    _normalize_openhermes_row,
    _sample_page_indices,
    _server_arguments,
    _server_request_body,
    summarize_metrics,
)


def test_normalize_openhermes_row_keeps_context_through_final_user_turn() -> None:
    request = _normalize_openhermes_row(
        DatasetViewerRow(
            row_idx=42,
            row=OpenHermesRow(
                conversations=(
                    OpenHermesTurn(OpenHermesSpeaker.SYSTEM, "system"),
                    OpenHermesTurn(OpenHermesSpeaker.HUMAN, "first"),
                    OpenHermesTurn(OpenHermesSpeaker.GPT, "answer"),
                    OpenHermesTurn(OpenHermesSpeaker.HUMAN, "second"),
                    OpenHermesTurn(OpenHermesSpeaker.GPT, "target answer"),
                ),
            ),
            truncated_cells=(),
        ),
    )

    assert request is not None
    assert request.sequence_id == "openhermes_0000042"
    assert tuple(message.role for message in request.messages) == (
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    )
    assert request.messages[-1].content == "second"


def test_page_sample_is_deterministic_and_unique() -> None:
    first = _sample_page_indices(total_rows=1_001_551, request_count=1024, seed=1337)
    second = _sample_page_indices(total_rows=1_001_551, request_count=1024, seed=1337)

    assert first == second
    assert len(first) == 10_016
    assert len(first) == len(set(first))


def test_server_request_serialization_is_greedy_and_uses_server_field_names() -> None:
    corpus = BenchmarkCorpus(
        schema_version=1,
        dataset="dataset",
        revision="revision",
        config="default",
        split="train",
        seed=1337,
        total_dataset_rows=1,
        requests=(
            CorpusRequest(
                sequence_id="request",
                dataset_row_index=0,
                messages=(BenchmarkMessage(role=MessageRole.USER, content="hello"),),
            ),
        ),
    )
    options = BenchmarkOptions(
        model_path=Path("model"),
        output_dir=Path("output"),
        corpus_path=None,
        request_count=1,
        seed=10,
        max_completion_tokens=80_000,
        duration_seconds=60.0,
        enable_thinking=True,
        host="127.0.0.1",
        port=8293,
        dataset_revision="revision",
        server_start_timeout_seconds=600.0,
        require_saturated=True,
    )

    request_json = _json_bytes(_server_request_body(corpus, options)).decode()

    assert '"role": "user"' in request_json
    assert '"temperature": 0.0' in request_json
    assert '"max_completion_tokens": 80000' in request_json


def test_server_command_carries_scheduler_configuration() -> None:
    options = BenchmarkOptions(
        model_path=Path("model"),
        output_dir=Path("output"),
        corpus_path=None,
        request_count=1,
        seed=10,
        max_completion_tokens=80_000,
        duration_seconds=60.0,
        enable_thinking=True,
        host="127.0.0.1",
        port=8294,
        dataset_revision="revision",
        server_start_timeout_seconds=600.0,
        require_saturated=True,
        page_size=32,
        total_pages=1024,
        slots=16,
        max_decode_batch_size=8,
        prefill_batch_size=2,
        prefill_chunk_size=64,
        decode_steps_per_prefill=4,
        decode_block_size=2,
    )

    arguments = _server_arguments(options)

    assert "server" in arguments
    assert arguments[arguments.index("--page-size") + 1] == "32"
    assert arguments[arguments.index("--total-pages") + 1] == "1024"
    assert arguments[arguments.index("--max-decode-batch-size") + 1] == "8"
    assert arguments[arguments.index("--decode-block-size") + 1] == "2"


def test_summarize_metrics_uses_server_lifetime_and_token_weighted_decode_intervals() -> None:
    metrics = BenchmarkMetricsSnapshot(
        batch_id="batch_test",
        elapsed_seconds=8.0,
        prompt_token_count=30,
        output_token_count=10,
        completed_requests=1,
        requests=(
            RequestBenchmarkMetrics(
                sequence_id="first",
                prompt_token_count=10,
                output_token_count=6,
                prefill_duration_seconds=1.0,
                first_token_at_seconds=2.0,
                completed_at_seconds=7.0,
            ),
            RequestBenchmarkMetrics(
                sequence_id="second",
                prompt_token_count=20,
                output_token_count=4,
                prefill_duration_seconds=3.0,
                first_token_at_seconds=4.0,
            ),
        ),
        inter_token_latency_samples=(
            WeightedLatencySample(sample_count=4, latency_seconds=0.2),
            WeightedLatencySample(sample_count=2, latency_seconds=0.4),
            WeightedLatencySample(sample_count=2, latency_seconds=0.3),
        ),
    )

    summary = summarize_metrics(metrics, server_lifetime_seconds=10.0, saturated_at_deadline=True)

    assert summary.prompt_tokens_per_second == 3.0
    assert summary.output_tokens_per_second == 1.0
    assert summary.total_tokens_per_second == 4.0
    assert summary.completed_requests == 1
    assert summary.saturated_at_deadline
    assert summary.time_to_first_token_seconds.count == 2
    assert summary.time_to_first_token_seconds.mean == 3.0
    assert summary.inter_token_latency_seconds.count == 8
    assert summary.inter_token_latency_seconds.mean == 0.275
    assert summary.completed_request_latency_seconds.count == 1
    assert summary.prefill_latency_seconds.mean == 2.0


def test_summarize_metrics_handles_compact_weighted_inter_token_samples() -> None:
    metrics = BenchmarkMetricsSnapshot(
        batch_id="batch_test",
        elapsed_seconds=8.0,
        prompt_token_count=3,
        output_token_count=5,
        completed_requests=0,
        requests=(RequestBenchmarkMetrics(sequence_id="request", output_token_count=5),),
        inter_token_latency_samples=(
            WeightedLatencySample(sample_count=3, latency_seconds=0.25),
            WeightedLatencySample(sample_count=1, latency_seconds=0.5),
        ),
    )

    summary = summarize_metrics(metrics, server_lifetime_seconds=10.0, saturated_at_deadline=True)

    assert summary.inter_token_latency_seconds.count == 4
    assert summary.inter_token_latency_seconds.mean == 0.3125
